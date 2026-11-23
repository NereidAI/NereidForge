# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NereidForge optimization entrypoint (quantization + graph opt + benchmark).

Takes a source ONNX model from seaweedfs and produces an optimized artifact
for one target device:

  quant_method:
    graph        ONNX Runtime graph optimization (ORT_ENABLE_ALL)
    fp16         float16 conversion (onnxconverter-common)
    int8-dynamic onnxruntime.quantization.quantize_dynamic (CPU/ROCm only)
    int8-static  OpenVINO target: NNCF static PTQ with accuracy-aware layer
                 revert (quantize_with_accuracy_control, mAP50 gate);
                 ONNX targets: onnxruntime quantize_static with a real
                 calibration set (calib_uri + calib_n images)

  target_device:
    cpu          artifact stays .onnx, benchmarked via onnxruntime CPU EP
    openvino     ONNX FP32 -> OV IR via ovc -> static reshape [1,3,S,S]:
                   fp16        -> save_model(compress_to_fp16=True)
                   int8-static -> NNCF accuracy-aware static quantization
                 ORT QDQ graphs are never converted to IR on this target
                 (the GPU plugin compiles them to all-NaN outputs, pit 178);
                 benchmarked via openvino.runtime CPU plugin
    tensorrt     requires a GPU node + trtexec; fails fast with a clear
                 message when unavailable (code path ready for GPU clusters)

Benchmark is standardized: sequential single-image inference, warmup 10,
measure 500 (configurable), reporting avg/p50/p95/p99 + qps for both source
and optimized models, plus the runtime environment for reproducibility.

Env contract:
  INPUT_URI    s3://... source model (.onnx)
  OUTPUT_DIR   s3://... artifact dir ({ARTIFACT_ID}.optimized.onnx etc.)
  ARTIFACT_ID  unique id (job id)
  CONFIG_JSON  optimization config

Exit codes: 0 ok.
"""

import json
import os
import platform
import socket
import sys
import time

from common.s3_utils import make_s3_client as s3_client


def parse_uri(uri):
    """Split an s3://bucket/key URI into (bucket, key-without-trailing-slash)."""
    rest = uri[len("s3://"):]
    parts = rest.split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "").rstrip("/")


def download(s3, uri, local):
    """Download one S3 object to a local path with a traced log line."""
    bucket, key = parse_uri(uri)
    s3.download_file(bucket, key, local)
    print(f"[opt] downloaded {uri} -> {local}", flush=True)


def upload(s3, local, uri):
    """Upload one local file to an S3 URI with a traced log line."""
    bucket, key = parse_uri(uri)
    s3.upload_file(local, bucket, key)
    print(f"[opt] uploaded {uri}", flush=True)


def env_block() -> dict:
    """Record the benchmark environment for reproducibility."""
    return {
        "hostname": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "threads": int(os.cpu_count() or 1),
        "python": platform.python_version(),
    }


def calibration_reader(s3, calib_uri, imgsz, n):
    """Yield n preprocessed calibration images from the dataset prefix."""
    import cv2
    import numpy as np
    from onnxruntime.quantization import CalibrationDataReader

    bucket, prefix = parse_uri(calib_uri)
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                keys.append(k)
    keys = sorted(keys)[:n]
    print(f"[opt] calibration images: {len(keys)}", flush=True)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.keys = list(keys)

        def get_next(self):
            if not self.keys:
                return None
            k = self.keys.pop(0)
            try:
                s3.download_file(bucket, k, "/tmp/calib.img")
                img = cv2.imread("/tmp/calib.img")
                if img is None:
                    return self.get_next()
                img = cv2.resize(img, (imgsz, imgsz))
                arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                return {"images": arr.transpose(2, 0, 1)[None]}
            except Exception as e:  # noqa: BLE001
                print(f"[opt] calib skip {k}: {e}", flush=True)
                return self.get_next()

    return Reader()


# ---------------------------------------------------------------------------
# NNCF static PTQ for the OpenVINO edge path (solution D2/D3, gate-1)
# ---------------------------------------------------------------------------

def _letterbox(path, imgsz):
    """Letterbox an image exactly like evaluator/packager do; returns the
    NCHW FP32 RGB/255 tensor plus unmap metadata (w0, h0, scale, pad_x, pad_y)."""
    import cv2
    import numpy as np

    img = cv2.imread(path)
    if img is None:
        return None
    h0, w0 = img.shape[:2]
    r = min(imgsz / w0, imgsz / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    x = canvas[:, :, ::-1].astype("float32").transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), w0, h0, r, pad_x, pad_y


def coco_entries(s3, dataset_uri, limit):
    """Read up to `limit` annotated entries from a canonical COCO dataset.

    Returns (entries, class_names); each entry is {bucket, key, image_id,
    boxes{cls: [[x1,y1,x2,y2], ...]}} in SOURCE-pixel xyxy (COCO bbox space).
    """
    import numpy as np

    bucket, prefix = parse_uri(dataset_uri)
    prefix = prefix.rstrip("/")
    ann = json.loads(
        s3.get_object(Bucket=bucket, Key=f"{prefix}/annotations.json")["Body"].read()
    )
    cats = sorted(ann.get("categories", []) or [], key=lambda c: int(c.get("id", 0)))
    names = [str(c.get("name") or "") for c in cats]
    cat_index = {int(c["id"]): i for i, c in enumerate(cats)}
    boxes_by_img: dict = {}
    for a in ann.get("annotations", []) or []:
        if a.get("iscrowd") in (1, "1"):
            continue
        b = a.get("bbox") or []
        if not isinstance(b, list) or len(b) < 4 or max(b[:4]) <= 0:
            continue
        cls = cat_index.get(int(a.get("category_id", -1)))
        if cls is None:
            continue
        x, y, w, h = b[:4]
        boxes_by_img.setdefault(int(a["image_id"]), {}).setdefault(cls, []).append(
            [float(x), float(y), float(x) + float(w), float(y) + float(h)]
        )
    entries = []
    for img in ann.get("images", []) or []:
        iid = int(img.get("id", -1))
        if iid not in boxes_by_img:
            continue
        fn = str(img.get("file_name") or "").lstrip("/")
        if fn.startswith("s3://"):
            b, k = fn[len("s3://"):].split("/", 1)
            if b != bucket:
                continue
            key = k
        elif fn.startswith(("http://", "https://")):
            continue
        elif fn.startswith(("images/", "img/", "data/")):
            key = f"{prefix}/{fn}"
        else:
            key = f"{prefix}/images/{fn}"
        entries.append({
            "bucket": bucket,
            "key": key,
            "image_id": iid,
            "boxes": {c: np.asarray(v, dtype=np.float64)
                      for c, v in boxes_by_img[iid].items()},
        })
        if len(entries) >= limit:
            break
    return entries, names


def _dataset_relative_keys(s3, bucket, prefix):
    """_dataset_relative_keys lists sorted version-relative keys across S3 pages."""
    base = f"{prefix}/" if prefix else ""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=base):
        for obj in page.get("Contents", []) or []:
            key = str(obj.get("Key") or "")
            if key.startswith(base):
                keys.append(key[len(base):])
    return sorted(keys)


def _yolo_class_names(raw):
    """_yolo_class_names preserves canonical list/dict data.yaml class order."""
    import yaml

    parsed = yaml.safe_load(raw) or {}
    if not isinstance(parsed, dict):
        return []
    names = parsed.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    nc = parsed.get("nc")
    return [f"class{i}" for i in range(nc)] if isinstance(nc, int) and nc > 0 else []


def _yolo_split(relative_key):
    """_yolo_split maps canonical split aliases to calibration/validation groups."""
    first = relative_key.split("/", 1)[0].lower()
    if first == "train":
        return "train"
    if first in ("val", "valid", "validation", "dev"):
        return "val"
    if first in ("test", "eval"):
        return "test"
    return "flat"


def _sequential_calib_val(entries, calib_n, val_n):
    """_sequential_calib_val retains the legacy split for COCO and flat YOLO."""
    calib = entries[:calib_n]
    val = entries[calib_n:calib_n + val_n]
    if not val:
        val = entries[:min(20, len(entries))]
    return calib, val[:val_n]


def _yolo_calib_val(s3, bucket, prefix, relative_keys, names, calib_n, val_n):
    """_yolo_calib_val converts sibling YOLO labels into source-pixel entries.

    Train images calibrate and val images validate whenever those splits exist;
    test is only a validation fallback. Flat images/labels retain sequential
    splitting. Label paths are derived from each image's own split, so identical
    stems in train/val cannot collide.
    """
    import cv2
    import numpy as np

    key_set = set(relative_keys)
    groups = {"train": [], "val": [], "test": [], "flat": []}
    for rel in relative_keys:
        if not rel.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        if rel.startswith("images/"):
            label_rel = "labels/" + rel[len("images/"):]
        elif "/images/" in rel:
            label_rel = rel.replace("/images/", "/labels/", 1)
        else:
            continue
        label_rel = label_rel.rsplit(".", 1)[0] + ".txt"
        if label_rel not in key_set:
            continue
        split = _yolo_split(rel)
        group = groups[split]
        if ((split == "train" and len(group) >= calib_n)
                or (split in ("val", "test") and len(group) >= val_n)
                or (split == "flat" and len(group) >= calib_n + val_n)):
            continue
        full_label_key = f"{prefix}/{label_rel}" if prefix else label_rel
        try:
            lines = s3.get_object(Bucket=bucket, Key=full_label_key)["Body"].read().decode().splitlines()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        records = []
        for line in lines:
            fields = line.split()
            if len(fields) != 5:
                continue
            try:
                cls = int(fields[0])
                values = [float(value) for value in fields[1:]]
            except ValueError:
                continue
            if cls < 0 or cls >= len(names) or not np.isfinite(values).all():
                continue
            cx, cy, width, height = values
            if width > 0 and height > 0:
                records.append((cls, cx, cy, width, height))
        if not records:
            continue
        image_key = f"{prefix}/{rel}" if prefix else rel
        try:
            raw_image = s3.get_object(Bucket=bucket, Key=image_key)["Body"].read()
        except FileNotFoundError:
            continue
        image = cv2.imdecode(np.frombuffer(raw_image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        source_h, source_w = image.shape[:2]
        boxes: dict = {}
        for cls, cx, cy, width, height in records:
            boxes.setdefault(cls, []).append([
                (cx - width / 2) * source_w,
                (cy - height / 2) * source_h,
                (cx + width / 2) * source_w,
                (cy + height / 2) * source_h,
            ])
        group.append({
            "bucket": bucket,
            "key": image_key,
            "image_id": len(group) + 1,
            "boxes": {cls: np.asarray(values, dtype=np.float64) for cls, values in boxes.items()},
        })
        if len(groups["train"]) >= calib_n and len(groups["val"]) >= val_n:
            break

    if groups["train"]:
        calib = groups["train"]
        val = (groups["val"] or groups["test"])[:val_n]
        if not val:
            val = calib[:min(20, len(calib))]
        return calib, val
    entries = groups["flat"] or groups["val"] + groups["test"]
    return _sequential_calib_val(entries, calib_n, val_n)


def calibration_entries(s3, dataset_uri, calib_n, val_n):
    """calibration_entries loads canonical YOLO first and COCO as fallback.

    Returns (calibration_entries, validation_entries, class_names). Every entry
    carries source-pixel xyxy boxes, the shape consumed by the OV mAP50 gate.
    """
    bucket, prefix = parse_uri(dataset_uri)
    prefix = prefix.rstrip("/")
    relative_keys = _dataset_relative_keys(s3, bucket, prefix)
    if "data.yaml" in relative_keys:
        manifest_key = f"{prefix}/data.yaml" if prefix else "data.yaml"
        raw = s3.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
        names = _yolo_class_names(raw)
        calib, val = _yolo_calib_val(
            s3, bucket, prefix, relative_keys, names, calib_n, val_n,
        )
        if calib and names:
            return calib, val, names
    entries, names = coco_entries(s3, dataset_uri, calib_n + val_n)
    calib, val = _sequential_calib_val(entries, calib_n, val_n)
    return calib, val, names


def _ov_map50(compiled, entries, names, imgsz, nseval, tensor_cache):
    """mAP50 of an OV compiled model over the given annotated entries (CPU gate-1).

    Entries are the RAW nncf.Dataset items; tensor_cache maps an
    entry to its preprocessed letterbox tensor while memoizing downloads.
    """
    import numpy as np

    nc = len(names)
    preds, gts = [], []
    for e in entries:
        t = tensor_cache(e)
        if t is None:
            continue
        raw = np.asarray(list(compiled.infer_new_request([t]).values())[0])
        if not np.isfinite(raw).all():
            return 0.0
        boxes, scores, cls_ids = nseval.decode_detections(raw, nc, 0.001, 0.6)
        _, _, r, pad_x, pad_y = e["_meta"]
        if r > 0 and len(boxes):
            boxes = (boxes - np.array([pad_x, pad_y, pad_x, pad_y])) / r
        per_cls: dict = {}
        for b, s, c in zip(boxes, scores, cls_ids):
            per_cls.setdefault(int(c), []).append((np.asarray(b, dtype=np.float64), float(s)))
        preds.append(per_cls)
        gts.append(e["boxes"])
    return float(nseval.detection_metrics(preds, gts, nc, names).get("mAP50") or 0.0)


def _count_fake_quantize(ov_model) -> int:
    """Count FakeQuantize ops (used to derive layers reverted to FP16)."""
    return sum(1 for op in ov_model.get_ops() if op.get_type_name() == "FakeQuantize")


def nncf_static_quantize(s3, cfg, ov_model, imgsz):
    """NNCF accuracy-aware static PTQ (gate-1); returns (quantized_model, info).

    Canonical YOLO uses train for calibration and val for validation; flat YOLO
    and COCO retain sequential selection with the existing small-set fallback.
    The validation metric is full mAP50 vs the FP32 IR; layers that violate
    max_map_drop_abs are reverted to FP16 by NNCF.
    Failure to satisfy the drop raises QUANTIZATION_GATE_FAILED (exit code 4
    semantics): gate-2/FP16 sibling are unaffected and no INT8 is packaged.
    """
    import nncf

    try:
        import nseval
    except ModuleNotFoundError:  # builders layout keeps it under common/
        from common import nseval

    calib_uri = str(cfg.get("calib_uri") or "")
    if not calib_uri:
        raise SystemExit("int8-static requires calib_uri (dataset prefix)")
    calib_n = int(cfg.get("calib_n", 200) or 200)
    val_n = int(cfg.get("val_n", 100) or 100)
    max_drop = float(cfg.get("max_map_drop_abs", 0.01) or 0.01)

    calib_entries, val_entries, names = calibration_entries(
        s3, calib_uri, calib_n, val_n,
    )
    unique_entries = {
        (entry["bucket"], entry["key"])
        for entry in [*calib_entries, *val_entries]
    }
    if len(names) < 1 or len(unique_entries) < 4:
        raise SystemExit(
            f"not enough annotated images for NNCF PTQ under {calib_uri} "
            f"(entries={len(unique_entries)}, classes={len(names)})"
        )
    print(f"[opt] NNCF PTQ: {len(calib_entries)} calib / {len(val_entries)} val images",
          flush=True)

    import openvino as ov

    input_name = ov_model.input(0).get_any_name()
    meta_cache: dict = {}

    def tensor_for(entry):
        """Letterbox/preprocess one raw annotated entry; memoize per image key."""
        k = entry["key"]
        if k not in meta_cache:
            s3.download_file(entry["bucket"], k, "/tmp/nncf-img")
            meta_cache[k] = _letterbox("/tmp/nncf-img", imgsz)
        meta = meta_cache[k]
        if meta is None:
            return None
        entry["_meta"] = meta[1:]
        return meta[0]

    def transform_fn(item):
        """NNCF item transform: raw annotated entry -> model input tensor dict."""
        return {input_name: tensor_for(item)}

    # FP32 reference score, evaluated by NF (not by NNCF's internal baseline):
    # gate-1 compares every quantized candidate against THIS number.
    fp32_compiled = ov.Core().compile_model(ov_model, "CPU")
    baseline_map50 = _ov_map50(
        fp32_compiled, val_entries, names, imgsz, nseval, tensor_for
    )
    print(f"[opt] FP32 gate-1 baseline mAP50={baseline_map50:.4f}", flush=True)
    del fp32_compiled

    candidate_scores: list = []
    full_validation_scores: list = []

    def validation_fn(model, validation_dataset):
        """NNCF callback (nncf 3.x two-arg form): score one model variant.

        Iterate the supplied raw-entry iterable so annotated entries (with GT
        boxes) are available; the tensor cache supplies model inputs. Returns
        one scalar where higher is better (mAP50).
        """
        # NNCF's OpenVINO backend supplies an already compiled model; compiling
        # it again makes OpenVINO interpret its repr as a filesystem path. Its
        # second argument is a raw iterable, optionally an IterationCounter,
        # rather than the original nncf.Dataset object.
        raw_entries = list(validation_dataset)
        map50 = _ov_map50(
            model, raw_entries, names, imgsz, nseval, tensor_for
        )
        candidate_scores.append(map50)
        # NNCF interleaves full-dataset validations with one-item probes while
        # detecting metric mode and collecting per-item values. Only the full
        # calls can identify the initial quantized and final reverted scores.
        if len(raw_entries) == len(val_entries):
            full_validation_scores.append(map50)
        print(f"[opt] NNCF validation mAP50={map50:.4f}", flush=True)
        return map50

    calib_ds = nncf.Dataset(calib_entries, transform_fn)
    # The validation dataset carries the same transform (NNCF contract); the
    # callback iterates raw items to keep GT boxes for detection metrics.
    val_ds = nncf.Dataset(val_entries, transform_fn)
    try:
        quantized = nncf.quantize_with_accuracy_control(
            ov_model,
            calib_ds,
            val_ds,
            validation_fn,
            max_drop=max_drop,
            preset=nncf.QuantizationPreset.MIXED,
        )
    except Exception as e:  # NNCF raises when the drop cannot be reverted away
        print(f"[opt] QUANTIZATION_GATE_FAILED: {e}", flush=True)
        raise SystemExit(
            f"QUANTIZATION_GATE_FAILED: gate-1 mAP50 drop exceeds "
            f"max_drop_absolute={max_drop} even after layer revert: {e}"
        ) from e

    candidate = (
        (full_validation_scores or candidate_scores)[-1]
        if candidate_scores else baseline_map50
    )
    # The first full callback evaluates the initial model and the second the
    # fully quantized model. A one-image validation set is indistinguishable
    # from NNCF's one-item probes, so leave this informational metric unknown.
    full_quantized = (
        full_validation_scores[1]
        if len(val_entries) > 1 and len(full_validation_scores) > 1 else None
    )
    # Number of quantizers the accuracy-aware loop reverted to FP16 = FQ nodes
    # of a plain PTQ minus FQ nodes of the accuracy-aware result.
    reverted = None
    try:
        plain = nncf.quantize(
            ov_model, calib_ds, preset=nncf.QuantizationPreset.MIXED,
        )
        reverted = max(0, _count_fake_quantize(plain) - _count_fake_quantize(quantized))
    except Exception as e:  # noqa: BLE001 - the count is informational only
        print(f"[opt] warn: could not count reverted layers: {e}", flush=True)

    gate1 = {
        "baseline_map50": round(baseline_map50, 4),
        "candidate_map50": round(candidate, 4),
        "full_quantized_map50": (
            round(full_quantized, 4) if full_quantized is not None else None
        ),
        "max_drop_absolute": max_drop,
        "validation_images": len(val_entries),
        "calibration_images": len(calib_entries),
        "passed": bool((baseline_map50 - candidate) <= max_drop + 1e-9),
    }
    info = {
        "gate1": gate1,
        "quantizer_accuracy_aware": {
            "applied": True,
            "layers_reverted_to_fp16": reverted,
        },
    }
    print(f"[opt] NNCF PTQ ok: {gate1} reverted={reverted}", flush=True)
    return quantized, info


def input_shape_for(sess, imgsz):
    """Materialize symbolic dims into a concrete YOLO-style input shape.

    Preserves the model's fixed spatial dims (e.g. a 320-exported model stays
    320) and only fills symbolic/batch dims with imgsz defaults.
    """
    shape = sess.get_inputs()[0].shape
    out = []
    for i, d in enumerate(shape):
        if d is None or d == -1 or not isinstance(d, int):
            out.append(1 if i == 0 else imgsz)
        else:
            out.append(int(d))
    if len(out) < 4:
        out = [1, 3, imgsz, imgsz]
    return out


def benchmark_ort(model_path, shape, warmup, measure):
    """Sequential single-image onnxruntime benchmark; returns latency stats."""
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    dummy = np.random.rand(*shape).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: dummy})
    lat = []
    for _ in range(measure):
        t0 = time.time()
        sess.run(None, {name: dummy})
        lat.append((time.time() - t0) * 1000)
    lat.sort()
    avg = sum(lat) / len(lat)
    return sess, {
        "avg_ms": round(avg, 3),
        "p50_ms": round(lat[len(lat) // 2], 3),
        "p95_ms": round(lat[int(len(lat) * 0.95) - 1], 3),
        "p99_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.99))], 3),
        "qps": round(1000.0 / avg, 2) if avg > 0 else 0.0,
        "measure_count": len(lat),
        "method": f"sequential, warmup {warmup}, measure {measure}",
    }


def main():
    """Run one optimization: convert/quantize, benchmark, upload artifacts.

    All inputs arrive through environment variables (see module docstring);
    the OpenVINO target now produces static FP16 IR or NNCF accuracy-aware
    INT8 IR, while cpu/rocm/ascend keep their ORT quantization paths.
    """
    input_uri = os.environ.get("INPUT_URI", "")
    output_uri = os.environ.get("OUTPUT_DIR", "")
    artifact_id = os.environ.get("ARTIFACT_ID", "")
    cfg = json.loads(os.environ.get("CONFIG_JSON", "{}"))

    s3 = s3_client()
    src = "/tmp/input.onnx"
    download(s3, input_uri, src)
    orig_size = os.path.getsize(src)

    device = cfg.get("target_device", "cpu")
    method = cfg.get("quant_method", "graph")
    imgsz = int(cfg.get("imgsz", 640))
    warmup = int(cfg.get("warmup", 10))
    measure = int(cfg.get("measure", 500))

    import numpy as np  # noqa: F401
    import onnx
    import onnxruntime as ort

    # --- Quantize / convert --------------------------------------------------
    ext, extra_files = ".onnx", []
    optimized = "/tmp/optimized.onnx"
    if device == "openvino":
        # Intel edge profile (solution D2/D3): ONNX FP32 canonical -> OV IR ->
        # static [1,3,S,S] for ALL precisions (NPU hard requirement + NI
        # static-shape validation), then FP16 via compress_to_fp16 or INT8 via
        # NNCF accuracy-aware static PTQ. ORT QDQ graphs are never converted to
        # IR here: the GPU plugin compiles foreign Q/DQ nodes into all-NaN
        # outputs (pit 178); ORT quantization stays on cpu/rocm/ascend only.
        try:
            import openvino as ov
            from openvino.tools.ovc import convert_model
        except ImportError as e:
            raise SystemExit(f"openvino not available in this image: {e}")

        ov_model = convert_model(src)
        try:
            static_shapes = {}
            for port in ov_model.inputs:
                if len(port.partial_shape) == 4:
                    static_shapes[port] = [1, 3, imgsz, imgsz]
            if static_shapes:
                ov_model.reshape(static_shapes)
                print(f"[opt] reshaped IR inputs to static {imgsz}x{imgsz}", flush=True)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"failed to make IR static at {imgsz}: {e}") from e

        names = _onnx_names(src)
        if names is not None:
            try:
                ov_model.set_rt_info(names, "names")
            except Exception as e:  # noqa: BLE001
                print(f"[opt] warn: could not embed names in IR: {e}", flush=True)

        quantizer_info: dict = {}
        if method == "fp16":
            ov.save_model(ov_model, "/tmp/optimized.xml", compress_to_fp16=True)
            precision = "FP16"
        elif method in ("int8-static", "int8"):
            ov_model, quantizer_info = nncf_static_quantize(s3, cfg, ov_model, imgsz)
            # Keep FakeQuantize scales as authored by NNCF; compress_to_fp16
            # would rewrite them.
            ov.save_model(ov_model, "/tmp/optimized.xml", compress_to_fp16=False)
            precision = "INT8"
        elif method == "int8-dynamic":
            raise SystemExit(
                "int8-dynamic is not supported on the openvino target (ORT QDQ "
                "graphs are not GPU-safe, pit 178); use int8-static NNCF PTQ"
            )
        else:  # graph / explicit fp32 fallback
            ov.save_model(ov_model, "/tmp/optimized.xml", compress_to_fp16=False)
            precision = "FP32"
        optimized, ext = "/tmp/optimized.xml", ".xml"
        extra_files = ["/tmp/optimized.bin"]
    elif device == "tensorrt":
        # TensorRT engine build requires trtexec + NVIDIA GPU. The code path
        # is kept ready; on CPU-only clusters it fails fast with a clear
        # message instead of a confusing crash.
        raise SystemExit(
            "tensorrt target requires a GPU node with trtexec installed; "
            "this cluster has no NVIDIA GPUs available for optimization jobs"
        )
    elif device in ("onnx-rocm", "ascend"):
        # AMD (ROCm) and HUAWEI (Ascend) targets produce the quantized ONNX
        # as their deployable artifact in this image; the actual EP/runtime
        # acceleration is applied on the target hardware. Benchmark is skipped
        # when no local provider can run the model (no AMD GPU / CANN toolchain).
        quantized = apply_quantization(src, method, cfg, s3, imgsz)
        if quantized != optimized:
            os.replace(quantized, optimized)
        if device == "ascend":
            try:
                import subprocess as _sp  # noqa: PLC0415
                _sp.run(["atc", "--version"], capture_output=True, timeout=20)
            except Exception as e:  # noqa: BLE001
                raise SystemExit(
                    f"huawei/ascend target requires the CANN 'atc' toolchain in "
                    f"this image; not found ({e})"
                )
    else:
        quantized = apply_quantization(src, method, cfg, s3, imgsz)
        if quantized != optimized:
            os.replace(quantized, optimized)
    opt_size = os.path.getsize(optimized) + sum(os.path.getsize(f) for f in extra_files)

    # --- Benchmark (standardized) ---------------------------------------------
    shape = input_shape_for(ort.InferenceSession(src, providers=["CPUExecutionProvider"]), imgsz)
    _, src_stats = benchmark_ort(src, shape, warmup, measure)
    if device == "openvino":
        opt_stats = benchmark_openvino(optimized, shape, warmup, measure)
    elif device in ("onnx-rocm", "ascend"):
        # No local provider for the target hardware (AMD GPU / Ascend NPU) in
        # this image: report the source benchmark and mark speedup unavailable
        # so the gate skips latency/throughput for these targets.
        opt_stats = {
            "avg_ms": None, "qps": None, "measure_count": 0,
            "method": "skipped (no local AMD/ASCEND runtime)",
        }
    else:
        _, opt_stats = benchmark_ort(optimized, shape, warmup, measure)
    speedup = (
        src_stats["avg_ms"] / opt_stats["avg_ms"] if opt_stats.get("avg_ms")
        else None
    )
    print(f"[opt] src avg {src_stats['avg_ms']}ms opt avg "
          f"{opt_stats.get('avg_ms') or 'n/a'}ms speedup {speedup or 'n/a'}", flush=True)

    # --- Upload artifacts + metrics --------------------------------------------
    out_base = f"{output_uri}/{artifact_id}.optimized"
    if device == "openvino":
        precision_enum = precision
        optimizer_name = f"openvino:{method}"
    else:
        precision_enum = {
            "graph": "FP32", "fp16": "FP16",
            "int8-dynamic": "INT8", "int8-static": "INT8",
        }.get(method, "FP32")
        optimizer_name = f"onnxruntime:{method}" if device == "cpu" else device
    metrics = {
        "optimizer": optimizer_name,
        "target_device": device,
        "quant_method": method,
        # Contract precision enum (nereid-model descriptor artifact.precision).
        "precision": precision_enum,
        "dtype": cfg.get("dtype", "fp32" if method == "graph" else ("fp16" if method == "fp16" else "int8")),
        "imgsz": imgsz,
        "source_size_bytes": orig_size,
        "optimized_size_bytes": opt_size,
        "size_reduction_pct": round((1 - opt_size / orig_size) * 100, 2) if orig_size else 0,
        "speedup_ratio": round(speedup, 3) if speedup is not None else None,
        "benchmark": opt_stats,
        "source_benchmark": src_stats,
        "environment": env_block(),
    }
    # Gate-1 (NNCF accuracy-aware PTQ) details; consumed by the packager when
    # authoring descriptor.verification.quality_gate.
    if quantizer_info:
        metrics["gate1"] = quantizer_info.get("gate1")
        metrics["quantizer_accuracy_aware"] = quantizer_info.get("quantizer_accuracy_aware")
    if output_uri:
        main_uri = f"{out_base}{ext}"
        upload(s3, optimized, main_uri)
        metrics["optimized_uri"] = main_uri
        for f in extra_files:
            fext = os.path.splitext(f)[1]
            furi = f"{output_uri}/{artifact_id}.optimized{fext}"
            upload(s3, f, furi)
            metrics.setdefault("extra_artifacts", []).append(furi)
        try:
            import nseval
        except ModuleNotFoundError:  # builders layout keeps it under common/
            from common import nseval
        upload_text(s3, f"{output_uri}/{artifact_id}.metrics.json", nseval.dump_json(metrics))

    print("[opt] SUCCESS", flush=True)
    sys.exit(0)


def convert_fp16_custom(src: str, out: str) -> str:
    """Convert an ONNX model to float16 via a manual graph pass.

    onnxconverter-common (as of 1.16) produces invalid graphs for models with
    Resize nodes (the YOLO export layout): it inserts *_output_cast0 casts that
    leave fp32/fp16 type mismatches that ORT refuses to load. Instead of
    patching that library, do the conversion directly: every float32 initializer
    and intermediate tensor becomes float16, and the graph inputs/outputs stay
    float32 with explicit Cast nodes inserted so the external contract is
    unchanged while the whole internal graph is uniformly fp16.
    """
    import onnx
    import numpy as np
    from onnx import helper

    m = onnx.load(src)
    graph = m.graph
    FLOAT = onnx.TensorProto.FLOAT
    FLOAT16 = onnx.TensorProto.FLOAT16

    # Resize requires its scale/roi inputs to stay float32 (ORT rejects fp16
    # scale); collect those tensor names so we skip converting them.
    keep_fp32: set[str] = set()
    for node in graph.node:
        if node.op_type == "Resize":
            # inputs: [X, roi, scales] (scales index 2, roi index 1)
            for idx in (1, 2):
                if len(node.input) > idx and node.input[idx]:
                    keep_fp32.add(node.input[idx])
        elif node.op_type == "Cast":
            # YOLO's built-in detect head casts outputs to fp32; keep the cast
            # io fp32 so the head contract is preserved where possible.
            for o in node.output:
                if o:
                    keep_fp32.add(o)
            for i in node.input:
                if i:
                    keep_fp32.add(i)

    # Rewrite all float32 initializers to float16.
    for init in graph.initializer:
        if init.data_type == FLOAT and init.name not in keep_fp32:
            arr = np.frombuffer(init.raw_data, dtype=np.float32).astype(np.float16)
            init.data_type = FLOAT16
            init.raw_data = arr.tobytes()
            if len(init.float_data):
                init.ClearField("float_data")
            if len(init.int32_data):
                init.ClearField("int32_data")

    # Rewrite intermediate value_info float32 tensor types to float16.
    for vi in graph.value_info:
        if vi.name in keep_fp32:
            continue
        tt = vi.type.tensor_type
        if tt.elem_type == FLOAT:
            tt.elem_type = FLOAT16

    # Graph inputs stay fp32; insert a Cast (fp32 -> fp16) right after each one.
    inserted_in: dict[str, str] = {}
    for g_in in list(graph.input):
        tt = g_in.type.tensor_type
        if tt.elem_type != FLOAT:
            continue
        name = g_in.name
        cast_name = f"{name}/to_fp16"
        new_tensor = f"{name}/casted_fp16"
        cast_node = helper.make_node(
            "Cast", inputs=[name], outputs=[new_tensor], name=cast_name,
            to=FLOAT16,
        )
        graph.node.insert(0, cast_node)
        inserted_in[name] = new_tensor

    # Rewrite the ORIGINAL consumer nodes' input references to read the casted
    # fp16 tensor. Only touch nodes that existed before the inserted casts
    # (a cast's own input must stay the fp32 graph input).
    for node in graph.node:
        if node.name.endswith("/to_fp16"):
            continue
        for i, inp in enumerate(node.input):
            if inp in inserted_in:
                node.input[i] = inserted_in[inp]

    # Graph outputs stay fp32; the producer of each output must feed fp16 into
    # an fp32 Cast. Since intermediates are fp16, prepend a Cast to fp32 that
    # produces the output name.
    inserted_out: dict[str, str] = {}
    for g_out in list(graph.output):
        tt = g_out.type.tensor_type
        if tt.elem_type != FLOAT:
            continue
        out_name = g_out.name
        casted = f"{out_name}/casted_fp32"
        # Rename the real producer tensor (fp16) to `casted`, then Cast to fp32
        # under the original output name.
        for node in graph.node:
            for i, o in enumerate(node.output):
                if o == out_name:
                    node.output[i] = casted
        cast_node = helper.make_node(
            "Cast", inputs=[casted], outputs=[out_name],
            name=f"{out_name}/from_fp16", to=FLOAT,
        )
        graph.node.append(cast_node)

    # Cast attributes for image-scale parameters (e.g. Resize scales) that are
    # float32 initializers would already be fp16; keep them fp32 is handled by
    # leaving non-float tensors alone. Node float attributes stay as-is.

    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, out)
    return out


def apply_quantization(src, method, cfg, s3, imgsz):
    """Apply the requested quantization/graph-opt to src; returns local path."""
    import onnx
    import onnxruntime as ort

    if method == "graph":
        # ORT graph optimization written out to a file
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.optimized_model_filepath = "/tmp/q.onnx"
        ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
        return "/tmp/q.onnx"
    if method == "fp16":
        out = "/tmp/q.onnx"
        try:
            convert_fp16_custom(src, out)
            ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        except Exception as e:
            # The YOLO export (with built-in detect head) is not reliably
            # convertible to fp16; fall back to graph optimization so the job
            # still produces a usable artifact instead of failing hard.
            print(f"[opt] fp16 conversion failed ({e}); falling back to graph opt", flush=True)
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.optimized_model_filepath = out
            ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
            print("[opt] fallback: graph-optimized artifact produced", flush=True)
        return out
    if method == "int8-dynamic":
        from onnxruntime.quantization import quantize_dynamic, QuantType
        out = "/tmp/q.onnx"
        # YOLO's output is raw bbox/class logits; quantizing it corrupts the
        # coordinates (mAP collapses to ~0). Exclude the nodes feeding the
        # graph outputs so only the internal activations/weights are int8.
        exclude = _output_feeders(src)
        quantize_dynamic(src, out, weight_type=QuantType.QInt8,
                         nodes_to_exclude=exclude)
        return out
    if method == "int8-static":
        from onnxruntime.quantization import (CalibrationDataReader, QuantType,
                                              quantize_static)
        calib_uri = cfg.get("calib_uri", "")
        calib_n = int(cfg.get("calib_n", 200))
        if not calib_uri:
            raise SystemExit("int8-static requires calib_uri (dataset prefix)")
        reader = calibration_reader(s3, calib_uri, imgsz, calib_n)
        out = "/tmp/q.onnx"
        # Same output-layer protection as dynamic quantization.
        exclude = _output_feeders(src)
        quantize_static(src, out, reader, weight_type=QuantType.QInt8,
                        nodes_to_exclude=exclude)
        return out
    raise SystemExit(f"unknown quant_method: {method}")


def _onnx_names(path: str) -> list[str] | None:
    """Read the class-name list from an ONNX model's 'names' metadata."""
    import json
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        for p in m.metadata_props:
            if p.key == "names":
                raw = p.value.strip()
                parsed = None
                try:
                    parsed = json.loads(raw)
                except Exception:
                    try:
                        import ast
                        parsed = ast.literal_eval(raw)
                    except Exception:
                        parsed = None
                if isinstance(parsed, dict):
                    try:
                        return [parsed[str(i)] for i in range(len(parsed))]
                    except Exception:
                        return [parsed[k] for k in sorted(parsed, key=lambda k: int(k))]
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
    except Exception:
        pass
    return None


def _output_feeders(src: str) -> list[str]:
    """Return node names that must stay float to preserve detection precision.

    For YOLO-style graphs the detect head (the last "/model.NN/" block feeding
    the graph outputs) decodes raw bbox coordinates / class logits; quantizing
    it collapses mAP. We exclude every node inside that head block only, so
    the backbone/neck layers above it still get quantized (real size win).
    """
    import onnx
    model = onnx.load(src)
    out_names = {o.name for o in model.graph.output}
    node_by_out: dict[str, str] = {}
    for n in model.graph.node:
        for o in n.output:
            node_by_out[o] = n.name
    # The head block prefix is the "/model.NN/" shared by all output feeders.
    import re
    head_prefix: str | None = None
    for on in out_names:
        n = node_by_out.get(on)
        if not n:
            continue
        m = re.match(r"^(.*?/model\.\d+/)", n)
        if not m:
            continue
        pfx = m.group(1)
        head_prefix = pfx if head_prefix is None or len(pfx) > len(head_prefix) else head_prefix
    if head_prefix is None:
        # Fall back to excluding just the direct output producers.
        return [node_by_out[o] for o in out_names if node_by_out.get(o)]
    return [n.name for n in model.graph.node if n.name.startswith(head_prefix)]


def benchmark_openvino(xml_path, shape, warmup, measure):
    """Benchmark an OpenVINO IR model with the CPU plugin."""
    import numpy as np
    import openvino as ov

    core = ov.Core()
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, "CPU")
    port = compiled.input(0)
    dummy = np.random.rand(*shape).astype(np.float32)
    for _ in range(warmup):
        compiled([dummy])
    lat = []
    for _ in range(measure):
        t0 = time.time()
        compiled([dummy])
        lat.append((time.time() - t0) * 1000)
    lat.sort()
    avg = sum(lat) / len(lat)
    return {
        "avg_ms": round(avg, 3),
        "p50_ms": round(lat[len(lat) // 2], 3),
        "p95_ms": round(lat[int(len(lat) * 0.95) - 1], 3),
        "p99_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.99))], 3),
        "qps": round(1000.0 / avg, 2) if avg > 0 else 0.0,
        "measure_count": len(lat),
        "engine": f"openvino {ov.__version__} CPU",
        "method": f"sequential, warmup {warmup}, measure {measure}",
    }


def upload_text(s3, uri, text):
    """Upload a UTF-8 string (metrics JSON) to an S3 URI."""
    bucket, key = parse_uri(uri)
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode())
    print(f"[opt] uploaded {uri}", flush=True)


if __name__ == "__main__":
    main()
