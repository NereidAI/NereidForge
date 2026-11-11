# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NereidForge evaluation entrypoint (real metrics, detection + classification).

Evaluates an ONNX (or OpenVINO IR) model against a dataset from seaweedfs.
Ground truth comes from either layout:
  COCO export: {prefix}/coco/annotations.json + images (dataset versions)
  YOLO train:  {prefix}/images/*.jpg + {prefix}/labels/*.txt

Computes real detection metrics (NMS + IoU matching + mAP@50/@50:95, P/R/F1,
per-class) or classification metrics (top-1/top-5, per-class P/R). Supports a
precomputed baseline metrics dict or an in-process baseline model for deltas;
supports hard/soft gates (optimization flow) and exits 3 on hard-gate failure
(report is still written).

Env contract:
  INPUT_URI    s3://... model to evaluate (.onnx or .xml)
  BASELINE_URI s3://... optional baseline model evaluated identically
  DATASET_URI  s3://... dataset prefix
  OUTPUT_URI   s3://... output dir for {ARTIFACT_ID}.metrics.json
  REPORT_URI   s3://... optional full report.json target (file or dir)
  ARTIFACT_ID  unique id (job id)
  CONFIG_JSON  evaluation config

Exit codes: 0 ok; 3 hard-gate failure.
"""

import json
import os
import sys

import boto3


def read_model_names(path: str) -> list[str] | None:
    """Read the model's class-name list from ONNX metadata or OpenVINO IR.

    Ultralytics writes names into exported ONNX metadata; ovc preserves them
    in the IR's runtime info. Models without names return None (index-aligned
    matching for dataset-trained models).
    """
    if path.lower().endswith(".xml"):
        try:
            import openvino as ov
            core = ov.Core()
            m = core.read_model(path)
            rt = m.get_rt_info()
            if "names" not in rt:
                return None
            names = rt["names"]
            val = names.value if hasattr(names, "value") else None
            if val is None:
                try:
                    val = names.get()
                except Exception:
                    val = None
            if isinstance(val, list):
                return [str(x) for x in val]
            if isinstance(val, dict):
                try:
                    return [val[str(i)] for i in range(len(val))]
                except Exception:
                    return [val[k] for k in sorted(val, key=lambda k: int(k))]
            if isinstance(val, str):
                raw = val.strip()
                # Space-separated names ("person bicycle car ...").
                if " " in raw and not raw.startswith(("{", "[")):
                    parts = raw.split()
                    if len(parts) > 1:
                        return parts
                try:
                    import ast
                    parsed = ast.literal_eval(raw)
                except Exception:
                    parsed = raw
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
                    # {0: "person", 1: "bicycle", ...} -> ordered list.
                    try:
                        return [parsed[str(i)] for i in range(len(parsed))]
                    except Exception:
                        return [parsed[k] for k in sorted(parsed, key=lambda k: int(k))]
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
    except Exception:
        pass
    return None


def _map_model_cls(model_cls: int, num_classes: int, model_names: list[str] | None,
                   dataset_names: dict | None) -> int | None:
    """Map a model class index onto the dataset's class index by name.

    model_names come from the model's own metadata (not hardcoded). Returns
    None when the class cannot be matched (dropped from evaluation). When the
    model's class count equals the dataset's category count, the raw index is
    used unchanged (dataset-trained model).
    """
    if not dataset_names:
        return int(model_cls)
    if num_classes == len(dataset_names):
        return int(model_cls)
    if not model_names or model_cls >= len(model_names):
        return None
    model_name = str(model_names[model_cls]).lower()
    for ds_cls, ds_name in dataset_names.items():
        if str(ds_name).lower() == model_name:
            return int(ds_cls)
    return None

S3_ENDPOINT = os.environ.get(
    "S3_ENDPOINT",
    "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333",
)


def s3_client():
    """s3_client creates the standalone evaluator's client with explicit credentials."""
    access_key = os.environ.get("S3_ACCESS_KEY", "")
    secret_key = os.environ.get("S3_SECRET_KEY", "")
    if not access_key.strip() or not secret_key.strip():
        raise RuntimeError(
            "S3_ACCESS_KEY and S3_SECRET_KEY must contain non-empty "
            "operator-supplied credentials before S3 access."
        )
    return boto3.client(
        "s3", endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )


def parse_uri(uri):
    rest = uri[len("s3://"):]
    parts = rest.split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "").rstrip("/")


def list_keys(s3, bucket, prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def s3_exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def download(s3, uri, local):
    bucket, key = parse_uri(uri)
    s3.download_file(bucket, key, local)
    print(f"[eval] downloaded {uri} -> {local}", flush=True)


def upload_text(s3, uri, text):
    bucket, key = parse_uri(uri)
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode())
    print(f"[eval] uploaded {uri}", flush=True)


def gt_from_coco(s3, bucket, prefix, max_images):
    """Load ground truth from a COCO annotations.json export.

    Returns (entries, class_names) where each entry is
    {key, w, h, boxes: {cls: (K,4) xyxy}, label: cls} — label is the first
    annotation's category (classification mode).
    """
    ann_key = None
    for cand in (f"{prefix}/coco/annotations.json", f"{prefix}/annotations.json"):
        if s3_exists(s3, bucket, cand):
            ann_key = cand
            break
    if not ann_key:
        return None, None
    doc = json.loads(s3.get_object(Bucket=bucket, Key=ann_key)["Body"].read().decode())
    names = {c["id"]: str(c.get("name", c["id"])) for c in doc.get("categories", [])}
    by_img: dict[int, list] = {}
    for a in doc.get("annotations", []):
        by_img.setdefault(a["image_id"], []).append(a)
    entries = []
    for img in doc.get("images", []):
        iid, fname = img["id"], img["file_name"]
        key = None
        for base in (f"{prefix}/coco/images", f"{prefix}/images"):
            cand = f"{base}/{fname}"
            if s3_exists(s3, bucket, cand):
                key = cand
                break
        if key is None:
            continue
        w, h = int(img.get("width") or 0), int(img.get("height") or 0)
        boxes: dict[int, list] = {}
        label = None
        for a in by_img.get(iid, []):
            c = int(a["category_id"])
            x, y, bw, bh = a["bbox"]
            # Detect percentage-space boxes (Label Studio export bug where the
            # 0-100 geometry was written verbatim): when the image is > 100px
            # and every box fits in 0-100, rescale to real pixels.
            if w > 100 and h > 100 and max(abs(x), abs(y), abs(bw), abs(bh)) <= 100:
                x, y = x * w / 100.0, y * h / 100.0
                bw, bh = bw * w / 100.0, bh * h / 100.0
            boxes.setdefault(c, []).append([x, y, x + bw, y + bh])
            if label is None:
                label = c
        entries.append({"key": key, "w": w, "h": h, "boxes": boxes, "label": label})
    if max_images > 0:
        entries = entries[:max_images]
    print(f"[eval] COCO ground truth: {len(entries)} images, classes: {sorted(names)}", flush=True)
    return entries, names


def _rel_split(rel):
    """Return the canonical split token (train/val/test) of a version-relative
    key, or None for flat layouts (images/..., labels/...)."""
    first = rel.split("/", 1)[0].lower()
    if first == "train":
        return "train"
    if first in ("val", "valid", "validation", "dev"):
        return "val"
    if first in ("test", "eval"):
        return "test"
    return None


def _filter_split(img_keys, prefix, split):
    """Filter image keys to one split with documented fallbacks.

    "val" (default) falls back to train (the importer points data.yaml's val at
    train when an export shipped no validation split), "test" falls back to
    val then train; "all" and flat layouts (no split directories) use every
    image. Returns (keys, split_used).
    """
    def by(token):
        return [k for k in img_keys
                if _rel_split(k[len(prefix):].lstrip("/")) == token]

    flat = [k for k in img_keys
            if _rel_split(k[len(prefix):].lstrip("/")) is None]
    if not split or split == "all":
        return img_keys, split or "all"
    wanted = "val" if split in ("val", "valid", "validation", "dev") else split
    chain = {"val": ["val", "train"], "test": ["test", "val", "train"],
             "train": ["train"]}.get(wanted, [wanted])
    for token in chain:
        hits = by(token)
        if hits:
            return hits, token
    return flat, "flat" if flat else (img_keys, "all")


def _read_yolo_names(s3, bucket, prefix):
    """Read {class_id: name} from the canonical root data.yaml (best effort)."""
    try:
        raw = s3.get_object(Bucket=bucket, Key=f"{prefix}/data.yaml")["Body"].read().decode()
    except Exception:
        return None
    try:
        import yaml
        parsed = yaml.safe_load(raw) or {}
        names = parsed.get("names")
        if isinstance(names, dict):
            return {int(k): str(names[k]) for k in sorted(names, key=lambda x: int(x))}
        if isinstance(names, (list, tuple)):
            return {i: str(n) for i, n in enumerate(names)}
    except Exception:
        pass
    # Dependency-free fallback for the importer's own data.yaml shape
    # ("names:\n  0: helmet\n  1: head\nnc: 2").
    names: dict[int, str] = {}
    in_block = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_block = True
            continue
        if in_block:
            if not line[:1].isspace() or stripped.startswith("nc:") or ":" not in stripped:
                if names:
                    break
                continue
            head, _, val = stripped.partition(":")
            if head.strip().isdigit():
                names[int(head)] = val.strip().strip("'\"")
    return names or None


def gt_from_yolo(s3, bucket, prefix, max_images, split="val"):
    """Load ground truth from canonical split or flat YOLO layouts.

    Labels resolve as the sibling ``<split>/labels/<stem>.txt`` of each image
    (so identical stems in train/ and val/ never collide); a flat stem index
    remains as a legacy fallback. Class names come from the root data.yaml.
    """
    all_imgs = sorted(k for k in list_keys(s3, bucket, prefix)
                      if k.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
    img_keys, used_split = _filter_split(all_imgs, prefix, split)
    all_lbl = [k for k in list_keys(s3, bucket, prefix)
               if k.lower().endswith(".txt") and not k.endswith("data.yaml")]
    lbl_by_stem = {k.rsplit("/", 1)[-1].rsplit(".", 1)[0]: k for k in all_lbl}
    entries = []
    for k in img_keys:
        rel = k[len(prefix):].lstrip("/")
        lbl_key = None
        if "/images/" in rel:
            sib = rel.replace("/images/", "/labels/", 1)
            sib = sib.rsplit(".", 1)[0] + ".txt"
            cand = f"{prefix}/{sib}"
            if s3_exists(s3, bucket, cand):
                lbl_key = cand
        if lbl_key is None:
            lbl_key = lbl_by_stem.get(k.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        entries.append({"key": k, "lbl_key": lbl_key, "w": None, "h": None})
    if max_images > 0:
        entries = entries[:max_images]
    names = _read_yolo_names(s3, bucket, prefix)
    print(f"[eval] YOLO layout ({used_split} split): {len(entries)} images, "
          f"{len(all_lbl)} labels", flush=True)
    return entries, names


def gt_from_classfolders(s3, bucket, prefix, max_images, split="val"):
    """Load classification ground truth from an ImageFolder tree.

    Handles both canonical shapes: ``<split>/<class>/<file>`` and flat
    ``images/<class>/<file>``. The label index follows the root
    class_names.json written by the importer (same order the timm trainer
    uses); absent that, class folders are sorted alphabetically.
    """
    names = None
    try:
        ordered = json.loads(s3.get_object(
            Bucket=bucket, Key=f"{prefix}/class_names.json")["Body"].read().decode())
        names = {i: str(n) for i, n in enumerate(ordered)}
        class_to_idx = {str(n): i for i, n in names.items()}
    except Exception:
        class_to_idx = {}
    imgs, classes = [], set()
    for k in list_keys(s3, bucket, prefix):
        if not k.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        rel = k[len(prefix):].lstrip("/")
        parts = rel.split("/")
        cls = None
        if len(parts) >= 3 and parts[0] in ("train", "val", "valid", "test", "images"):
            cls = parts[1]
        if cls:
            imgs.append(k)
            classes.add(cls)
    if not class_to_idx:
        class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        names = {i: c for c, i in class_to_idx.items()}
    img_keys, used_split = _filter_split(sorted(imgs), prefix, split)
    entries = []
    for k in img_keys:
        parts = k[len(prefix):].lstrip("/").split("/")
        cls = parts[1] if len(parts) >= 3 else ""
        if cls not in class_to_idx:
            continue
        entries.append({"key": k, "label": class_to_idx[cls],
                        "w": None, "h": None})
    if max_images > 0:
        entries = entries[:max_images]
    print(f"[eval] ImageFolder layout ({used_split} split): {len(entries)} images, "
          f"{len(class_to_idx)} classes", flush=True)
    return entries, names


def main():
    input_uri = os.environ.get("INPUT_URI", "")
    baseline_uri = os.environ.get("BASELINE_URI", "")
    dataset_uri = os.environ.get("DATASET_URI", "")
    output_uri = os.environ.get("OUTPUT_URI", "")
    report_uri = os.environ.get("REPORT_URI", "")
    artifact_id = os.environ.get("ARTIFACT_ID", "")
    cfg = json.loads(os.environ.get("CONFIG_JSON", "{}"))

    s3 = s3_client()
    ext = ".xml" if input_uri.rstrip("/").endswith(".xml") else ".onnx"
    local_model = f"/tmp/model{ext}"
    download(s3, input_uri, local_model)
    if ext == ".xml":
        # OpenVINO IR is a (xml, bin) pair; read_model resolves the .bin next
        # to the .xml, so download the sibling binary too.
        bin_uri = input_uri.rstrip("/")[:-len(".xml")] + ".bin"
        try:
            download(s3, bin_uri, f"/tmp/model.bin")
        except Exception:
            print("[eval] no sibling .bin for OpenVINO IR; "
                  "weights may be embedded in the xml", flush=True)
    if baseline_uri:
        download(s3, baseline_uri, "/tmp/baseline.onnx")

    # Sidecar manifest written by the trainer (`<artifact>.metadata.json`):
    # authoritative task, label names and preprocessing. Models from other
    # sources may not have one, in which case the config/legacy heuristics win.
    manifest: dict = {}
    sidecars = []
    if input_uri:
        base = input_uri.rsplit(".", 1)[0]
        sidecars = [f"{base}.metadata.json", f"{input_uri.rsplit('/', 1)[0]}/metadata.json"]
    for cand in sidecars:
        try:
            download(s3, cand, "/tmp/model.metadata.json")
            with open("/tmp/model.metadata.json", encoding="utf-8") as fh:
                manifest = json.load(fh)
            print(f"[eval] manifest: {cand}", flush=True)
            break
        except Exception:
            continue

    import cv2
    import numpy as np
    import onnxruntime as ort

    import nseval

    # Task: explicit config wins, then the manifest, then detection. The
    # vocabulary is the trainer's (classification/detection/instance_seg/...).
    _task_aliases = {
        "cls": "classification", "classify": "classification", "class": "classification",
        "obb": "detection", "rotation": "detection", "object_detection": "detection",
        "seg": "instance_seg", "segmentation": "instance_seg", "mask": "instance_seg",
        "keypoint": "detection", "keypoints": "detection", "pose": "detection",
        "semantic_segmentation": "semantic_seg",
    }
    raw_task = str(cfg.get("task_type") or manifest.get("task") or "detection").lower()
    task = _task_aliases.get(raw_task, raw_task)
    if task == "multilabel":
        task = "classification"
    if task == "semantic_seg":
        raise SystemExit(
            "[eval] semantic-segmentation scoring is not implemented in the evaluator; "
            "the trainer writes pixel metrics at train time (evaluation-spec.md)"
        )
    cfg_imgsz = int(cfg.get("imgsz", 0) or 0)
    imgsz = cfg_imgsz or int(manifest.get("input_size") or 640)
    # Mutable so run_model can pin the model's real input resolution.
    model_imgsz = [imgsz]
    conf_thr = float(cfg.get("conf_threshold", 0.25))
    nms_thr = float(cfg.get("iou_threshold", 0.45))
    max_images = int(cfg.get("max_images", 0))  # 0 = all
    # Which exported split to score: val by default (canonical importer writes
    # train/val[/test] directories); "all" unions every split as before.
    eval_split = str(cfg.get("eval_split", "val") or "val").lower()
    # NF exports record the exact normalisation contract; only apply it when
    # the manifest declares one (upstream /255 exports must stay untouched).
    preprocess_mean = preprocess_std = None
    pre_manifest = manifest.get("preprocess") if isinstance(manifest, dict) else None
    if isinstance(pre_manifest, dict) and pre_manifest.get("mean") and pre_manifest.get("std"):
        preprocess_mean = np.asarray(pre_manifest["mean"], dtype=np.float32)
        preprocess_std = np.asarray(pre_manifest["std"], dtype=np.float32)

    # --- Dataset + ground truth ----------------------------------------------
    bucket, prefix = parse_uri(dataset_uri)
    coco_names = None
    if task == "classification":
        # Canonical ImageFolder first; legacy COCO-classification second.
        entries, coco_names = gt_from_classfolders(s3, bucket, prefix, max_images, eval_split)
        if not entries:
            entries, coco_names = gt_from_coco(s3, bucket, prefix, max_images)
        # Align ground-truth labels with the model's own class order (the
        # trainer writes it into the manifest; folder order may differ).
        trained_labels = manifest.get("labels") if isinstance(manifest, dict) else None
        if isinstance(trained_labels, list) and coco_names:
            trained = [str(n) for n in trained_labels]
            current = {i: str(coco_names[i]) for i in sorted(coco_names)}
            if set(trained) == set(current.values()):
                remap = {i: trained.index(name) for i, name in current.items()}
                for entry in entries:
                    entry["label"] = remap.get(entry.get("label"), entry.get("label"))
                coco_names = {i: name for i, name in enumerate(trained)}
        coco_mode = False
    else:
        entries, coco_names = gt_from_coco(s3, bucket, prefix, max_images)
        coco_mode = entries is not None
        if not coco_mode:
            entries, coco_names = gt_from_yolo(s3, bucket, prefix, max_images, eval_split)
    if not entries:
        raise SystemExit(f"no evaluable images found under {dataset_uri}")

    def load_image(entry_or_key):
        """Download + letterbox preprocess; returns (tensor, w, h, scale, pad_x, pad_y)."""
        key = entry_or_key["key"] if isinstance(entry_or_key, dict) else entry_or_key
        s3.download_file(bucket, key, "/tmp/img")
        img = cv2.imread("/tmp/img")
        if img is None:
            return None, 0, 0, 0, 0, 0
        h, w = img.shape[:2]
        # Letterbox: scale to fit model resolution keeping aspect ratio.
        res = model_imgsz[0]
        scale = min(res / w, res / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((res, res, 3), 114, dtype=np.uint8)
        pad_x = (res - nw) // 2
        pad_y = (res - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if preprocess_mean is not None:
            rgb = (rgb - preprocess_mean) / preprocess_std
        return rgb.transpose(2, 0, 1)[None], w, h, scale, pad_x, pad_y

    def load_runtime(path):
        """Return (run_fn, output_shape, model_imgsz_list) for .onnx or .xml."""
        if path.endswith(".xml"):
            import openvino as ov
            core = ov.Core()
            compiled = core.compile_model(core.read_model(path), "CPU")
            try:
                out_shape = list(compiled.outputs[0].partial_shape.to_shape())
            except RuntimeError:
                # Dynamic output (e.g. YOLO detect head NMS). Resolve the real
                # output size with a dummy inference so the caller can tell
                # detection (3D) from classification and count classes.
                dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
                res = np.asarray(list(compiled([dummy]).values())[0])
                out_shape = list(res.shape)
                del res, dummy
            # Use the IR's real input resolution (like the ONNX branch).
            in_ps = compiled.input(0).partial_shape
            try:
                in_h = int(in_ps.to_shape()[2])
            except Exception:
                in_h = 0
            ov_model_imgsz = in_h if in_h else imgsz

            def run(arr):
                return [np.asarray(list(compiled([arr]).values())[0])]

            return run, out_shape, [ov_model_imgsz], []
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        # Use the model's real input H when the model pins a fixed resolution;
        # fall back to the configured imgsz for dynamic-shape exports.
        in_shape = sess.get_inputs()[0].shape
        model_h = int(in_shape[2]) if len(in_shape) >= 3 and isinstance(in_shape[2], int) else 0
        model_imgsz = model_h if (model_h and in_shape[1] == 3) else imgsz

        def run_ort(arr):
            return sess.run(None, {name: arr})

        return run_ort, sess.get_outputs()[0].shape, [model_imgsz], [
            str(o.name) for o in sess.get_outputs()
        ]

    def run_model(model_path, tag):
        """Infer every entry; builds preds/gts or probs/labels + latency."""
        run_fn, out_shape, model_imgsz[:], out_names = load_runtime(model_path)
        model_names = read_model_names(model_path)
        if not model_names and isinstance(manifest.get("labels"), list):
            model_names = [str(n) for n in manifest["labels"]]
        named_det = "boxes" in out_names
        # OpenVINO dynamic NMS output yields an empty shape; treat detection
        # as 3D [1, N, 6] post-NMS output.
        det_out = len(out_shape) == 3 or not out_shape or named_det
        num_classes = 80
        if named_det and model_names:
            num_classes = len(model_names)
        elif det_out and len(out_shape) >= 2 and isinstance(out_shape[1], int):
            num_classes = max(out_shape[1] - 4, 1)
        elif not det_out and isinstance(out_shape[1], int):
            num_classes = out_shape[1]
        preds, gts = [], []
        probs, labels = [], []
        latencies = []
        nan_inf = 0
        import time
        for entry in entries:
            arr, w, h, scale, pad_x, pad_y = load_image(entry)
            if arr is None:
                continue
            t0 = time.time()
            out = run_fn(arr)
            latencies.append((time.time() - t0) * 1000)
            raw = np.asarray(out[0])
            if not np.isfinite(raw).all():
                nan_inf += int(np.size(raw) - np.isfinite(raw).sum())
            if task == "detection" and det_out:
                if named_det:
                    boxes, scores, cls_ids = nseval.decode_named(out, out_names, conf_thr)
                else:
                    boxes, scores, cls_ids = nseval.decode_detections(
                        raw, num_classes, conf_thr, nms_thr)
                if named_det and model_names:
                    num_classes = len(model_names)
                # Map boxes from letterbox input space back to original pixels.
                if scale > 0:
                    boxes = (boxes - np.array([pad_x, pad_y, pad_x, pad_y])) / scale
                img_preds: dict[int, list] = {}
                for b, s, c in zip(boxes, scores, cls_ids):
                    # Align the model's class index to the dataset's classes by
                    # name. Pretrained COCO models emit 80 classes; the eval
                    # dataset may have fewer / differently-ordered categories.
                    mapped = _map_model_cls(int(c), num_classes, model_names, coco_names)
                    if mapped is not None:
                        img_preds.setdefault(mapped, []).append((b, float(s)))
                preds.append(img_preds)
                if coco_mode:
                    # COCO bbox is already in original image pixels.
                    gt = {c: np.asarray(bs, dtype=np.float64) for c, bs in entry["boxes"].items()}
                    gts.append(gt)
                else:
                    txt = ""
                    if entry.get("lbl_key"):
                        txt = s3.get_object(Bucket=bucket, Key=entry["lbl_key"])["Body"].read().decode()
                    gts.append(nseval.parse_yolo_label(txt, w, h, imgsz))
            elif task != "classification":
                raise SystemExit(
                    f"[eval] task '{task}' is not scorable: the graph exposes "
                    f"{out_names or 'a single output'} (expected 'boxes' for detection)"
                )
            elif task == "classification":
                label = entry.get("label")
                if label is None:
                    continue
                labels.append(int(label))
                probs.append(raw.reshape(-1))
        bench = nseval.benchmark_summary(latencies)
        # When the model's classes were remapped to the dataset's categories,
        # evaluate against the dataset's class set only.
        if coco_names:
            n_names = len(coco_names)
            if task == "classification" or (coco_mode and num_classes != n_names):
                num_classes = n_names
        class_names = None
        if coco_names:
            class_names = [coco_names.get(i, str(i)) for i in range(num_classes)]
        if task == "detection":
            return dict(preds=preds, gts=gts, num_classes=num_classes,
                        bench=bench, nan_inf=nan_inf, class_names=class_names)
        return dict(probs=np.asarray(probs), labels=np.asarray(labels),
                    num_classes=num_classes, bench=bench, nan_inf=nan_inf,
                    class_names=class_names)

    def metrics_from(res):
        if task == "detection":
            m = nseval.detection_metrics(res["preds"], res["gts"], res["num_classes"],
                                         res.get("class_names"))
        else:
            m = nseval.classification_metrics(res["probs"], res["labels"],
                                              res.get("class_names"))
        m["nan_inf_count"] = res["nan_inf"]
        m["images_evaluated"] = len(res.get("gts", res.get("labels", [])))
        m["benchmark"] = res["bench"]
        m["avg_latency_ms"] = res["bench"].get("avg_ms", 0)
        return m

    # --- Candidate model ------------------------------------------------------
    res = run_model(local_model, "candidate")
    metrics = metrics_from(res)
    print(f"[eval] metrics: {json.dumps({k: v for k, v in metrics.items() if k != 'per_class'})}", flush=True)

    # --- Baseline (precomputed dict from the batch baseline step, or a model
    # URI evaluated identically in-process) -------------------------------------
    baseline_metrics = cfg.get("baseline_metrics")
    if baseline_uri:
        b_res = run_model("/tmp/baseline.onnx", "baseline")
        baseline_metrics = metrics_from(b_res)
        print(f"[eval] baseline: {json.dumps({k: v for k, v in baseline_metrics.items() if k != 'per_class'})}", flush=True)

    # --- Gates (optimization flow) ---------------------------------------------
    gates_out = None
    if cfg.get("gates"):
        gates_out = nseval.evaluate_gates(metrics, baseline_metrics, cfg["gates"])
        print(f"[eval] gate: {gates_out['gate_result']} {json.dumps(gates_out['gates'])}", flush=True)

    # --- Outputs ----------------------------------------------------------------
    job_type = cfg.get("job_type", "evaluation")
    report = nseval.build_report(
        job_type, artifact_id, task, input_uri, dataset_uri, metrics,
        baseline_uri=baseline_uri, baseline_metrics=baseline_metrics,
        gates_out=gates_out,
    )
    if output_uri:
        upload_text(s3, f"{output_uri}/{artifact_id}.metrics.json",
                    nseval.dump_json(metrics))
    # Local copy so pipeline components can forward metrics as outputs
    with open(f"/tmp/{artifact_id}.metrics.json", "w") as f:
        f.write(nseval.dump_json(metrics))
    ruri = report_uri or (f"{output_uri}/{artifact_id}.report.json" if output_uri else "")
    if ruri:
        upload_text(s3, ruri, nseval.dump_json(report))
        with open("/tmp/report.json", "w") as f:
            f.write(nseval.dump_json(report))

    if gates_out and not gates_out["hard_gates_passed"]:
        failed = [f"{k}: {v['value']} vs thr {v['threshold']}"
                  for k, v in gates_out["gates"].items() if v.get("hard") and not v["passed"]]
        print(f"[eval] GATE_FAIL: {'; '.join(failed)}", flush=True)
        sys.exit(3)

    print("[eval] SUCCESS", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
