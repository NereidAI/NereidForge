# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Build a contract-compliant ``nereid-model`` edge package (schema nereid-model/1).

This script is the ONLY author of ``nereid-model.json`` and package-result/1.
NF core invokes it as a single-container Kubernetes Batch job with one argument:

    python /app/model_package.py --spec-uri s3://.../package-request.json

IR runs produce FP16 or INT8 packages with mandatory CPU/GPU numerical checks.
Direct ONNX runs produce FP32 packages with graph/contract/hash validation only;
they require no dataset and never claim unexecuted golden or device checks. The
container owns tensor introspection, server repository layout, descriptor
authoring, reproducible tar, staging upload, and result writing. NF core never
shares those helpers (ADR-017; edge package solution D4-D12).

The loose payload files are uploaded to ``output.staging_uri`` BEFORE the
reproducible tar.gz is derived from those same files; the tar remains the only
distributed object (solution D12).

Exit codes: 0 ok; 2 request/source/verification failed (a result is written
whenever the request output URI is known).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tarfile
import tempfile

from common.s3_utils import make_s3_client as s3_client

SCHEMA = "nereid-model/1"
REQUEST_SCHEMA = "nereidforge.package-request/1"
RESULT_SCHEMA = "nereidforge.package-result/1"
# OVMS_SERVER_FAMILY selects the managed OpenVINO Model Server for IR packages.
OVMS_SERVER_FAMILY = "ovms"
# OVMS_SERVER_MIN_VERSION preserves the established IR runtime floor.
OVMS_SERVER_MIN_VERSION = "2025.1"
# TRITON_SERVER_FAMILY selects the managed ONNX Runtime server for ONNX packages.
TRITON_SERVER_FAMILY = "triton"
ONNX_STRUCTURAL = "onnx-structural"

# Descriptor precision enum (artifact.precision).
PRECISIONS = {"FP16", "FP32", "INT8"}

# OV element-type token -> descriptor dtype string (NI hard-compares "FP32").
_DTYPE_MAP = {
    "f32": "FP32", "f16": "FP16", "i64": "INT64", "i32": "INT32",
    "i8": "INT8", "u8": "UINT8", "boolean": "BOOL",
}

DOC_FILE_NAMES = {
    "optimization_metrics": "optimization-metrics.json",
    "gate_report": "gate-report.json",
}


class PackageError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def log(msg):
    """Emit one packager log line (stdout is captured into the Job log tail)."""
    print(f"[pkg] {msg}", flush=True)


def parse_uri(uri):
    """Split an s3://bucket/key URI into its (bucket, key) tuple."""
    rest = uri[len("s3://"):]
    bucket, key = rest.split("/", 1)
    return bucket, key


def s3_download(s3, uri, dst):
    """Download one S3 object to dst, creating parent directories."""
    bucket, key = parse_uri(uri)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    s3.download_file(bucket, key, dst)
    log(f"downloaded {uri} -> {dst} ({os.path.getsize(dst)} B)")


def s3_read_bytes(s3, uri):
    """Read one S3 object fully into memory and return its raw bytes."""
    bucket, key = parse_uri(uri)
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def s3_read_json(s3, uri):
    """Download one S3 object and parse it as a UTF-8 JSON document."""
    return json.loads(s3_read_bytes(s3, uri))


def s3_upload_bytes(s3, uri, data, content_type="application/octet-stream"):
    """Put in-memory bytes at uri with an explicit content type."""
    bucket, key = parse_uri(uri)
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    log(f"uploaded {uri} ({len(data)} B)")


def s3_upload_file(s3, uri, path):
    """Stream-upload a local file to uri (used for staging loose payload)."""
    bucket, key = parse_uri(uri)
    s3.upload_file(path, bucket, key)
    log(f"uploaded {uri} <- {path} ({os.path.getsize(path)} B)")


def s3_join(base_uri, rel):
    """Join a base s3 prefix with a relative path without doubling slashes."""
    return base_uri.rstrip("/") + "/" + rel.lstrip("/")


# -- request/dataset boundary -------------------------------------------------

def server_model_name(model_id):
    """server_model_name returns a unique, filesystem-safe server repository name."""
    out = "".join(c.lower() if c.isalnum() else "-" for c in model_id)
    return "-".join(filter(None, out.split("-"))) or "model"


def normalize_family(hint):
    """Map a base-model/architecture hint to the decoder family this packager supports."""
    raw = str(hint or "").lower()
    if "yolov8" in raw or raw in ("yolo", "yolov8", "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"):
        return "yolov8"
    if "yolov5" in raw:
        return "yolov5"
    if "yolo11" in raw or "yolov11" in raw:
        return "yolo11"
    return "yolov8"


def _require(mapping, path):
    """Resolve a dotted field path, raising BAD_REQUEST when it is missing/empty."""
    cur = mapping
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise PackageError("BAD_REQUEST", f"request field {path} is required")
        cur = cur[part]
    if cur is None or (isinstance(cur, str) and not cur.strip()):
        raise PackageError("BAD_REQUEST", f"request field {path} must not be empty")
    return cur


def _s3_field(mapping, path):
    """Require a field to be a non-empty s3:// URI."""
    value = str(_require(mapping, path)).strip()
    if not value.startswith("s3://") or len(value.split("/", 3)) < 4:
        raise PackageError("BAD_REQUEST", f"request field {path} must be an s3:// URI")
    return value


def parse_spec(spec):
    """Validate one immutable IR or direct-ONNX request before any download."""
    if not isinstance(spec, dict):
        raise PackageError("BAD_REQUEST", "package request must be a JSON object")
    if spec.get("schema") != REQUEST_SCHEMA:
        raise PackageError("BAD_REQUEST", f"expected schema {REQUEST_SCHEMA}")
    precision = str(_require(spec, "precision")).strip().upper()
    if precision not in PRECISIONS:
        raise PackageError("BAD_REQUEST", f"unsupported precision {precision!r}")
    raw_source = spec.get("source") if isinstance(spec.get("source"), dict) else {}
    onnx_source = raw_source.get("onnx_uri") is not None
    if onnx_source and precision != "FP32":
        raise PackageError("BAD_REQUEST", "direct ONNX packages require FP32 precision")
    if not onnx_source and precision == "FP32":
        raise PackageError("BAD_REQUEST", "IR packages support only FP16 and INT8")
    if onnx_source:
        source = {
            "onnx_uri": _s3_field(spec, "source.onnx_uri"),
            "family_hint": str(_require(spec, "source.family_hint")).lower(),
            "supporting_artifacts": raw_source.get("supporting_artifacts") or [],
        }
    else:
        source = {
            "ir_xml_uri": _s3_field(spec, "source.ir_xml_uri"),
            "ir_bin_uri": _s3_field(spec, "source.ir_bin_uri"),
            "dataset_version_uri": _s3_field(spec, "source.dataset_version_uri"),
            "family_hint": str(raw_source.get("family_hint") or "yolov8"),
            "supporting_artifacts": raw_source.get("supporting_artifacts") or [],
        }
    hardware = spec.get("hardware") if isinstance(spec.get("hardware"), dict) else {}
    # Touch the complete required contract before downloading any source.
    return {
        "request_id": str(_require(spec, "request_id")),
        "tenant_id": str(_require(spec, "tenant_id")),
        "precision": precision,
        "model": {
            "id": str(_require(spec, "model.id")),
            "version": str(_require(spec, "model.version")),
            "revision": str(_require(spec, "model.revision")),
            "task": str(_require(spec, "model.task")),
        },
        "source": source,
        "hardware": {
            "profile": str(
                hardware.get("profile")
                or (ONNX_STRUCTURAL if onnx_source else "edge-verified")
            ),
            "require_numeric_verification": (
                False if onnx_source else bool(hardware.get("require_numeric_verification", True))
            ),
        },
        "output": {
            "package_uri": _s3_field(spec, "output.package_uri"),
            "sidecar_uri": _s3_field(spec, "output.sidecar_uri"),
            "result_uri": _s3_field(spec, "output.result_uri"),
            "staging_uri": _s3_field(spec, "output.staging_uri"),
        },
        "policy": {
            "conf": float(spec.get("policy", {}).get("confidence_threshold", 0.25)),
            "iou": float(spec.get("policy", {}).get("nms_iou_threshold", 0.45)),
            "score_delta": float(spec.get("policy", {}).get("score_delta", 0.02)),
            "box_iou": float(spec.get("policy", {}).get("box_iou", 0.9)),
        },
    }


def first_annotated_image(base_uri, ann):
    """Select the fixed positive sample: first non-crowd, positive COCO bbox."""
    images = {
        int(img.get("id")): img
        for img in ann.get("images", [])
        if img.get("id") is not None and img.get("file_name")
    }
    for a in ann.get("annotations", []) or []:
        if a.get("iscrowd") in (1, "1"):
            continue
        bbox = a.get("bbox") or []
        if not isinstance(bbox, list) or len(bbox) < 4 or max(bbox[:4]) <= 0:
            continue
        img = images.get(int(a.get("image_id", -1)))
        if not img:
            continue
        file_name = str(img["file_name"]).lstrip("/")
        if file_name.startswith(("s3://", "http://", "https://")):
            return file_name
        # COCO exports normally use images/<file>. Legacy exports may use a
        # bare filename or a path already including coco/.
        if file_name.startswith(("images/", "img/", "data/")):
            return s3_join(base_uri, file_name)
        return s3_join(base_uri, f"images/{file_name}")
    return ""


def _dataset_relative_keys(s3, dataset_uri):
    """List every version-relative key so YOLO selection is deterministic."""
    bucket, prefix = parse_uri(dataset_uri)
    prefix = prefix.rstrip("/") + "/"
    keys = []
    token = None
    try:
        while True:
            kwargs = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            page = s3.list_objects_v2(**kwargs)
            for item in page.get("Contents", []) or []:
                key = str(item.get("Key") or "")
                if key.startswith(prefix):
                    keys.append(key[len(prefix):])
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
            if not token:
                raise PackageError("BAD_REQUEST", f"incomplete S3 listing at {dataset_uri}")
    except PackageError:
        raise
    except Exception as e:  # noqa: BLE001
        raise PackageError("BAD_REQUEST", f"cannot list dataset at {dataset_uri}: {e}") from e
    return sorted(keys)


def _yolo_class_names(raw, yaml_uri):
    """Read class order from canonical list/dict data.yaml representations."""
    import yaml

    try:
        parsed = yaml.safe_load(raw) or {}
    except Exception as e:  # noqa: BLE001
        raise PackageError("BAD_REQUEST", f"cannot parse YOLO manifest at {yaml_uri}: {e}") from e
    if not isinstance(parsed, dict):
        raise PackageError("BAD_REQUEST", f"YOLO manifest must be an object: {yaml_uri}")
    names = parsed.get("names")
    try:
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        if isinstance(names, (list, tuple)):
            return [str(name) for name in names]
    except (TypeError, ValueError) as e:
        raise PackageError("BAD_REQUEST", f"invalid YOLO class names at {yaml_uri}: {e}") from e
    nc = parsed.get("nc")
    if isinstance(nc, int) and nc > 0:
        return [f"class{i}" for i in range(nc)]
    return []


def first_positive_yolo_image(s3, dataset_uri, relative_keys):
    """Select the first sorted image with a valid, non-empty YOLO detection label."""
    key_set = set(relative_keys)
    for image_rel in relative_keys:
        if not image_rel.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        if image_rel.startswith("images/"):
            label_rel = "labels/" + image_rel[len("images/"):]
        elif "/images/" in image_rel:
            label_rel = image_rel.replace("/images/", "/labels/", 1)
        else:
            continue
        label_rel = label_rel.rsplit(".", 1)[0] + ".txt"
        if label_rel not in key_set:
            continue
        raw_label = s3_read_bytes(s3, s3_join(dataset_uri, label_rel))
        try:
            lines = raw_label.decode().splitlines()
        except UnicodeDecodeError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
            except ValueError:
                continue
            # Detection records carry positive normalized width/height. Finite
            # values reject corrupt labels before they become golden samples.
            if class_id >= 0 and values[2] > 0 and values[3] > 0 and all(
                math.isfinite(value) for value in values
            ):
                return s3_join(dataset_uri, image_rel)
    return ""


def resolve_labels_and_sample(s3, dataset_uri):
    """Resolve canonical YOLO input first, retaining COCO as a fallback."""
    labels = []
    class_names_uri = s3_join(dataset_uri, "class_names.json")
    try:
        class_names = s3_read_json(s3, class_names_uri)
        if isinstance(class_names, list):
            labels = [str(x) for x in class_names]
    except Exception as e:  # noqa: BLE001
        log(f"warn: no class_names.json at {class_names_uri}: {e}")

    relative_keys = _dataset_relative_keys(s3, dataset_uri)
    if "data.yaml" in relative_keys:
        yaml_uri = s3_join(dataset_uri, "data.yaml")
        yolo_labels = _yolo_class_names(s3_read_bytes(s3, yaml_uri), yaml_uri)
        if not labels:
            labels = yolo_labels
        sample_uri = first_positive_yolo_image(s3, dataset_uri, relative_keys)
        if sample_uri:
            return labels, sample_uri
        log(f"no positive YOLO sample under {dataset_uri}; trying COCO fallback")

    ann_uri = s3_join(dataset_uri, "annotations.json")
    try:
        ann = s3_read_json(s3, ann_uri)
    except Exception as e:  # noqa: BLE001
        raise PackageError("BAD_REQUEST", f"cannot read COCO annotations at {ann_uri}: {e}") from e
    if not isinstance(ann, dict):
        raise PackageError("BAD_REQUEST", f"COCO annotations must be an object: {ann_uri}")
    if not labels:
        labels = [
            str(c.get("name") or "")
            for c in ann.get("categories", []) or []
            if isinstance(c, dict) and c.get("name")
        ]
    sample_uri = first_annotated_image(dataset_uri, ann)
    if not sample_uri:
        raise PackageError("BAD_REQUEST", "dataset contains no non-crowd positive sample")
    return labels, sample_uri


# -- graph introspection ----------------------------------------------------

def inspect_onnx(onnx_path):
    """Inspect runtime ONNX tensors while excluding initializer-backed inputs.

    Old exports may serialize weights as graph inputs. Only non-initializer
    inputs are runtime endpoints; external-data payloads are rejected because
    this package format deliberately carries one self-contained ONNX file.
    """
    try:
        import onnx
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception as exc:  # noqa: BLE001
        raise PackageError("BAD_REQUEST", f"cannot parse ONNX artifact: {exc}") from exc
    external = [
        tensor.name for tensor in model.graph.initializer
        if tensor.data_location == onnx.TensorProto.EXTERNAL
    ]
    if external:
        raise PackageError(
            "UNSUPPORTED_ONNX",
            "external ONNX tensor data is not supported by the single-file package layout",
        )
    dtype_names = {
        "FLOAT": "FP32", "FLOAT16": "FP16", "DOUBLE": "FP64",
        "INT64": "INT64", "INT32": "INT32", "INT8": "INT8",
        "UINT8": "UINT8", "BOOL": "BOOL",
    }

    def _tensor(value):
        tensor_type = value.type.tensor_type
        shape = [
            dim.dim_value if dim.HasField("dim_value") else -1
            for dim in tensor_type.shape.dim
        ]
        dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        return {
            "name": value.name,
            "shape": shape,
            "dtype": dtype_names.get(dtype, dtype or "UNKNOWN"),
        }

    initializers = {tensor.name for tensor in model.graph.initializer}
    inputs = [
        _tensor(value) for value in model.graph.input
        if value.name not in initializers
    ]
    outputs = [_tensor(value) for value in model.graph.output]
    return inputs, outputs


def _static_face_io(inputs, outputs):
    """Validate the common static batch-one FP32 face-model boundary."""
    if len(inputs) != 1 or not outputs:
        raise PackageError(
            "UNSUPPORTED_ONNX", "face packages require one runtime input and declared outputs",
        )
    for tensor in [*inputs, *outputs]:
        dims = tensor.get("shape") or []
        if not dims or any(not isinstance(dim, int) or dim <= 0 for dim in dims):
            raise PackageError(
                "UNSUPPORTED_ONNX",
                f"dynamic tensor {tensor.get('name')!r} shape {dims} is not packageable",
            )
        if tensor.get("dtype") != "FP32":
            raise PackageError(
                "UNSUPPORTED_ONNX", f"tensor {tensor.get('name')!r} must be FP32",
            )
    input_shape = inputs[0]["shape"]
    if len(input_shape) != 4 or input_shape[:2] != [1, 3]:
        raise PackageError(
            "UNSUPPORTED_ONNX", f"face input must be static NCHW [1,3,H,W], got {input_shape}",
        )


def _yunet_io(inputs, outputs, policy):
    """Generate and verify the modern twelve-head YuNet tensor/decoder contract."""
    input_spec = inputs[0]
    height, width = input_spec["shape"][2:]
    if height % 32 or width % 32:
        raise PackageError(
            "UNSUPPORTED_ONNX", "YuNet input dimensions must be divisible by 32",
        )
    by_name = {item["name"]: item for item in outputs}
    if len(by_name) != len(outputs):
        raise PackageError("UNSUPPORTED_ONNX", "YuNet output names must be unique")
    declared_outputs = []
    strides = []
    for stride in (8, 16, 32):
        anchors = (height // stride) * (width // stride)
        stride_spec = {"stride": stride}
        for role, channels in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)):
            name = f"{role}_{stride}"
            expected_shape = [1, anchors, channels]
            actual = by_name.get(name)
            if actual is None or actual["shape"] != expected_shape:
                raise PackageError(
                    "UNSUPPORTED_ONNX",
                    f"YuNet output {name} must have shape {expected_shape}, got "
                    f"{actual['shape'] if actual else 'missing'}",
                )
            declared_outputs.append({
                **actual, "layout": "NAC", "role": role,
                "stride": stride, "anchors": anchors,
            })
            stride_spec[role] = name
        strides.append(stride_spec)
    if set(by_name) != {item["name"] for item in declared_outputs}:
        raise PackageError(
            "UNSUPPORTED_ONNX", "modern YuNet must declare exactly twelve cls/obj/bbox/kps outputs",
        )
    input_spec = {
        **input_spec, "layout": "NCHW", "color_order": "BGR", "role": "source_image",
        "preprocess": {
            "resize": "letterbox", "scale": "1", "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0], "pad_value": 0,
        },
    }
    decoder = {
        "kind": "detection.yunet",
        "activation": "probability",
        "threshold": policy["conf"],
        "nms": {"iou": policy["iou"], "top_k": 5000},
        "strides": strides,
        "score": {
            "formula": "sqrt(clamp(cls,0,1)*clamp(obj,0,1))",
            "inputs": "cls_and_obj_are_probabilities_not_logits",
        },
        "boxes": {
            "format": "xyxy", "coordinate_space": "model-input-pixels",
            "center_x": "(col+bbox[0])*stride",
            "center_y": "(row+bbox[1])*stride",
            "width": "exp(bbox[2])*stride", "height": "exp(bbox[3])*stride",
            "corners": "[cx-w/2,cy-h/2,cx+w/2,cy+h/2]",
            "anchor_order": "row_major_col_fastest",
        },
        "landmarks": {
            "count": 5, "coordinate_space": "model-input-pixels",
            "x": "(kps[2*i]+col)*stride", "y": "(kps[2*i+1]+row)*stride",
            "order": [
                "right_eye", "left_eye", "nose_tip",
                "right_mouth_corner", "left_mouth_corner",
            ],
        },
    }
    return input_spec, declared_outputs, decoder


def _sface_io(inputs, outputs):
    """Generate and verify the aligned batch-one SFace embedding contract."""
    input_spec = inputs[0]
    if input_spec["shape"] != [1, 3, 112, 112]:
        raise PackageError(
            "UNSUPPORTED_ONNX",
            f"SFace input must be [1,3,112,112], got {input_spec['shape']}",
        )
    if len(outputs) != 1 or outputs[0]["shape"] not in ([128], [1, 128]):
        raise PackageError(
            "UNSUPPORTED_ONNX", "SFace requires exactly one [1,128] or [128] output",
        )
    landmark_order = [
        "right_eye", "left_eye", "nose_tip", "right_mouth_corner", "left_mouth_corner",
    ]
    input_spec = {
        **input_spec, "layout": "NCHW", "color_order": "RGB",
        "role": "five_point_aligned_face",
        "preprocess": {
            "resize": "stretch", "scale": "1", "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0], "pad_value": 0,
            "alignment": {
                "template": [
                    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041],
                ],
                "source_landmark_order": landmark_order,
                "interpolation": "linear", "coordinate_space": "model-input-pixels",
            },
        },
    }
    output = {
        **outputs[0], "layout": "NC", "role": "face_embedding", "dimensions": 128,
        "value_semantics": "raw_sface_embedding",
    }
    decoder = {
        "kind": "embeddings.vector", "output": output["name"], "dimensions": 128,
        "l2_normalize": True, "max_dims": 128,
        "value_semantics": "l2_normalized_128d_face_embedding",
    }
    return input_spec, [output], decoder


def build_onnx_descriptor(spec, onnx_path, payload_rel):
    """build_onnx_descriptor authors managed-Triton metadata from validated graph I/O.

    NI selects its ONNX Runtime backend from ``artifact.format``. No OVMS
    version floor or OpenVINO hardware constraint applies to this package.
    """
    family = str(spec["source"].get("family_hint") or "").lower()
    if family not in {"yunet", "sface"}:
        raise PackageError(
            "UNSUPPORTED_ONNX", "direct ONNX packaging supports family_hint yunet or sface",
        )
    inputs, outputs = inspect_onnx(onnx_path)
    _static_face_io(inputs, outputs)
    model_name = server_model_name(spec["model"]["id"])
    expected_payload = f"repository/{model_name}/1/model.onnx"
    if payload_rel != expected_payload:
        raise PackageError(
            "BAD_REQUEST", f"ONNX payload path must be {expected_payload}, got {payload_rel}",
        )
    if family == "yunet":
        model_input, model_outputs, decoder = _yunet_io(inputs, outputs, spec["policy"])
    else:
        model_input, model_outputs, decoder = _sface_io(inputs, outputs)
    return {
        "schema": SCHEMA,
        "model": {
            "id": spec["model"]["id"], "version": spec["model"]["version"],
            "revision": spec["model"]["revision"], "task": spec["model"]["task"],
            "family": family,
        },
        "artifact": {
            "format": "onnx", "precision": "FP32",
            "files": [{
                "path": payload_rel, "bytes": os.path.getsize(onnx_path),
                "sha256": sha256_file(onnx_path),
            }],
        },
        "runtime": {
            "backend": "kserve", "model_name": model_name,
            "repository_path": f"repository/{model_name}",
            "server": {"family": TRITON_SERVER_FAMILY},
        },
        "io": {"inputs": [model_input], "outputs": model_outputs},
        "decoder": decoder,
        "verification": {
            "passed": True, "mode": "structural", "numeric_status": "not_run",
            "devices": [],
        },
    }


def inspect_ir(xml_path):
    """Return (inputs, outputs, rt_names) from the real IR tensors."""
    import openvino as ov

    model = ov.Core().read_model(xml_path)

    def _shape(port):
        out = []
        for d in port.partial_shape:
            out.append(-1 if d.is_dynamic else int(d.get_length()))
        return out

    def _dtype(t):
        return _DTYPE_MAP.get(t.get_type_name(), "FP32")

    inputs = [{
        "name": p.get_any_name(),
        "shape": _shape(p),
        "dtype": _dtype(p.get_element_type()),
    } for p in model.inputs]
    outputs = [{
        "name": p.get_any_name(),
        "shape": _shape(p),
        "dtype": _dtype(p.get_element_type()),
    } for p in model.outputs]

    names = None
    try:
        rt = model.get_rt_info()
        if "names" in rt:
            val = rt["names"].value if hasattr(rt["names"], "value") else None
            if val is None:
                val = rt["names"].get()
            if isinstance(val, dict):
                names = [str(val[k]) for k in sorted(val, key=lambda x: int(x))]
            elif isinstance(val, (list, tuple)):
                names = [str(x) for x in val]
    except Exception as e:  # noqa: BLE001
        log(f"warn: could not read names from IR rt_info: {e}")
    return inputs, outputs, names, ov


def assert_static_io(inputs, outputs):
    """D7: every edge IR tensor must be static (NI validates vision H/W + A;
    NPU compile also requires static shapes). The packager never converts."""
    for t in [*inputs, *outputs]:
        dims = t.get("shape") or []
        if any(d == -1 for d in dims):
            raise PackageError(
                "UNSUPPORTED_IR",
                f"dynamic tensor {t.get('name')!r} shape {dims} is not packageable; "
                "the optimizer must emit a fully static IR",
            )


# -- preprocess + YOLOv8 anchor decode + NMS --------------------------------

def letterbox(im, sz):
    """Letterbox BGR image -> NCHW FP32 RGB/255; returns (tensor, meta).

    meta = (orig_w, orig_h, scale, pad_left, pad_top) is used to map the
    expected boxes back into SOURCE-frame normalized xyxy coordinates.
    """
    import cv2
    import numpy as np

    h0, w0 = im.shape[:2]
    r = min(sz / w0, sz / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
    top, left = (sz - nh) // 2, (sz - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    # DS preprocesses in RGB (color_order), scale 1/255, NCHW.
    x = canvas[:, :, ::-1].astype("float32").transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), (w0, h0, r, left, top)


def _iou(box, boxes):
    import numpy as np

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + areas - inter + 1e-9)


def decode_yolov8(raw, nc, conf, iou_thr, agnostic=False):
    """Decode a [1,4+nc,A] yolov8 anchor tensor into letterbox-pixel boxes."""
    import numpy as np

    pred = raw.reshape(4 + nc, -1).T  # [A, 4+nc]
    cls_scores = pred[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    confs = cls_scores.max(axis=1)
    keep = confs >= conf
    pred, cls_ids, confs = pred[keep], cls_ids[keep], confs[keep]
    xywh = pred[:, :4]
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    dets = []
    groups = (np.zeros(len(cls_ids)) if agnostic else cls_ids)
    order = np.argsort(-confs)
    suppressed = np.zeros(len(order), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        same = groups == groups[idx]
        candidates = [j for j in order
                      if not suppressed[j] and same[j] and j != idx]
        if candidates:
            ious = _iou(xyxy[idx], xyxy[candidates])
            for j, iouv in zip(candidates, ious):
                if iouv > iou_thr:
                    suppressed[j] = True
        dets.append({
            "class_id": int(cls_ids[idx]),
            "score": round(float(confs[idx]), 6),
            "box": [round(float(v), 6) for v in xyxy[idx]],
        })
    return dets


def run_infer(compiled, x):
    import numpy as np

    out = list(compiled.infer_new_request([x]).values())[0]
    arr = np.asarray(out, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise RuntimeError(
            f"non-finite values in model output "
            f"({int((~np.isfinite(arr)).sum())} / {arr.size})"
        )
    return arr


def compare_dets(ref, got, labels, score_delta_tol, box_iou_min):
    """Match got detections against the CPU reference.

    Raises RuntimeError on count/class/score/box drift; returns the measured
    ``(score_delta_max, box_iou_min)`` for the verification matrix.
    """
    import numpy as np

    if len(got) != len(ref):
        raise RuntimeError(
            f"detection count mismatch: reference {len(ref)}, device {len(got)}"
        )
    used = set()
    max_delta, min_iou = 0.0, 1.0
    for r in ref:
        best, best_iou = None, 0.0
        for j, g in enumerate(got):
            if j in used or g["class_id"] != r["class_id"]:
                continue
            iouv = float(_iou(np.array(r["box"]), np.array([g["box"]]))[0])
            if iouv > best_iou:
                best, best_iou = j, iouv
        if best is None:
            raise RuntimeError(
                f"no matching {labels[r['class_id']]!r} detection on device"
            )
        delta = abs(got[best]["score"] - r["score"])
        if delta > score_delta_tol:
            raise RuntimeError(
                f"score drift for {labels[r['class_id']]}: ref {r['score']} "
                f"vs device {got[best]['score']} (delta > {score_delta_tol})"
            )
        if best_iou < box_iou_min:
            raise RuntimeError(
                f"box drift for {labels[r['class_id']]}: IoU {best_iou:.3f} "
                f"< {box_iou_min}"
            )
        used.add(best)
        max_delta = max(max_delta, delta)
        min_iou = min(min_iou, best_iou)
    return round(max_delta, 6), round(min_iou, 6)


def source_normalized_box(box, meta):
    """Map letterbox-pixel xyxy to SOURCE-frame normalized [0,1] xyxy."""
    w0, h0, r, pad_x, pad_y = meta

    def _clip(v, hi):
        return round(float(min(max(v, 0.0), hi) / hi), 6)

    return [
        _clip((box[0] - pad_x) / r, w0),
        _clip((box[1] - pad_y) / r, h0),
        _clip((box[2] - pad_x) / r, w0),
        _clip((box[3] - pad_y) / r, h0),
    ]


def sha256_file(path):
    """Stream-hash one local file (1 MiB chunks) and return its hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reproducible_tar(root, rels):
    """Build a byte-reproducible gzip tar: sorted members, fixed mtime/uid/gid/
    mode, no user/group names, gzip mtime 0 (contract §2 / solution D10)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for rel in rels:
                p = os.path.join(root, rel)
                st = os.stat(p)
                info = tarfile.TarInfo(name=rel)
                info.size = st.st_size
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with open(p, "rb") as f:
                    tar.addfile(info, f)
    return buf.getvalue()


def result_document(spec, status, error=None, package=None, verification=None,
                    staging=None, precision=None):
    doc = {
        "schema": RESULT_SCHEMA,
        "request_id": spec["request_id"],
        "precision": precision or spec.get("precision") or "FP16",
        "status": status,
        "error": error or {"code": "", "message": ""},
    }
    if package:
        doc["package"] = package
        doc["sidecar_uri"] = spec["output"]["sidecar_uri"]
        doc["staging"] = staging or {"uri": spec["output"]["staging_uri"], "files": []}
    if verification is not None:
        doc["verification"] = verification
    return doc


def write_failure(s3, spec_uri, exc, code="INTERNAL_ERROR"):
    """Best-effort failure result for an already-readable request."""
    try:
        spec = parse_spec(s3_read_json(s3, spec_uri))
        error_code = getattr(exc, "code", code)
        result = result_document(
            spec,
            "failed",
            error={"code": error_code, "message": str(exc)[:1000]},
            verification={
                "passed": False,
                "detection_count": None,
                "devices": [],
            },
        )
        s3_upload_bytes(
            s3, spec["output"]["result_uri"],
            json.dumps(result, ensure_ascii=False, indent=2).encode(),
            "application/json",
        )
    except Exception as write_exc:  # noqa: BLE001
        log(f"could not write failure result: {write_exc}")


def _download_supporting_docs(s3, spec):
    docs, opt_metrics, gate_report = [], {}, {}
    for item in spec["source"].get("supporting_artifacts") or []:
        if not isinstance(item, dict):
            continue
        artifact_type = str(item.get("type") or "")
        uri = str(item.get("uri") or "")
        fname = DOC_FILE_NAMES.get(artifact_type)
        if not fname or not uri.startswith("s3://"):
            continue
        try:
            data = s3_read_bytes(s3, uri)
        except Exception as e:  # noqa: BLE001
            log(f"warn: skipping doc {uri}: {e}")
            continue
        if artifact_type == "optimization_metrics":
            try:
                opt_metrics = json.loads(data)
            except Exception:  # noqa: BLE001
                opt_metrics = {}
        elif artifact_type == "gate_report":
            try:
                gate_report = json.loads(data)
            except Exception:  # noqa: BLE001
                gate_report = {}
        docs.append((fname, data))
    return docs, opt_metrics, gate_report


def build_quality_gate(precision, opt_metrics, gate_report):
    """Descriptor verification.quality_gate from the optimizer gate-1 metrics
    and the evaluator gate-2 report (solution D3/D8)."""
    gate = {
        "baseline_precision": "FP32",
        "metric": "map50",
        "baseline": None,
        "candidate": None,
        "max_drop_absolute": 0.01,
        "passed": True,
    }
    if isinstance(gate_report, dict) and gate_report:
        baseline = gate_report.get("baseline_metrics") or {}
        metrics = gate_report.get("metrics") or {}
        if isinstance(baseline, dict) and baseline.get("mAP50") is not None:
            gate["baseline"] = float(baseline["mAP50"])
        if isinstance(metrics, dict) and metrics.get("mAP50") is not None:
            gate["candidate"] = float(metrics["mAP50"])
        gates = gate_report.get("gates") or {}
        abs_gate = gates.get("max_map_drop_abs") if isinstance(gates, dict) else None
        if isinstance(abs_gate, dict) and abs_gate.get("threshold") is not None:
            gate["max_drop_absolute"] = float(abs_gate["threshold"])
        if gate_report.get("hard_gates_passed") is not None:
            gate["passed"] = bool(gate_report["hard_gates_passed"])
        elif gate_report.get("gate_result"):
            gate["passed"] = str(gate_report["gate_result"]).upper() in ("PASS", "YELLOW")
    else:
        # No full-eval report staged: gate-1 NNCF numbers are the only source.
        gate1 = opt_metrics.get("gate1") if isinstance(opt_metrics, dict) else None
        if isinstance(gate1, dict):
            gate["baseline"] = gate1.get("baseline_map50")
            gate["candidate"] = gate1.get("candidate_map50")
            if gate1.get("max_drop_absolute") is not None:
                gate["max_drop_absolute"] = float(gate1["max_drop_absolute"])
            gate["passed"] = bool(gate1.get("passed", True))
    if precision == "INT8":
        qaa = opt_metrics.get("quantizer_accuracy_aware") if isinstance(opt_metrics, dict) else None
        gate["quantizer_accuracy_aware"] = (
            qaa if isinstance(qaa, dict)
            else {"applied": True, "layers_reverted_to_fp16": None}
        )
    return gate


def _publish_package(spec, s3, root, descriptor, payload, verification):
    """Publish one descriptor/payload tree through the shared immutable result path."""
    output = spec["output"]
    precision = spec["precision"]
    desc_rel = "nereid-model.json"
    with open(os.path.join(root, desc_rel), "w", encoding="utf-8") as file:
        json.dump(descriptor, file, ensure_ascii=False, indent=2)
        file.write("\n")

    # Staging and tar always come from the same tree, so their hashes cannot
    # diverge through independent serialization or payload selection.
    staging_base = output["staging_uri"].rstrip("/")
    for rel in [desc_rel, *payload]:
        s3_upload_file(s3, f"{staging_base}/{rel}", os.path.join(root, rel))
    tar_bytes = reproducible_tar(root, [desc_rel, *payload])
    tar_sha = hashlib.sha256(tar_bytes).hexdigest()
    s3_upload_bytes(s3, output["package_uri"], tar_bytes, "application/gzip")
    sidecar = {
        "package_uri": output["package_uri"], "package_sha256": tar_sha,
        "package_bytes": len(tar_bytes), "precision": precision,
        "staging_uri": staging_base + "/", "descriptor": descriptor,
    }
    s3_upload_bytes(
        s3, output["sidecar_uri"],
        json.dumps(sidecar, ensure_ascii=False, indent=2).encode(), "application/json",
    )
    result_verification = {**verification, "precision": precision}
    staging = {
        "uri": staging_base + "/",
        "files": [
            {
                "path": desc_rel, "bytes": os.path.getsize(os.path.join(root, desc_rel)),
                "sha256": sha256_file(os.path.join(root, desc_rel)),
            },
            *descriptor["artifact"]["files"],
        ],
    }
    result = result_document(
        spec, "succeeded",
        package={
            "uri": output["package_uri"], "bytes": len(tar_bytes), "sha256": tar_sha,
        },
        verification=result_verification, staging=staging,
    )
    s3_upload_bytes(
        s3, output["result_uri"],
        json.dumps(result, ensure_ascii=False, indent=2).encode(), "application/json",
    )
    log(f"SUCCESS {output['package_uri']} ({len(tar_bytes)} B sha256 {tar_sha[:16]}...)")
    summary = {
        "package_uri": output["package_uri"], "sha256": tar_sha,
        "bytes": len(tar_bytes), "precision": precision,
        "devices": [
            item["device"] + ":" + item["status"]
            for item in result_verification.get("devices") or []
        ],
    }
    if result_verification.get("detection_count") is not None:
        summary["detections"] = result_verification["detection_count"]
    print(json.dumps(summary))


def _run_onnx(spec, s3):
    """Build one self-contained FP32 ONNX package without dataset or device claims."""
    with tempfile.TemporaryDirectory(prefix="nmpkg-onnx-") as work:
        source_path = os.path.join(work, "source.onnx")
        s3_download(s3, spec["source"]["onnx_uri"], source_path)
        model_name = server_model_name(spec["model"]["id"])
        root = os.path.join(work, "pkg")
        payload = f"repository/{model_name}/1/model.onnx"
        payload_path = os.path.join(root, payload)
        os.makedirs(os.path.dirname(payload_path))
        shutil.copy2(source_path, payload_path)
        descriptor = build_onnx_descriptor(spec, payload_path, payload)
        _publish_package(
            spec, s3, root, descriptor, [payload], descriptor["verification"],
        )


def run(spec_uri):
    """Build one precision package end to end.

    Direct ONNX requests take the structural FP32 path. IR requests download
    labels and a fixed positive sample, run mandatory CPU/Intel-GPU checks, and
    retain their existing numerical contract. Both paths publish staging, a
    reproducible tar, a sidecar, and a result through one shared owner.
    Raises PackageError on any contract/verification failure (exit 2).
    """
    s3 = s3_client()
    spec = parse_spec(s3_read_json(s3, spec_uri))
    source, output, policy = spec["source"], spec["output"], spec["policy"]
    precision = spec["precision"]
    if source.get("onnx_uri"):
        _run_onnx(spec, s3)
        return
    docs, opt_metrics, gate_report = _download_supporting_docs(s3, spec)

    work = tempfile.mkdtemp(prefix="nmpkg-")
    src = os.path.join(work, "src")
    xml_path = os.path.join(src, "model.xml")
    bin_path = os.path.join(src, "model.bin")
    sample_path = os.path.join(src, "positive.jpg")
    s3_download(s3, source["ir_xml_uri"], xml_path)
    s3_download(s3, source["ir_bin_uri"], bin_path)

    dataset_labels, sample_uri = resolve_labels_and_sample(
        s3, source["dataset_version_uri"]
    )
    s3_download(s3, sample_uri, sample_path)

    inputs, outputs, ir_names, ov = inspect_ir(xml_path)
    labels = dataset_labels or ir_names or []
    if not labels:
        raise PackageError("BAD_REQUEST", "labels unavailable from dataset or IR rt_info")
    nc = len(labels)

    anchor_out = next(
        (o for o in outputs if isinstance(o["shape"], list) and len(o["shape"]) == 3),
        outputs[0],
    )
    channels = anchor_out["shape"][1]
    if channels != 4 + nc and channels > 0:
        raise PackageError(
            "BAD_REQUEST",
            f"labels/output channels inconsistent: {nc} labels but output "
            f"{anchor_out['name']} has {channels} channels (expected {4 + nc})",
        )

    # D7: edge IRs must be fully static for every precision and every tensor.
    assert_static_io(inputs, outputs)
    in_shape = inputs[0]["shape"]
    imgsz = int(in_shape[2]) if len(in_shape) == 4 else int(opt_metrics.get("imgsz") or 640)
    if len(in_shape) != 4 or in_shape[2] != in_shape[3]:
        log(f"warn: unexpected input shape {in_shape}")

    import cv2

    im = cv2.imread(sample_path)
    if im is None:
        raise PackageError("BAD_REQUEST", f"cannot read sample image {sample_uri}")
    x, meta = letterbox(im, imgsz)

    # -- CPU reference (golden vector + first device row) --------------------
    core = ov.Core()
    log("compiling reference on CPU ...")
    try:
        cpu_model = core.read_model(xml_path)
        cpu_compiled = core.compile_model(cpu_model, "CPU")
        ref_raw = run_infer(cpu_compiled, x)
    except Exception as e:  # noqa: BLE001
        raise PackageError("NUMERIC_MISMATCH", f"CPU verification failed: {e}") from e
    ref_dets = decode_yolov8(ref_raw, nc, policy["conf"], policy["iou"])
    log(f"CPU reference detections: {len(ref_dets)}")
    if not ref_dets:
        raise PackageError(
            "NUMERIC_MISMATCH",
            "positive sample yields zero detections on CPU; refusing to package "
            "without a real golden result",
        )
    devices = [{
        "device": "intel:CPU",
        "status": "passed",
        "detections": len(ref_dets),
        "score_delta_max": 0.0,
        "box_iou_min": 1.0,
        "openvino_version": ov.__version__,
    }]

    # -- Intel GPU comparison (mandatory on the edge-verified profile) -------
    log("compiling + verifying on Intel GPU ...")
    try:
        gpu_compiled = core.compile_model(core.read_model(xml_path), "GPU")
        gpu_raw = run_infer(gpu_compiled, x)
        gpu_dets = decode_yolov8(gpu_raw, nc, policy["conf"], policy["iou"])
        score_delta_max, box_iou_min = compare_dets(
            ref_dets, gpu_dets, labels,
            policy["score_delta"], policy["box_iou"],
        )
    except PackageError:
        raise
    except Exception as e:  # noqa: BLE001
        raise PackageError("NUMERIC_MISMATCH", f"GPU verification failed: {e}") from e
    devices.append({
        "device": "intel:GPU",
        "status": "passed",
        "detections": len(gpu_dets),
        "score_delta_max": score_delta_max,
        "box_iou_min": box_iou_min,
        "openvino_version": ov.__version__,
    })
    log(f"GPU verification passed ({len(gpu_dets)} dets, within tolerance)")

    # NPU cannot be claimed without hardware (no NPU in the build cluster).
    devices.append({
        "device": "intel:NPU",
        "status": "unverified",
        "reason": "no-npu-hardware-in-build-cluster",
    })

    expected_dets = [{
        "class_id": d["class_id"],
        "label": labels[d["class_id"]],
        "score": d["score"],
        # Fixed-sample expected results use SOURCE-frame normalized [0,1] xyxy.
        "box": source_normalized_box(d["box"], meta),
    } for d in ref_dets[:100]]

    model_name = server_model_name(spec["model"]["id"])
    root = os.path.join(work, "pkg")
    repo = os.path.join(root, "repository", model_name, "1")
    docs_dir = os.path.join(root, "docs")
    samples_dir = os.path.join(root, "samples")
    os.makedirs(repo)
    os.makedirs(docs_dir)
    os.makedirs(samples_dir)
    shutil.copy2(xml_path, os.path.join(repo, "model.xml"))
    shutil.copy2(bin_path, os.path.join(repo, "model.bin"))
    shutil.copy2(sample_path, os.path.join(samples_dir, "positive.jpg"))
    for fname, data in docs:
        with open(os.path.join(docs_dir, fname), "wb") as f:
            f.write(data)

    labels_rel = "labels.txt"
    with open(os.path.join(root, labels_rel), "w", encoding="utf-8") as f:
        f.write("\n".join(labels) + "\n")

    doc_rels = [f"docs/{fname}" for fname, _ in docs]
    payload = sorted([
        f"repository/{model_name}/1/model.xml",
        f"repository/{model_name}/1/model.bin",
        labels_rel,
        "samples/positive.jpg",
        *doc_rels,
    ])
    files = []
    for rel in payload:
        p = os.path.join(root, rel)
        files.append({
            "path": rel,
            "bytes": os.path.getsize(p),
            "sha256": sha256_file(p),
        })

    sample_sha = next(
        f["sha256"] for f in files if f["path"] == "samples/positive.jpg"
    )
    family = normalize_family(source["family_hint"])
    quality_gate = build_quality_gate(precision, opt_metrics, gate_report)
    descriptor = {
        "schema": SCHEMA,
        "model": {
            "id": spec["model"]["id"],
            "version": spec["model"]["version"],
            "revision": spec["model"]["revision"],
            "task": spec["model"]["task"],
            "family": family,
            "labels_file": labels_rel,
        },
        "artifact": {
            "format": "openvino-ir",
            "precision": precision,
            "files": files,
        },
        "runtime": {
            "backend": "kserve",
            "model_name": model_name,
            "repository_path": f"repository/{model_name}",
            "server": {"family": OVMS_SERVER_FAMILY, "min_version": OVMS_SERVER_MIN_VERSION},
        },
        # Device compatibility constraints only; device SELECTION never enters
        # the package (solution D1).
        "hardware": {
            "constraints": {"openvino": f">={OVMS_SERVER_MIN_VERSION}"},
        },
        "io": {
            "inputs": [{
                "name": inputs[0]["name"],
                "shape": in_shape,
                "layout": "NCHW",
                "dtype": "FP32",
                "color_order": "RGB",
                "preprocess": {
                    "resize": "letterbox",
                    "scale": "1/255",
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                    "pad_value": 114,
                },
            }],
            "outputs": [{
                "name": anchor_out["name"],
                "shape": anchor_out["shape"],
                "layout": "NCA",
                "dtype": anchor_out["dtype"],
            }],
        },
        "decoder": {
            "kind": "detection.anchors",
            "output": anchor_out["name"],
            "threshold": policy["conf"],
            "nms": {"iou": policy["iou"], "agnostic": False, "top_k": 300},
        },
        "verification": {
            "sample": {"path": "samples/positive.jpg", "sha256": sample_sha},
            "coords": "source-normalized-xyxy",
            "expect": {
                "detection_count": len(ref_dets),
                "detections": expected_dets,
            },
            "tolerance": {
                "score_delta": policy["score_delta"],
                "box_iou": policy["box_iou"],
            },
            "devices": devices,
            "quality_gate": quality_gate,
        },
    }

    verification = {
        "passed": True,
        "detection_count": len(ref_dets),
        "devices": devices,
        "quality_gate": quality_gate,
    }
    _publish_package(spec, s3, root, descriptor, payload, verification)


def main(argv=None):
    """CLI entry point: run one spec and write a result even on failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-uri", required=True)
    args = parser.parse_args(argv)
    try:
        run(args.spec_uri)
    except Exception as e:  # noqa: BLE001
        log(f"FAILED: {getattr(e, 'code', 'INTERNAL_ERROR')}: {e}")
        write_failure(s3_client(), args.spec_uri, e)
        sys.exit(2)


if __name__ == "__main__":
    main()
