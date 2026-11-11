# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""ONNX inference adapter for the NF ML backend.

Runs models exported to ONNX without a docker daemon or framework install.
The image-trial registry covers detection, instance segmentation, semantic
segmentation, and classification. Automatic decoding requires one complete
YuNet named-output contract, an exact named-output contract, or explicit
manifest / model-family metadata. Custom mappings declare every input and output
semantic; rank and shape alone never select a task.

The legacy ``detect_onnx``, ``classify_onnx`` and ``segment_onnx`` functions
are compatibility wrappers around that registry. Temporal video classification
retains its separate frame iterator. Class names come from metadata embedded
in the ONNX file when present;
otherwise the caller's label config order maps to output class indices.
The optional NF manifest (`<stem>.metadata.json`, `metadata.json` next to the
file, or the `nf_manifest` key in the ONNX metadata map) overrides task,
input size, label names and the normalisation contract.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_session_cache: dict[str, Any] = {}


def _cached_session(path: str) -> Any:
    """Load (and cache) an onnxruntime InferenceSession."""
    import onnxruntime as ort

    if path not in _session_cache:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session_cache[path] = ort.InferenceSession(
            path, so, providers=["CPUExecutionProvider"]
        )
    return _session_cache[path]


def get_session(path: str) -> Any:
    """Public accessor for the shared onnxruntime session cache."""
    return _cached_session(path)


def read_onnx_names(path: str) -> Optional[list[str]]:
    """Read embedded class names from ONNX metadata, if any.

    ultralytics exports store names as a Python-dict repr string
    ("{0: 'person', 1: 'car'}"), not JSON — handle both plus plain lists.
    """
    try:
        import ast

        sess = _cached_session(path)
        meta = sess.get_modelmeta().custom_metadata_map
        for key in ("names", "class_names", "categories"):
            if key not in meta:
                continue
            val = meta[key].strip()
            names = None
            try:
                import json

                parsed = json.loads(val)
                names = parsed if isinstance(parsed, (list, dict)) else None
            except Exception:
                # python repr dict/list
                try:
                    parsed = ast.literal_eval(val)
                    names = parsed if isinstance(parsed, (list, dict)) else None
                except Exception:
                    names = None
            if names is None:
                names = [s.strip() for s in val.split(",") if s.strip()]
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=lambda x: int(x))]
            if names:
                return [str(n) for n in names]
    except Exception as e:  # noqa: BLE001
        logger.debug("no embedded names in %s: %s", path, e)
    # NF exports carry the label list in the sidecar manifest, not in the
    # ONNX metadata map (the graph is re-exportable without rewriting it).
    labels = read_onnx_manifest(path).get("labels")
    if isinstance(labels, list) and labels:
        return [str(n) for n in labels]
    return None


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.45) -> list[int]:
    """Plain NMS over xyxy boxes."""
    keep: list[int] = []
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou_v = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou_v <= iou]
    return keep


def _manifest_pixel_norm(
    manifest: dict,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Pixel normalisation from the manifest; identity when it declares none.

    Upstream single-tensor exports (YOLO-style) expect /255 only, so the
    ImageNet default must not be applied unless the manifest says so. NF-trained
    graphs always record `preprocess.mean/std`.
    """
    if isinstance(manifest, dict) and isinstance(manifest.get("preprocess"), dict):
        return _manifest_norm(manifest)
    return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)


_YUNET_STRIDES = (8, 16, 32)
_YUNET_OUTPUT_NAMES = tuple(
    f"{kind}_{stride}"
    for kind in ("cls", "obj", "bbox", "kps")
    for stride in _YUNET_STRIDES
)
_YUNET_LEGACY_OUTPUT_NAMES = ("loc", "conf", "iou")
_YUNET_LEGACY_MIN_SIZES = (
    (10, 16, 24),
    (32, 48),
    (64, 96),
    (128, 192, 256),
)
_YUNET_LEGACY_STEPS = (8, 16, 32, 64)


def _prepare_yunet_input(
    sess: Any, image: np.ndarray, dynamic_pad_divisor: int
) -> tuple[np.ndarray, int, int, float, int, int]:
    """Build YuNet's unnormalized BGR input and its original-image transform.

    Static exports receive zero-padded letterboxing so arbitrary UI images fit.
    Dynamic modern graphs pad to their 32-pixel divisor, while the legacy graph
    retains the exact source size because its official decoder has no padding.
    """
    import cv2

    inp = sess.get_inputs()[0]
    if len(inp.shape) != 4 or inp.shape[1] != 3:
        raise ValueError(f"unsupported YuNet input shape {inp.shape}")

    source_h, source_w = image.shape[:2]
    static_hw = isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int)
    if static_hw:
        input_h, input_w = int(inp.shape[2]), int(inp.shape[3])
        scale = min(input_w / source_w, input_h / source_h)
        resized_w = int(round(source_w * scale))
        resized_h = int(round(source_h * scale))
        resized = cv2.resize(image, (resized_w, resized_h))
        pad_x = (input_w - resized_w) // 2
        pad_y = (input_h - resized_h) // 2
        canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    else:
        input_w = ((source_w - 1) // dynamic_pad_divisor + 1) * dynamic_pad_divisor
        input_h = ((source_h - 1) // dynamic_pad_divisor + 1) * dynamic_pad_divisor
        scale = 1.0
        pad_x = pad_y = 0
        canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        canvas[:source_h, :source_w] = image

    blob = canvas.astype(np.float32).transpose(2, 0, 1)[None]
    return blob, input_w, input_h, scale, pad_x, pad_y


def _yunet_face_results(
    boxes: np.ndarray,
    scores: np.ndarray,
    names: Optional[list[str]],
    *,
    scale: float,
    pad_x: int,
    pad_y: int,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Apply YuNet's pre-NMS bound and restore boxes to source coordinates."""
    if len(scores) > 5000:
        top = np.argsort(-scores, kind="stable")[:5000]
        boxes = boxes[top]
        scores = scores[top]

    face_name = names[0] if names else "face"
    results: list[dict[str, Any]] = []
    for index in _nms(boxes, scores, iou=iou_threshold):
        x1, y1, x2, y2 = boxes[index]
        results.append(
            {
                "xyxy": [
                    float((x1 - pad_x) / scale),
                    float((y1 - pad_y) / scale),
                    float((x2 - pad_x) / scale),
                    float((y2 - pad_y) / scale),
                ],
                "score": float(scores[index]),
                "cls": 0,
                "name": face_name,
            }
        )
    return sorted(results, key=lambda result: -result["score"])


def _detect_yunet(
    sess: Any,
    image: np.ndarray,
    conf_threshold: float,
    names: Optional[list[str]],
    iou_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """Decode OpenCV Zoo YuNet's three-scale raw heads into face detections.

    YuNet is not a single-matrix YOLO head: classification, objectness, boxes,
    and five landmarks are separate outputs. Its OpenCV reference consumes BGR
    pixels without RGB conversion or /255 normalisation, so this path also
    builds its own input instead of reusing the generic detector preprocessing.
    Static graphs use zero-padded letterboxing; dynamic graphs retain the
    source size and only pad to YuNet's 32-pixel divisor, matching the reference.
    """
    blob, input_w, input_h, scale, pad_x, pad_y = _prepare_yunet_input(
        sess, image, dynamic_pad_divisor=32
    )
    inp = sess.get_inputs()[0]
    values = sess.run(list(_YUNET_OUTPUT_NAMES), {inp.name: blob})
    outputs = {
        name: np.asarray(value)
        for name, value in zip(_YUNET_OUTPUT_NAMES, values, strict=True)
    }

    box_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for stride in _YUNET_STRIDES:
        cls = outputs[f"cls_{stride}"].reshape(-1)
        obj = outputs[f"obj_{stride}"].reshape(-1)
        bbox = outputs[f"bbox_{stride}"].reshape(-1, 4)
        rows = input_h // stride
        cols = input_w // stride
        anchors = rows * cols
        if not (len(cls) == len(obj) == len(bbox) == anchors):
            raise ValueError(
                f"invalid YuNet stride-{stride} output shapes for "
                f"{input_w}x{input_h} input"
            )

        scores = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
        selected = np.flatnonzero(scores >= conf_threshold)
        if not selected.size:
            continue
        row = selected // cols
        col = selected % cols
        raw = bbox[selected]
        cx = (col + raw[:, 0]) * stride
        cy = (row + raw[:, 1]) * stride
        width = np.exp(raw[:, 2]) * stride
        height = np.exp(raw[:, 3]) * stride
        boxes = np.column_stack(
            (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
        )
        box_parts.append(boxes)
        score_parts.append(scores[selected])

    if not box_parts:
        return []
    boxes = np.concatenate(box_parts).astype(np.float32)
    scores = np.concatenate(score_parts).astype(np.float32)
    return _yunet_face_results(
        boxes,
        scores,
        names,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        iou_threshold=iou_threshold,
    )


def _yunet_legacy_priors(input_w: int, input_h: int) -> np.ndarray:
    """Generate the pixel-space priors from OpenCV 4.7 FaceDetectorYN."""
    map_w = (input_w + 1) // 2 // 2
    map_h = (input_h + 1) // 2 // 2
    priors = []
    for min_sizes, step in zip(
        _YUNET_LEGACY_MIN_SIZES, _YUNET_LEGACY_STEPS, strict=True
    ):
        map_w //= 2
        map_h //= 2
        for row in range(map_h):
            for col in range(map_w):
                for min_size in min_sizes:
                    priors.append(
                        (
                            (col + 0.5) * step,
                            (row + 0.5) * step,
                            float(min_size),
                            float(min_size),
                        )
                    )
    return np.asarray(priors, dtype=np.float32).reshape(-1, 4)


def _detect_yunet_legacy(
    sess: Any,
    image: np.ndarray,
    conf_threshold: float,
    names: Optional[list[str]],
    iou_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """Decode the official 2021 YuNet loc/conf/iou export.

    The formulas mirror OpenCV 4.7 FaceDetectorYN: four feature levels use
    fixed priors, confidence combines the foreground and clamped IoU heads,
    and box/landmark offsets use variances 0.1 and 0.2. The graph's landmark
    values remain unused because the normalized trial DTO renders face boxes.
    """
    blob, input_w, input_h, scale, pad_x, pad_y = _prepare_yunet_input(
        sess, image, dynamic_pad_divisor=1
    )
    inp = sess.get_inputs()[0]
    values = sess.run(list(_YUNET_LEGACY_OUTPUT_NAMES), {inp.name: blob})
    loc = np.asarray(values[0]).reshape(-1, 14)
    conf = np.asarray(values[1]).reshape(-1, 2)
    iou = np.asarray(values[2]).reshape(-1)
    priors = _yunet_legacy_priors(input_w, input_h)
    if not (len(loc) == len(conf) == len(iou) == len(priors)):
        raise ValueError(
            f"invalid legacy YuNet output shapes for {input_w}x{input_h} input"
        )

    scores = np.sqrt(conf[:, 1] * np.clip(iou, 0.0, 1.0))
    selected = np.flatnonzero(scores >= conf_threshold)
    if not selected.size:
        return []
    raw = loc[selected]
    prior = priors[selected]
    center_x = prior[:, 0] + raw[:, 0] * 0.1 * prior[:, 2]
    center_y = prior[:, 1] + raw[:, 1] * 0.1 * prior[:, 3]
    width = prior[:, 2] * np.exp(raw[:, 2] * 0.1)
    height = prior[:, 3] * np.exp(raw[:, 3] * 0.2)
    boxes = np.column_stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        )
    ).astype(np.float32)
    return _yunet_face_results(
        boxes,
        scores[selected].astype(np.float32),
        names,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        iou_threshold=iou_threshold,
    )


def detect_onnx(
    path: str,
    image: np.ndarray,
    conf_threshold: float = 0.25,
    size: int = 0,
    iou_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """Run detection through the registry while preserving the legacy result DTO.

    Calling this detection-specific API is an explicit compatibility choice, so
    an unrecognized graph falls back to the manual YOLO decoder. Image trials
    use automatic mode directly and therefore still require an exact match.
    """
    manifest = read_onnx_manifest(path)
    session = _cached_session(path)
    names = read_onnx_names(path)
    inspection = inspect_decoders(session, manifest)
    selected = inspection["auto_decoder"]
    decoder = None
    if not selected or selected["task"] != "detection":
        decoder = {"mode": "builtin", "decoder_id": "yolo"}
    result = run_onnx_trial(
        session,
        image,
        manifest=manifest,
        decoder=decoder,
        names=names,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        input_shape=[1, 3, size, size] if size else None,
    )
    return result["detections"]


# ---------------------------------------------------------------------------
# NF manifest (task / labels / input contract)
# ---------------------------------------------------------------------------

MANIFEST_META_KEY = "nf_manifest"

# ImageNet normalisation is the default contract for torchvision backbones;
# the trainer records the exact values in the manifest so inference never has
# to guess when a profile changes them.
_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)
_DEFAULT_CLS_SIZE = 224


def read_onnx_manifest(path: str) -> dict:
    """Return the NF model manifest, or {} when absent/unreadable.

    Lookup order: sidecar `<stem>.metadata.json`, then `metadata.json` in the
    same directory, then the `nf_manifest` entry of the ONNX metadata map.
    Sidecar files win so a re-exported graph can be re-tagged without
    rewriting the protobuf.
    """
    import json

    candidates = (
        os.path.splitext(path)[0] + ".metadata.json",
        os.path.join(os.path.dirname(os.path.abspath(path)), "metadata.json"),
    )
    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        try:
            with open(cand, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except Exception as e:  # noqa: BLE001
            logger.debug("manifest %s unreadable: %s", cand, e)
    try:
        raw = (
            _cached_session(path)
            .get_modelmeta()
            .custom_metadata_map.get(MANIFEST_META_KEY)
        )
        if raw:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
    except Exception as e:  # noqa: BLE001
        logger.debug("no embedded nf manifest in %s: %s", path, e)
    return {}


def manifest_task(manifest: dict, default: str = "") -> str:
    """Task family from a manifest (accepts `task` or `metadata.task`)."""
    if not isinstance(manifest, dict):
        return default
    task = manifest.get("task")
    if not task:
        task = (manifest.get("metadata") or {}).get("task")
    return str(task or default)


def _manifest_input_size(manifest: dict, default: int) -> int:
    spec = manifest.get("input") if isinstance(manifest, dict) else None
    if isinstance(spec, dict):
        size = spec.get("size") or spec.get("image_size")
        try:
            if isinstance(size, (list, tuple)) and size:
                return int(size[-1])
            if size:
                return int(size)
        except (TypeError, ValueError):
            pass
    return default


def _manifest_norm(
    manifest: dict,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    spec = manifest.get("preprocess") if isinstance(manifest, dict) else None
    if not isinstance(spec, dict):
        spec = {}
    mean = spec.get("mean") or _DEFAULT_MEAN
    std = spec.get("std") or _DEFAULT_STD
    try:
        mean = tuple(float(v) for v in mean)[:3]
        std = tuple(float(v) for v in std)[:3]
    except (TypeError, ValueError):
        mean, std = _DEFAULT_MEAN, _DEFAULT_STD
    return mean, std  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class Probs:
    """Minimal stand-in for the ultralytics `Results.probs` attribute.

    The video/timeline path (utils/neural_nets.py) expects `frame.probs.data`
    to be a torch tensor, so ONNX results expose the same tiny shape instead
    of leaking framework-specific objects into the callers.
    """

    def __init__(self, data, top1: int = 0, top1conf: float = 0.0):
        self.data = data
        self.top1 = int(top1)
        self.top1conf = float(top1conf)


class FrameResult:
    """Minimal stand-in for an ultralytics `Results` item (probs only)."""

    def __init__(self, probs: Probs):
        self.probs = probs


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def classify_onnx(path: str, image: np.ndarray, topk: int = 5) -> list[dict[str, Any]]:
    """Classify through the registry; return legacy ``cls/name/prob`` entries."""
    manifest = read_onnx_manifest(path)
    session = _cached_session(path)
    inspection = inspect_decoders(session, manifest)
    selected = inspection["auto_decoder"]
    decoder = None
    if not selected or selected["task"] != "classification":
        decoder = {"mode": "builtin", "decoder_id": "torchvision_classification"}
    result = run_onnx_trial(
        session,
        image,
        manifest=manifest,
        decoder=decoder,
        names=read_onnx_names(path),
        top_k=topk,
    )
    return result["predictions"]


def classify_video_frames(
    path: str,
    video_path: str,
    cache_params: str = "",
    stride: int = 1,
):
    """Yield FrameResult per sampled video frame (temporal classifier input)."""
    import cv2

    manifest = read_onnx_manifest(path)
    size = _manifest_input_size(manifest, _DEFAULT_CLS_SIZE)
    mean, std = _manifest_norm(manifest)
    sess = _cached_session(path)
    inp = sess.get_inputs()[0]
    nhwc = inp.shape[-1] == 3 if isinstance(inp.shape[-1], int) else False
    hw = None
    if isinstance(inp.shape, (list, tuple)) and len(inp.shape) == 4:
        if isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
            hw = (inp.shape[2], inp.shape[3])
    height, width = hw or (size, size)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if stride > 1 and idx % stride:
                idx += 1
                continue
            rgb = (
                cv2.resize(frame, (width, height))[:, :, ::-1].astype(np.float32)
                / 255.0
            )
            rgb = (rgb - np.asarray(mean, dtype=np.float32)) / np.asarray(
                std, dtype=np.float32
            )
            blob = rgb[None] if nhwc else rgb.transpose(2, 0, 1)[None]
            outs = sess.run(None, {inp.name: blob})
            probs = _softmax(np.asarray(outs[0]).reshape(-1))
            import torch

            tensor = torch.from_numpy(probs.astype(np.float32))
            yield FrameResult(Probs(tensor, int(probs.argmax()), float(probs.max())))
            idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Instance segmentation
# ---------------------------------------------------------------------------


def segment_onnx(
    path: str,
    image: np.ndarray,
    conf_threshold: float = 0.25,
    size: int = 0,
) -> list[dict[str, Any]]:
    """Run instance segmentation through the registry without detection fallback."""
    manifest = read_onnx_manifest(path)
    session = _cached_session(path)
    inspection = inspect_decoders(session, manifest)
    selected = inspection["auto_decoder"]
    decoder = None
    if not selected or selected["task"] != "instance_segmentation":
        decoder = {"mode": "builtin", "decoder_id": "mask_rcnn"}
    result = run_onnx_trial(
        session,
        image,
        manifest=manifest,
        decoder=decoder,
        names=read_onnx_names(path),
        conf_threshold=conf_threshold,
        input_shape=[1, 3, size, size] if size else None,
    )
    return result["detections"]


# ---------------------------------------------------------------------------
# Image-trial decoder registry
# ---------------------------------------------------------------------------


class DecoderMappingError(ValueError):
    """Reject an incomplete or inconsistent declarative decoder mapping."""

    code = "invalid_decoder_mapping"


def _mapping_fields(
    spec: Any,
    required: set[str],
    optional: set[str],
    context: str,
) -> dict:
    """Validate a mapping object without accepting aliases or implicit fields."""
    if not isinstance(spec, dict):
        raise DecoderMappingError(f"{context} must be an object")
    missing = sorted(required - set(spec))
    unknown = sorted(set(spec) - required - optional)
    if missing:
        raise DecoderMappingError(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        raise DecoderMappingError(f"{context} unknown fields: {', '.join(unknown)}")
    return spec


def _mapping_choice(spec: dict, key: str, choices: set[str], context: str) -> str:
    """Read a required string enum from a declarative mapping."""
    value = spec.get(key)
    if not isinstance(value, str) or value not in choices:
        raise DecoderMappingError(
            f"{context}.{key} must be one of {', '.join(sorted(choices))}"
        )
    return value


def _mapping_int(spec: dict, key: str, context: str, minimum: int = 0) -> int:
    """Read an integer field while rejecting booleans and lossy float coercion."""
    value = spec.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DecoderMappingError(f"{context}.{key} must be an integer")
    if int(value) < minimum:
        raise DecoderMappingError(f"{context}.{key} must be >= {minimum}")
    return int(value)


def _mapping_number(spec: dict, key: str, context: str) -> float:
    """Read a finite numeric mapping field."""
    value = spec.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise DecoderMappingError(f"{context}.{key} must be a number")
    number = float(value)
    if not np.isfinite(number):
        raise DecoderMappingError(f"{context}.{key} must be finite")
    return number


def _custom_input(
    sess: Any, image: np.ndarray, spec: Any
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Preprocess and run an explicitly mapped image input.

    The graph's fixed dimensions are authoritative. Requested dimensions are a
    preprocessing contract, not permission to reshape an incompatible graph.
    """
    import cv2

    context = "decoder.input"
    spec = _mapping_fields(
        spec,
        {"tensor", "width", "height", "layout", "color", "resize", "scale"},
        {"mean", "std"},
        context,
    )
    tensor = spec["tensor"]
    if not isinstance(tensor, str) or not tensor:
        raise DecoderMappingError(f"{context}.tensor must be a tensor name")
    input_items = list(sess.get_inputs())
    if len(input_items) != 1:
        raise DecoderMappingError(
            "single-image decoder requires exactly one graph input"
        )
    inputs = {item.name: item for item in input_items}
    if tensor not in inputs:
        raise DecoderMappingError(f"unknown input tensor: {tensor}")
    width = _mapping_int(spec, "width", context, minimum=1)
    height = _mapping_int(spec, "height", context, minimum=1)
    layout = _mapping_choice(spec, "layout", {"NCHW", "NHWC"}, context)
    color = _mapping_choice(spec, "color", {"RGB", "BGR"}, context)
    resize = _mapping_choice(spec, "resize", {"letterbox", "stretch"}, context)
    pixel_scale = _mapping_number(spec, "scale", context)
    mean = spec.get("mean", [0.0, 0.0, 0.0])
    std = spec.get("std", [1.0, 1.0, 1.0])
    if (
        not isinstance(mean, (list, tuple))
        or len(mean) != 3
        or not isinstance(std, (list, tuple))
        or len(std) != 3
    ):
        raise DecoderMappingError(f"{context}.mean/std must contain three numbers")
    try:
        mean_array = np.asarray(mean, dtype=np.float32)
        std_array = np.asarray(std, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise DecoderMappingError(f"{context}.mean/std must contain numbers") from exc
    if not np.isfinite(mean_array).all() or not np.isfinite(std_array).all():
        raise DecoderMappingError(f"{context}.mean/std must be finite")
    if np.any(std_array == 0):
        raise DecoderMappingError(f"{context}.std values must be non-zero")

    shape = inputs[tensor].shape
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        raise DecoderMappingError(
            f"mapped input {tensor} must be a rank-4 image tensor"
        )
    batch, channels = (shape[0], shape[1]) if layout == "NCHW" else (shape[0], shape[3])
    fixed_hw = (shape[2], shape[3]) if layout == "NCHW" else (shape[1], shape[2])
    if isinstance(batch, int) and batch != 1:
        raise DecoderMappingError(
            f"mapped input {tensor} requires batch {batch}, not 1"
        )
    if isinstance(channels, int) and channels != 3:
        raise DecoderMappingError(f"mapped input {tensor} requires {channels} channels")
    if isinstance(fixed_hw[0], int) and int(fixed_hw[0]) != height:
        raise DecoderMappingError(
            f"decoder.input.height {height} does not match graph height {fixed_hw[0]}"
        )
    if isinstance(fixed_hw[1], int) and int(fixed_hw[1]) != width:
        raise DecoderMappingError(
            f"decoder.input.width {width} does not match graph width {fixed_hw[1]}"
        )

    source_h, source_w = image.shape[:2]
    if resize == "letterbox":
        ratio = min(width / source_w, height / source_h)
        resized_w = max(1, int(round(source_w * ratio)))
        resized_h = max(1, int(round(source_h * ratio)))
        resized = cv2.resize(image, (resized_w, resized_h))
        pad_x = (width - resized_w) // 2
        pad_y = (height - resized_h) // 2
        canvas = np.full((height, width, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    else:
        ratio = 1.0
        pad_x = pad_y = 0
        canvas = cv2.resize(image, (width, height))
    pixels = canvas[:, :, ::-1] if color == "RGB" else canvas
    pixels = pixels.astype(np.float32) * pixel_scale
    pixels = (pixels - mean_array) / std_array
    blob = pixels.transpose(2, 0, 1)[None] if layout == "NCHW" else pixels[None]
    values = sess.run(None, {tensor: blob.astype(np.float32, copy=False)})
    outputs = {
        output.name: np.asarray(value)
        for output, value in zip(sess.get_outputs(), values, strict=True)
    }
    geometry = {
        "resize": resize,
        "width": width,
        "height": height,
        "source_width": source_w,
        "source_height": source_h,
        "ratio": ratio,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }
    return outputs, geometry


def _restore_boxes(
    boxes: np.ndarray, space: str, geometry: dict[str, Any]
) -> np.ndarray:
    """Map model-input boxes back to the original image without value guessing."""
    restored = np.asarray(boxes, dtype=np.float32).copy()
    if space == "normalized":
        restored[:, [0, 2]] *= geometry["width"]
        restored[:, [1, 3]] *= geometry["height"]
    if geometry["resize"] == "letterbox":
        restored[:, [0, 2]] = (restored[:, [0, 2]] - geometry["pad_x"]) / geometry[
            "ratio"
        ]
        restored[:, [1, 3]] = (restored[:, [1, 3]] - geometry["pad_y"]) / geometry[
            "ratio"
        ]
    else:
        restored[:, [0, 2]] *= geometry["source_width"] / geometry["width"]
        restored[:, [1, 3]] *= geometry["source_height"] / geometry["height"]
    return restored


def _class_name(names: Optional[list[str]], cls: int) -> Optional[str]:
    """Resolve a non-negative class index through the optional label list."""
    return names[cls] if names and 0 <= cls < len(names) else None


def _class_id(value: Any, context: str, ignore_index: Optional[int] = None) -> int:
    """Validate one integral class ID before any lossy NumPy/Python conversion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise DecoderMappingError(f"{context} class ID must be an integer")
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number) or not number.is_integer():
            raise DecoderMappingError(
                f"{context} class ID must be a finite integer"
            )
    cls = int(value)
    if cls < 0 and cls != ignore_index:
        raise DecoderMappingError(f"{context} class ID must be non-negative")
    limits = np.iinfo(np.int64)
    if cls < limits.min or cls > limits.max:
        raise DecoderMappingError(f"{context} class ID exceeds the int64 range")
    return cls


def _custom_detection_contract(
    spec: Any, output_names: set[str]
) -> dict[str, Any]:
    """Parse every custom detection field before graph execution."""
    context = "decoder.output"
    spec = _mapping_fields(
        spec,
        {
            "prediction",
            "layout",
            "bbox_start",
            "bbox_encoding",
            "score_index",
            "class_index",
            "coordinate_space",
            "score_activation",
        },
        set(),
        context,
    )
    tensor = spec["prediction"]
    if not isinstance(tensor, str) or tensor not in output_names:
        raise DecoderMappingError(f"unknown output tensor: {tensor}")
    return {
        "tensor": tensor,
        "layout": _mapping_choice(spec, "layout", {"rows", "columns"}, context),
        "encoding": _mapping_choice(
            spec, "bbox_encoding", {"xyxy", "cxcywh"}, context
        ),
        "space": _mapping_choice(
            spec, "coordinate_space", {"pixels", "normalized"}, context
        ),
        "activation": _mapping_choice(
            spec, "score_activation", {"none", "sigmoid"}, context
        ),
        "bbox_start": _mapping_int(spec, "bbox_start", context),
        "score_index": _mapping_int(spec, "score_index", context),
        "class_index": _mapping_int(spec, "class_index", context),
    }


def _decode_custom_detection(
    outputs: dict[str, np.ndarray],
    geometry: dict[str, Any],
    spec: Any,
    names: Optional[list[str]],
    conf_threshold: float,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Decode a batch-1 matrix using only explicitly declared field indices."""
    contract = _custom_detection_contract(spec, set(outputs))
    tensor = contract["tensor"]
    layout = contract["layout"]
    encoding = contract["encoding"]
    space = contract["space"]
    activation = contract["activation"]
    bbox_start = contract["bbox_start"]
    score_index = contract["score_index"]
    class_index = contract["class_index"]
    matrix = outputs[tensor]
    if matrix.ndim == 3:
        if matrix.shape[0] != 1:
            raise DecoderMappingError(f"{tensor} must have batch 1")
        matrix = matrix[0]
    if matrix.ndim != 2:
        raise DecoderMappingError(f"{tensor} must be a 2D or batch-1 3D matrix")
    if layout == "columns":
        matrix = matrix.T
    fields = matrix.shape[1]
    required_indices = [bbox_start + offset for offset in range(4)]
    required_indices.extend([score_index, class_index])
    if not required_indices or max(required_indices) >= fields:
        raise DecoderMappingError(f"{tensor} does not contain all declared fields")

    boxes = matrix[:, bbox_start : bbox_start + 4].astype(np.float32)
    if encoding == "cxcywh":
        center = boxes.copy()
        boxes[:, 0] = center[:, 0] - center[:, 2] / 2
        boxes[:, 1] = center[:, 1] - center[:, 3] / 2
        boxes[:, 2] = center[:, 0] + center[:, 2] / 2
        boxes[:, 3] = center[:, 1] + center[:, 3] / 2
    scores = matrix[:, score_index].astype(np.float32)
    if activation == "sigmoid":
        scores = 1.0 / (1.0 + np.exp(-scores))
    selected = np.flatnonzero(np.isfinite(scores) & (scores >= conf_threshold))
    classes = np.asarray(
        [
            _class_id(value, tensor)
            for value in matrix[selected, class_index]
        ],
        dtype=np.int64,
    )
    boxes = _restore_boxes(boxes[selected], space, geometry)
    scores = scores[selected]
    results: list[dict[str, Any]] = []
    for cls in np.unique(classes):
        class_positions = np.flatnonzero(classes == cls)
        keep = _nms(boxes[class_positions], scores[class_positions], iou=iou_threshold)
        for position in class_positions[keep]:
            cls_id = int(cls)
            results.append(
                {
                    "xyxy": [float(value) for value in boxes[position]],
                    "score": float(scores[position]),
                    "cls": cls_id,
                    "name": _class_name(names, cls_id),
                }
            )
    return sorted(results, key=lambda result: -result["score"])


def _yolo_contract(manifest: dict) -> tuple[str, Optional[str], Optional[str]]:
    """Read explicit YOLO semantics without confusing raw heads with post-NMS output."""
    outputs = manifest.get("outputs") if isinstance(manifest, dict) else None
    contract = outputs.get("prediction") if isinstance(outputs, dict) else None
    if contract is None:
        contract = {}
    if not isinstance(contract, dict):
        raise DecoderMappingError("manifest outputs.prediction must be an object")
    output_format = contract.get("format", "raw_cxcywh")
    if output_format not in ("raw_cxcywh", "end2end_xyxy"):
        raise DecoderMappingError(
            "manifest YOLO format must be raw_cxcywh or end2end_xyxy"
        )
    layout = contract.get("layout")
    if layout is not None and layout not in ("rows", "columns"):
        raise DecoderMappingError("manifest YOLO layout must be rows or columns")
    coordinate_space = contract.get("coordinate_space")
    if coordinate_space is not None and coordinate_space not in ("pixels", "normalized"):
        raise DecoderMappingError(
            "manifest YOLO coordinate_space must be pixels or normalized"
        )
    return output_format, layout, coordinate_space


def _orient_yolo_matrix(
    matrix: np.ndarray, declared_layout: Optional[str], expected_fields: Optional[int]
) -> np.ndarray:
    """Resolve rows/columns only from a declaration or an unambiguous field contract."""
    if matrix.ndim == 3:
        if matrix.shape[0] != 1:
            raise DecoderMappingError("YOLO output must have batch 1")
        matrix = matrix[0]
    if matrix.ndim != 2:
        raise DecoderMappingError("YOLO output must be a batch-1 matrix")
    if declared_layout == "columns":
        matrix = matrix.T
    elif declared_layout is None:
        rows, columns = matrix.shape
        if expected_fields is not None:
            row_fields = columns == expected_fields
            column_fields = rows == expected_fields
            if row_fields == column_fields:
                raise DecoderMappingError(
                    "YOLO output layout is ambiguous for the declared class count"
                )
            if column_fields:
                matrix = matrix.T
        elif columns == 1 and 5 <= rows <= 208:
            matrix = matrix.T
        elif rows == 1 and 5 <= columns <= 208:
            pass
        elif rows == 0 and columns >= 5:
            pass
        elif columns == 0 and rows >= 5:
            matrix = matrix.T
        elif 5 <= columns <= 208 < rows:
            pass
        elif 5 <= rows <= 208 < columns:
            matrix = matrix.T
        else:
            raise DecoderMappingError(
                "YOLO output layout is ambiguous; declare outputs.prediction.layout "
                "or provide the class labels"
            )
    if expected_fields is not None and matrix.shape[1] != expected_fields:
        raise DecoderMappingError(
            f"YOLO output has {matrix.shape[1]} fields; expected {expected_fields} "
            "from the class labels"
        )
    return matrix


def _decode_yolo(
    outputs: dict[str, np.ndarray],
    geometry: dict[str, Any],
    names: Optional[list[str]],
    conf_threshold: float,
    iou_threshold: float,
    manifest: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """Decode declared YOLO raw and end-to-end forms without shape-only semantics."""
    if len(outputs) != 1:
        raise DecoderMappingError("yolo requires exactly one output tensor")
    output_format, declared_layout, coordinate_space = _yolo_contract(manifest or {})
    expected_fields = (
        len(names) + 4
        if output_format == "raw_cxcywh" and names
        else 6 if output_format == "end2end_xyxy" else None
    )
    matrix = _orient_yolo_matrix(
        next(iter(outputs.values())), declared_layout, expected_fields
    )
    if matrix.shape[1] < 5:
        raise DecoderMappingError("YOLO output has fewer than five fields")

    if output_format == "end2end_xyxy":
        if matrix.shape[1] != 6:
            raise DecoderMappingError("end-to-end YOLO output must have six fields")
        selected = np.flatnonzero(np.isfinite(matrix[:, 4]) & (matrix[:, 4] >= conf_threshold))
        matrix = matrix[selected]
        boxes = _restore_boxes(
            matrix[:, :4], coordinate_space or "pixels", geometry
        )
        results = []
        for index, row in enumerate(matrix):
            cls = _class_id(row[5], "end-to-end YOLO output")
            results.append(
                {
                    "xyxy": [float(value) for value in boxes[index]],
                    "score": float(row[4]),
                    "cls": cls,
                    "name": _class_name(names, cls),
                }
            )
        return sorted(results, key=lambda result: -result["score"])

    center = matrix[:, :4].astype(np.float32)
    boxes = np.column_stack(
        (
            center[:, 0] - center[:, 2] / 2,
            center[:, 1] - center[:, 3] / 2,
            center[:, 0] + center[:, 2] / 2,
            center[:, 1] + center[:, 3] / 2,
        )
    )
    class_scores = matrix[:, 4:]
    classes = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)
    selected = np.flatnonzero(np.isfinite(scores) & (scores >= conf_threshold))
    boxes = boxes[selected]
    classes = classes[selected]
    scores = scores[selected]
    # Raw YOLO exports use model-input pixels by default. Normalized geometry
    # must be declared rather than inferred from the current image's values.
    space = coordinate_space or "pixels"
    boxes = _restore_boxes(boxes, space, geometry)
    results = []
    for cls in np.unique(classes):
        positions = np.flatnonzero(classes == cls)
        keep = _nms(boxes[positions], scores[positions], iou=iou_threshold)
        for position in positions[keep]:
            cls_id = int(cls)
            results.append(
                {
                    "xyxy": [float(value) for value in boxes[position]],
                    "score": float(scores[position]),
                    "cls": cls_id,
                    "name": _class_name(names, cls_id),
                }
            )
    return sorted(results, key=lambda result: -result["score"])


def _decode_named_detection(
    outputs: dict[str, np.ndarray],
    geometry: dict[str, Any],
    names: Optional[list[str]],
    conf_threshold: float,
) -> list[dict[str, Any]]:
    """Decode the verified named boxes/scores/labels post-processing contract."""
    boxes = outputs["boxes"]
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise DecoderMappingError("named detection boxes must be [N,4] or [1,N,4]")
    vectors = {}
    for role in ("scores", "labels"):
        vector = outputs[role]
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        elif vector.ndim == 2 and vector.shape[1] == 1:
            vector = vector[:, 0]
        if vector.ndim != 1 or len(vector) != len(boxes):
            raise DecoderMappingError(f"named detection {role} must align with boxes")
        vectors[role] = vector
    restored = _restore_boxes(boxes, "pixels", geometry)
    results = []
    for index, box in enumerate(restored):
        score = float(vectors["scores"][index])
        if not np.isfinite(score) or score < conf_threshold:
            continue
        cls = _class_id(vectors["labels"][index], "named detection labels")
        results.append(
            {
                "xyxy": [float(value) for value in box],
                "score": score,
                "cls": cls,
                "name": _class_name(names, cls),
            }
        )
    return sorted(results, key=lambda result: -result["score"])


def _instance_segmentation_contract(
    spec: Any, output_names: set[str]
) -> dict[str, Any]:
    """Parse all instance-segmentation tensor roles and scalar options."""
    context = "decoder.output"
    spec = _mapping_fields(
        spec,
        {
            "boxes",
            "scores",
            "labels",
            "masks",
            "bbox_encoding",
            "coordinate_space",
            "mask_activation",
            "mask_threshold",
        },
        set(),
        context,
    )
    tensor_names = {}
    for role in ("boxes", "scores", "labels", "masks"):
        tensor = spec[role]
        if not isinstance(tensor, str) or tensor not in output_names:
            raise DecoderMappingError(f"unknown output tensor for {role}: {tensor}")
        tensor_names[role] = tensor
    return {
        "tensor_names": tensor_names,
        "encoding": _mapping_choice(
            spec, "bbox_encoding", {"xyxy", "cxcywh"}, context
        ),
        "space": _mapping_choice(
            spec, "coordinate_space", {"pixels", "normalized"}, context
        ),
        "activation": _mapping_choice(
            spec, "mask_activation", {"sigmoid", "none"}, context
        ),
        "threshold": _mapping_number(spec, "mask_threshold", context),
    }


def _decode_instance_segmentation(
    outputs: dict[str, np.ndarray],
    geometry: dict[str, Any],
    spec: Any,
    names: Optional[list[str]],
    conf_threshold: float,
) -> list[dict[str, Any]]:
    """Decode named boxes/scores/labels and NHW or N1HW instance masks."""
    contract = _instance_segmentation_contract(spec, set(outputs))
    tensor_names = contract["tensor_names"]
    encoding = contract["encoding"]
    space = contract["space"]
    activation = contract["activation"]
    threshold = contract["threshold"]

    boxes = outputs[tensor_names["boxes"]]
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise DecoderMappingError("instance boxes must be [N,4] or [1,N,4]")
    vectors = {}
    for role in ("scores", "labels"):
        vector = outputs[tensor_names[role]]
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        elif vector.ndim == 2 and vector.shape[1] == 1:
            vector = vector[:, 0]
        if vector.ndim != 1 or len(vector) != len(boxes):
            raise DecoderMappingError(f"instance {role} must contain one value per box")
        vectors[role] = vector
    masks = outputs[tensor_names["masks"]]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or len(masks) != len(boxes):
        raise DecoderMappingError(
            "instance masks must be NHW or N1HW with one mask per box"
        )

    boxes = boxes.astype(np.float32)
    if encoding == "cxcywh":
        center = boxes.copy()
        boxes[:, 0] = center[:, 0] - center[:, 2] / 2
        boxes[:, 1] = center[:, 1] - center[:, 3] / 2
        boxes[:, 2] = center[:, 0] + center[:, 2] / 2
        boxes[:, 3] = center[:, 1] + center[:, 3] / 2
    restored_boxes = _restore_boxes(boxes, space, geometry)
    results = []
    for index in range(len(boxes)):
        score = float(vectors["scores"][index])
        if not np.isfinite(score) or score < conf_threshold:
            continue
        cls = _class_id(vectors["labels"][index], "instance labels")
        mask = masks[index].astype(np.float32)
        if activation == "sigmoid":
            mask = 1.0 / (1.0 + np.exp(-mask))
        mask = _restore_mask(mask, geometry, nearest=False)
        binary = (mask >= threshold).astype(np.uint8) * 255
        results.append(
            {
                "xyxy": [float(value) for value in restored_boxes[index]],
                "score": score,
                "cls": cls,
                "name": _class_name(names, cls),
                "mask": binary,
            }
        )
    return sorted(results, key=lambda result: -result["score"])


def _restore_mask(
    mask: np.ndarray, geometry: dict[str, Any], *, nearest: bool
) -> np.ndarray:
    """Undo input resize/padding for a class map or instance mask."""
    import cv2

    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    restored = np.asarray(mask)
    if restored.shape != (geometry["height"], geometry["width"]):
        restored = cv2.resize(
            restored.astype(np.float32),
            (geometry["width"], geometry["height"]),
            interpolation=interpolation,
        )
    if geometry["resize"] == "letterbox":
        content_w = int(round(geometry["source_width"] * geometry["ratio"]))
        content_h = int(round(geometry["source_height"] * geometry["ratio"]))
        left = geometry["pad_x"]
        top = geometry["pad_y"]
        restored = restored[top : top + content_h, left : left + content_w]
    return cv2.resize(
        restored.astype(np.float32),
        (geometry["source_width"], geometry["source_height"]),
        interpolation=interpolation,
    )


def _semantic_contract(spec: Any, output_names: set[str]) -> dict[str, Any]:
    """Parse semantic output semantics, making class axes conditional on format."""
    context = "decoder.output"
    spec = _mapping_fields(
        spec,
        {"output", "format"},
        {"class_axis", "ignore_index"},
        context,
    )
    tensor = spec["output"]
    if not isinstance(tensor, str) or tensor not in output_names:
        raise DecoderMappingError(f"unknown output tensor: {tensor}")
    output_format = _mapping_choice(spec, "format", {"logits", "class_ids"}, context)
    if output_format == "logits" and "class_axis" not in spec:
        raise DecoderMappingError(f"{context} missing fields: class_axis")
    axis = None
    if "class_axis" in spec:
        raw_axis = spec["class_axis"]
        if isinstance(raw_axis, bool) or not isinstance(raw_axis, (int, np.integer)):
            raise DecoderMappingError(f"{context}.class_axis must be an integer")
        axis = int(raw_axis)
    ignore_index = None
    if "ignore_index" in spec:
        raw_ignore = spec["ignore_index"]
        if isinstance(raw_ignore, bool) or not isinstance(
            raw_ignore, (int, np.integer)
        ):
            raise DecoderMappingError(f"{context}.ignore_index must be an integer")
        ignore_index = int(raw_ignore)
    return {
        "tensor": tensor,
        "format": output_format,
        "axis": axis,
        "ignore_index": ignore_index,
    }


def _decode_semantic(
    outputs: dict[str, np.ndarray],
    geometry: dict[str, Any],
    spec: Any,
) -> tuple[np.ndarray, Optional[int]]:
    """Decode logits or an explicit NHW/N1HW class-ID map without lossy coercion."""
    contract = _semantic_contract(spec, set(outputs))
    tensor = contract["tensor"]
    output_format = contract["format"]
    raw_axis = contract["axis"]
    ignore_index = contract["ignore_index"]
    values = outputs[tensor]
    if output_format == "logits":
        if values.ndim not in (3, 4):
            raise DecoderMappingError(
                f"{tensor} logits must be rank 3 or batch-1 rank 4"
            )
        if values.ndim == 4 and values.shape[0] != 1:
            raise DecoderMappingError(f"{tensor} must have batch 1")
        axis = int(raw_axis)
        if axis < 0:
            axis += values.ndim
        if axis < 0 or axis >= values.ndim or (values.ndim == 4 and axis == 0):
            raise DecoderMappingError(f"decoder.output.class_axis is invalid for {tensor}")
        if values.shape[axis] == 0:
            raise DecoderMappingError(f"{tensor} class axis must not be empty")
        class_ids = np.argmax(values, axis=axis)
    else:
        class_ids = values
        if class_ids.ndim == 4:
            if class_ids.shape[0] != 1:
                raise DecoderMappingError(f"{tensor} must have batch 1")
            if raw_axis is None:
                raise DecoderMappingError(
                    "rank-4 class IDs require a declared singleton class_axis"
                )
            axis = raw_axis + class_ids.ndim if raw_axis < 0 else raw_axis
            if axis <= 0 or axis >= class_ids.ndim:
                raise DecoderMappingError(
                    f"decoder.output.class_axis is invalid for {tensor}"
                )
            if class_ids.shape[axis] != 1:
                raise DecoderMappingError(
                    f"{tensor} declared class axis must be singleton"
                )
            class_ids = np.squeeze(class_ids, axis=axis)
        elif raw_axis is not None:
            raise DecoderMappingError(
                "class_axis is only valid for rank-4 singleton class-ID maps"
            )
    if class_ids.ndim == 3:
        if class_ids.shape[0] != 1:
            raise DecoderMappingError(f"{tensor} must have batch 1")
        class_ids = class_ids[0]
    if class_ids.ndim != 2 or not all(size > 0 for size in class_ids.shape):
        raise DecoderMappingError(f"{tensor} must decode to one non-empty 2D class map")
    for value in np.unique(class_ids):
        _class_id(value, tensor, ignore_index=ignore_index)
    class_ids = class_ids.astype(np.int64)
    restored = _restore_mask(class_ids, geometry, nearest=True).astype(np.int64)
    return restored, ignore_index


def _classification_contract(
    spec: Any, output_names: set[str], top_k: int
) -> dict[str, Any]:
    """Parse classification fields and Top-K before graph execution."""
    context = "decoder.output"
    spec = _mapping_fields(
        spec,
        {"output", "class_axis", "activation"},
        set(),
        context,
    )
    tensor = spec["output"]
    if not isinstance(tensor, str) or tensor not in output_names:
        raise DecoderMappingError(f"unknown output tensor: {tensor}")
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise DecoderMappingError("top_k must be a positive integer")
    raw_axis = spec["class_axis"]
    if isinstance(raw_axis, bool) or not isinstance(raw_axis, (int, np.integer)):
        raise DecoderMappingError(f"{context}.class_axis must be an integer")
    return {
        "tensor": tensor,
        "axis": int(raw_axis),
        "activation": _mapping_choice(
            spec, "activation", {"softmax", "sigmoid", "none"}, context
        ),
    }


def _decode_classification(
    outputs: dict[str, np.ndarray],
    spec: Any,
    names: Optional[list[str]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Activate and rank one explicitly mapped batch-1 classification tensor."""
    contract = _classification_contract(spec, set(outputs), top_k)
    tensor = contract["tensor"]
    raw_axis = contract["axis"]
    activation = contract["activation"]
    values = outputs[tensor].astype(np.float32)
    if values.ndim < 2 or values.shape[0] != 1:
        raise DecoderMappingError(f"{tensor} must have batch 1")
    axis = int(raw_axis)
    if axis < 0:
        axis += values.ndim
    if axis <= 0 or axis >= values.ndim:
        raise DecoderMappingError(
            f"decoder.output.class_axis must identify a non-batch axis of {tensor}"
        )
    if values.shape[axis] == 0:
        raise DecoderMappingError(f"{tensor} class axis must not be empty")
    if activation == "softmax":
        shifted = values - values.max(axis=axis, keepdims=True)
        exponentials = np.exp(shifted)
        values = exponentials / exponentials.sum(axis=axis, keepdims=True)
    elif activation == "sigmoid":
        values = 1.0 / (1.0 + np.exp(-values))

    unbatched = values[0]
    class_axis = axis - 1
    for index, size in enumerate(unbatched.shape):
        if index != class_axis and size != 1:
            raise DecoderMappingError(
                f"{tensor} has non-singleton dimensions outside batch/class axes"
            )
    classes = np.moveaxis(unbatched, class_axis, 0).reshape(-1)
    order = classes.argsort()[::-1][: min(int(top_k), len(classes))]
    return [
        {
            "cls": int(index),
            "name": _class_name(names, int(index)),
            "prob": float(classes[index]),
        }
        for index in order
    ]


def _validate_custom_output(
    task: str, spec: Any, sess: Any, top_k: int
) -> None:
    """Validate declarative output semantics against graph metadata before run()."""
    output_names = {output.name for output in sess.get_outputs()}
    if task == "detection":
        _custom_detection_contract(spec, output_names)
    elif task == "instance_segmentation":
        _instance_segmentation_contract(spec, output_names)
    elif task == "semantic_segmentation":
        _semantic_contract(spec, output_names)
    elif task == "classification":
        _classification_contract(spec, output_names, top_k)
    else:
        raise DecoderMappingError(f"unsupported custom task: {task}")


def _manifest_scale(manifest: dict) -> float:
    """Read the declared pixel scale, accepting the trainer's ``1/255`` form."""
    preprocess = manifest.get("preprocess") if isinstance(manifest, dict) else None
    raw = preprocess.get("scale", 1 / 255) if isinstance(preprocess, dict) else 1 / 255
    if isinstance(raw, str):
        parts = raw.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (TypeError, ValueError, ZeroDivisionError):
                return 1 / 255
        try:
            return float(raw)
        except ValueError:
            return 1 / 255
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1 / 255


def _input_shape_hint(input_shape: Any, layout: str) -> Optional[tuple[int, int]]:
    """Extract only an H/W default hint; fixed graph dimensions still win."""
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 4:
        return None
    hw = input_shape[2:4] if layout == "NCHW" else input_shape[1:3]
    if all(isinstance(value, (int, np.integer)) and int(value) > 0 for value in hw):
        return int(hw[1]), int(hw[0])
    return None


def _builtin_input(
    sess: Any,
    image: np.ndarray,
    manifest: dict,
    task: str,
    input_shape: Any = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build the declared/default preprocessing contract for a verified built-in."""
    inputs = list(sess.get_inputs())
    if not inputs:
        raise DecoderMappingError("ONNX graph has no inputs")
    input_spec = manifest.get("input") if isinstance(manifest, dict) else {}
    if not isinstance(input_spec, dict):
        input_spec = {}
    by_name = {item.name: item for item in inputs}
    declared_tensor = input_spec.get("name")
    if declared_tensor is None:
        tensor = inputs[0].name
    elif not isinstance(declared_tensor, str) or declared_tensor not in by_name:
        raise DecoderMappingError(
            f"declared input tensor is not present in the graph: {declared_tensor}"
        )
    else:
        tensor = declared_tensor
    shape = by_name[tensor].shape
    declared_layout = input_spec.get("layout")
    if declared_layout in ("NCHW", "NHWC"):
        layout = declared_layout
    elif isinstance(shape, (list, tuple)) and len(shape) == 4 and shape[-1] == 3:
        layout = "NHWC"
    else:
        layout = "NCHW"
    fixed_hw = None
    if isinstance(shape, (list, tuple)) and len(shape) == 4:
        raw_hw = shape[2:4] if layout == "NCHW" else shape[1:3]
        if all(isinstance(value, int) and value > 0 for value in raw_hw):
            fixed_hw = int(raw_hw[1]), int(raw_hw[0])
    default_size = (
        224
        if task == "classification"
        else 512
        if task == "semantic_segmentation"
        else 640
    )
    size = _manifest_input_size(manifest, default_size)
    hinted_hw = _input_shape_hint(input_shape, layout)
    width, height = fixed_hw or hinted_hw or (size, size)
    if task in ("detection", "instance_segmentation"):
        mean, std = _manifest_pixel_norm(manifest)
        resize = "letterbox"
    else:
        mean, std = _manifest_norm(manifest)
        resize = "stretch"
    spec = {
        "tensor": tensor,
        "width": width,
        "height": height,
        "layout": layout,
        "color": input_spec.get("color")
        if input_spec.get("color") in ("RGB", "BGR")
        else "RGB",
        "resize": resize,
        "scale": _manifest_scale(manifest),
        "mean": list(mean),
        "std": list(std),
    }
    return _custom_input(sess, image, spec)


DECODER_REGISTRY = (
    {"id": "yunet", "label": "OpenCV YuNet", "task": "detection"},
    {"id": "yolo", "label": "YOLO single-tensor", "task": "detection"},
    {"id": "named_detection", "label": "Named detection outputs", "task": "detection"},
    {
        "id": "mask_rcnn",
        "label": "Mask R-CNN named outputs",
        "task": "instance_segmentation",
    },
    {
        "id": "torchvision_classification",
        "label": "Torchvision classification",
        "task": "classification",
    },
    {
        "id": "semantic_logits",
        "label": "Semantic logits (NCHW)",
        "task": "semantic_segmentation",
    },
)
"""Stable built-in decoder descriptors exposed by model inspection."""


class DecoderMappingRequired(ValueError):
    """Signal that automatic execution has no semantics-safe decoder match."""

    code = "decoder_mapping_required"

    def __init__(self) -> None:
        super().__init__("decoder mapping required")


def inspect_decoders(
    sess: Any,
    manifest: Optional[dict] = None,
    *,
    source: str = "",
    architecture: str = "",
) -> dict[str, Any]:
    """Return built-in candidates and the unique automatic decoder choice.

    Named output signatures and explicit model-family metadata are the only
    automatic signals; tensor ranks and dimensions never select a task.
    """
    manifest = manifest or {}
    output_names = {output.name for output in sess.get_outputs()}
    task = manifest_task(manifest).strip().lower()
    architecture_hint = str(manifest.get("architecture") or "")
    family = " ".join((source, architecture, architecture_hint)).strip().lower()
    instance_declared = task in (
        "segment",
        "segmentation",
        "instance_seg",
        "instance_segmentation",
    )
    semantic_declared = task in ("semantic_seg", "semantic_segmentation")
    yunet_modern = set(_YUNET_OUTPUT_NAMES).issubset(output_names)
    yunet_legacy = set(_YUNET_LEGACY_OUTPUT_NAMES).issubset(output_names)
    matched = {
        # A graph carrying both complete contracts is ambiguous, not a reason to
        # let output order choose between two different anchor geometries.
        "yunet": yunet_modern != yunet_legacy,
        "mask_rcnn": {"boxes", "scores", "labels", "masks"}.issubset(output_names),
        "named_detection": (
            {"boxes", "scores", "labels"}.issubset(output_names)
            and "masks" not in output_names
            and not instance_declared
            and not semantic_declared
        ),
        "semantic_logits": task == "semantic_seg",
        "torchvision_classification": (
            task in ("classify", "classification", "multilabel")
            or source.strip().lower() == "torchvision"
        ),
        "yolo": (
            not semantic_declared
            and not instance_declared
            and "yunet" not in family
            and (
                task in ("detect", "detection")
                or "yolo" in family
                or source.strip().lower()
                in ("ultralytics", "paddle", "paddledetection")
            )
        ),
    }
    priority = (
        "yunet",
        "mask_rcnn",
        "named_detection",
        "semantic_logits",
        "torchvision_classification",
        "yolo",
    )
    decoders = [
        {**descriptor, "matched": matched.get(descriptor["id"], False)}
        for descriptor in DECODER_REGISTRY
    ]
    selected = next(
        (
            decoder
            for decoder_id in priority
            for decoder in decoders
            if decoder["id"] == decoder_id and decoder["matched"]
        ),
        None,
    )
    auto = None
    if selected is not None:
        auto = {"id": selected["id"], "task": selected["task"]}
    return {
        "decoders": decoders,
        "auto_decoder": auto,
        "mapping_required": auto is None,
    }


def run_onnx_trial(
    sess: Any,
    image: np.ndarray,
    *,
    manifest: Optional[dict] = None,
    decoder: Optional[dict] = None,
    names: Optional[list[str]] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    top_k: int = 5,
    source: str = "",
    architecture: str = "",
    input_shape: Any = None,
) -> dict[str, Any]:
    """Execute one image through an automatically or explicitly selected decoder."""
    request = decoder or {"mode": "auto"}
    if not isinstance(request, dict):
        raise DecoderMappingError("decoder must be an object")
    mode = str(request.get("mode") or "auto")
    if mode == "custom":
        _mapping_fields(
            request,
            {"mode", "task", "input", "output"},
            {"labels"},
            "decoder",
        )
        task = _mapping_choice(
            request,
            "task",
            {
                "detection",
                "instance_segmentation",
                "semantic_segmentation",
                "classification",
            },
            "decoder",
        )
        custom_names = request.get("labels")
        if custom_names is not None:
            if not isinstance(custom_names, list) or not all(
                isinstance(label, str) for label in custom_names
            ):
                raise DecoderMappingError("decoder.labels must be a string array")
            names = custom_names
        _validate_custom_output(task, request["output"], sess, top_k)
        outputs, geometry = _custom_input(sess, image, request["input"])
        if task == "detection":
            return {
                "kind": task,
                "decoder": {"mode": mode, "id": "custom_detection", "task": task},
                "detections": _decode_custom_detection(
                    outputs,
                    geometry,
                    request["output"],
                    names,
                    conf_threshold,
                    iou_threshold,
                ),
            }
        if task == "classification":
            return {
                "kind": task,
                "decoder": {"mode": mode, "id": "custom_classification", "task": task},
                "predictions": _decode_classification(
                    outputs, request["output"], names, top_k
                ),
            }
        if task == "instance_segmentation":
            return {
                "kind": task,
                "decoder": {
                    "mode": mode,
                    "id": "custom_instance_segmentation",
                    "task": task,
                },
                "detections": _decode_instance_segmentation(
                    outputs,
                    geometry,
                    request["output"],
                    names,
                    conf_threshold,
                ),
            }
        if task == "semantic_segmentation":
            mask, ignore_index = _decode_semantic(outputs, geometry, request["output"])
            return {
                "kind": task,
                "decoder": {
                    "mode": mode,
                    "id": "custom_semantic_segmentation",
                    "task": task,
                },
                "mask": mask,
                "ignore_index": ignore_index,
            }
        raise DecoderMappingError(f"unsupported custom task: {task}")
    inspection = inspect_decoders(
        sess, manifest, source=source, architecture=architecture
    )
    if mode == "auto":
        selected = inspection["auto_decoder"]
        if selected is None:
            raise DecoderMappingRequired()
        decoder_id = selected["id"]
    elif mode == "builtin":
        decoder_id = str(request.get("decoder_id") or "")
        if decoder_id not in {item["id"] for item in DECODER_REGISTRY}:
            raise DecoderMappingError(f"unknown decoder: {decoder_id}")
    else:
        raise DecoderMappingError(f"unsupported decoder mode: {mode}")

    descriptor = next(item for item in DECODER_REGISTRY if item["id"] == decoder_id)
    kind = descriptor["task"]
    if decoder_id == "yunet":
        available = {output.name for output in sess.get_outputs()}
        modern = set(_YUNET_OUTPUT_NAMES).issubset(available)
        legacy = set(_YUNET_LEGACY_OUTPUT_NAMES).issubset(available)
        if modern == legacy:
            raise DecoderMappingError(
                "yunet requires exactly one complete output contract: "
                "cls/obj/bbox/kps heads or loc/conf/iou"
            )
        detect = _detect_yunet if modern else _detect_yunet_legacy
        return {
            "kind": kind,
            "decoder": {"mode": mode, "id": decoder_id, "task": kind},
            "detections": detect(
                sess,
                image,
                conf_threshold,
                names,
                iou_threshold=iou_threshold,
            ),
        }
    if decoder_id == "semantic_logits":
        declared_outputs = (manifest or {}).get("outputs") or {}
        if not isinstance(declared_outputs, dict):
            raise DecoderMappingError("manifest outputs must be an object")
        declared_contract = declared_outputs.get("logits") or {}
        if not isinstance(declared_contract, dict):
            raise DecoderMappingError("manifest outputs.logits must be an object")
        declared_tensor = declared_contract.get("tensor")
        available_outputs = {output.name for output in sess.get_outputs()}
        if declared_tensor is not None and (
            not isinstance(declared_tensor, str)
            or declared_tensor not in available_outputs
        ):
            raise DecoderMappingError(
                f"declared logits output tensor is not present in the graph: "
                f"{declared_tensor}"
            )
    outputs, geometry = _builtin_input(
        sess, image, manifest or {}, kind, input_shape=input_shape
    )
    if decoder_id == "torchvision_classification":
        if "logits" in outputs:
            tensor = "logits"
        elif outputs:
            # The legacy classifier contract puts logits first and may append
            # auxiliary feature embeddings that are not classification outputs.
            tensor = next(iter(outputs))
        else:
            raise DecoderMappingError("torchvision_classification requires an output")
        activation = (
            "sigmoid"
            if manifest_task(manifest or {}).lower() == "multilabel"
            else "softmax"
        )
        return {
            "kind": kind,
            "decoder": {"mode": mode, "id": decoder_id, "task": kind},
            "predictions": _decode_classification(
                outputs,
                {"output": tensor, "class_axis": -1, "activation": activation},
                names,
                top_k,
            ),
        }
    if decoder_id == "semantic_logits":
        declared_outputs = (manifest or {}).get("outputs") or {}
        contract = declared_outputs.get("logits") or {}
        if not isinstance(contract, dict):
            contract = {}
        declared_tensor = contract.get("tensor")
        if isinstance(declared_tensor, str):
            tensor = declared_tensor
        elif "logits" in outputs:
            tensor = "logits"
        elif len(outputs) == 1:
            tensor = next(iter(outputs))
        else:
            raise DecoderMappingError(
                "semantic_logits requires a declared logits output or one output tensor"
            )
        layout = contract.get("layout", "NCHW")
        output_spec = {
            "output": tensor,
            "format": "logits",
            "class_axis": -1 if layout == "NHWC" else 1,
        }
        ignore_index = contract.get(
            "ignore_index", (manifest or {}).get("ignore_index")
        )
        if ignore_index is not None:
            output_spec["ignore_index"] = ignore_index
        mask, resolved_ignore = _decode_semantic(outputs, geometry, output_spec)
        return {
            "kind": kind,
            "decoder": {"mode": mode, "id": decoder_id, "task": kind},
            "mask": mask,
            "ignore_index": resolved_ignore,
        }
    if decoder_id == "yolo":
        return {
            "kind": kind,
            "decoder": {"mode": mode, "id": decoder_id, "task": kind},
            "detections": _decode_yolo(
                outputs, geometry, names, conf_threshold, iou_threshold, manifest
            ),
        }
    if decoder_id == "named_detection":
        missing = {"boxes", "scores", "labels"} - set(outputs)
        if missing:
            raise DecoderMappingError(
                f"named_detection missing outputs: {', '.join(sorted(missing))}"
            )
        return {
            "kind": kind,
            "decoder": {"mode": mode, "id": decoder_id, "task": kind},
            "detections": _decode_named_detection(
                outputs, geometry, names, conf_threshold
            ),
        }
    if decoder_id == "mask_rcnn":
        output_spec = {
            "boxes": "boxes",
            "scores": "scores",
            "labels": "labels",
            "masks": "masks",
            "bbox_encoding": "xyxy",
            "coordinate_space": "pixels",
            "mask_activation": "none",
            "mask_threshold": 0.5,
        }
        return {
            "kind": kind,
            "decoder": {"mode": mode, "id": decoder_id, "task": kind},
            "detections": _decode_instance_segmentation(
                outputs, geometry, output_spec, names, conf_threshold
            ),
        }
    raise DecoderMappingRequired()
