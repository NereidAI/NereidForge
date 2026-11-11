# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Inspect and execute ONNX image trials through the decoder registry.

The routes resolve only the uploaded artifact and image. Model semantics come
from a verified automatic named/family contract, an explicit built-in choice,
or a declarative custom input/output mapping; tensor rank never selects a task.

    POST /try/inspect
    {"model_path": "s3://bucket/models/<id>/model.onnx", "source": ...,
     "architecture": ..., "input_shape": [1, 3, 640, 640]}

    POST /try/predict
    {"image_url" | "image_base64": ..., "model_path": ...,
     "decoder": {"mode": "auto"}, "conf": 0.25, "iou": 0.45,
     "top_k": 5, "names": {"0": "person"}}

Grad-CAM is intentionally unavailable because ONNX Runtime has no gradients.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
import urllib.request
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TrialRequestError(ValueError):
    """Reject malformed HTTP trial options before decoder or inference execution."""


def _load_onnx_session(model_ref: str):
    """Resolve an s3:// / local ONNX reference and return a session + path."""
    from control_models.base import ControlModel
    from onnx_predict import get_session

    if model_ref.startswith("s3://") or not os.path.isabs(model_ref):
        resolved = ControlModel.resolve_local_model_file(model_ref)
        if not resolved:
            raise FileNotFoundError(f"model not available: {model_ref}")
    else:
        resolved = model_ref
    if not resolved.lower().endswith(".onnx"):
        raise ValueError(
            f"unsupported model artifact '{model_ref}': the ML backend consumes "
            "ONNX exports only (export the model in the NF UI)."
        )
    return get_session(resolved), resolved


def _decode_image(body: dict) -> Optional[Any]:
    """Read the request image (base64 bytes or URL) into an OpenCV BGR array."""
    import cv2

    b64 = str(body.get("image_base64") or "")
    url = str(body.get("image_url") or "")
    tmp = None
    try:
        if b64:
            raw = base64.b64decode(b64)
            return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if not url:
            return None
        tmp = tempfile.mktemp(suffix=".jpg")
        req = urllib.request.Request(url, headers={"User-Agent": "nf-ml-backend"})
        with (
            urllib.request.urlopen(req, timeout=30) as response,
            open(tmp, "wb") as output,
        ):
            output.write(response.read())
        return cv2.imread(tmp)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def _names(session: Any, path: str, body: dict, manifest: dict) -> dict[int, str]:
    """Resolve labels as request override > manifest/embedded model metadata."""
    del session
    raw = body.get("names") or {}
    if raw:
        if not isinstance(raw, dict):
            raise TrialRequestError("names must be an object")
        labels = {}
        for key, value in raw.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise TrialRequestError(
                    "names keys must be non-negative integer indices"
                ) from exc
            if isinstance(key, bool) or index < 0 or index in labels:
                raise TrialRequestError(
                    "names keys must be non-negative integer indices"
                )
            labels[index] = str(value)
        return labels

    from onnx_predict import read_onnx_names

    labels = read_onnx_names(path) or list(manifest.get("labels") or [])
    return {index: str(name) for index, name in enumerate(labels)}


def _tensor_info(tensor: Any) -> dict[str, Any]:
    """Describe one ONNX input/output without altering symbolic dimensions."""
    shape = getattr(tensor, "shape", [])
    dtype = getattr(tensor, "type", None) or getattr(tensor, "dtype", "")
    return {
        "name": str(tensor.name),
        "shape": list(shape) if isinstance(shape, (list, tuple)) else [],
        "dtype": str(dtype),
    }


def _label(label_map: dict[int, str], cls: int, fallback: Any = None) -> str:
    """Resolve a response label while retaining numeric labels as a last resort."""
    return str(label_map.get(cls, fallback if fallback not in (None, "") else cls))


def _box_payload(detection: dict, label_map: dict[int, str]) -> dict:
    """Convert one normalized detection to the legacy image-trial box DTO."""
    cls = int(detection.get("cls", 0))
    return {
        "bbox": [round(float(value), 2) for value in detection["xyxy"]],
        "score": round(float(detection["score"]), 5),
        "cls": cls,
        "label": _label(label_map, cls, detection.get("name")),
    }


def _instance_polygons(detections: list[dict], label_map: dict[int, str]) -> list[dict]:
    """Convert retained original-resolution masks into the existing polygon DTO."""
    import cv2

    polygons = []
    for detection in detections:
        contours, _ = cv2.findContours(
            detection["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, 0.002 * cv2.arcLength(contour, True), True)
        if len(approx) < 3:
            continue
        cls = int(detection.get("cls", 0))
        polygons.append(
            {
                "score": round(float(detection["score"]), 5),
                "cls": cls,
                "label": _label(label_map, cls, detection.get("name")),
                "points": [
                    [round(float(point[0][0]), 2), round(float(point[0][1]), 2)]
                    for point in approx
                ],
            }
        )
    return polygons


_SEMANTIC_COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)


def _semantic_color(cls: int) -> list[int]:
    """Return a stable RGBA color; background and ignored pixels stay transparent."""
    if cls == 0:
        return [0, 0, 0, 0]
    red, green, blue = _SEMANTIC_COLORS[(cls - 1) % len(_SEMANTIC_COLORS)]
    return [red, green, blue, 128]


def _semantic_payload(
    mask: np.ndarray,
    ignore_index: Optional[int],
    label_map: dict[int, str],
) -> dict[str, Any]:
    """Render an original-resolution RGBA PNG and its present-class legend."""
    import cv2

    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    legend = []
    for cls in sorted(int(value) for value in np.unique(mask)):
        if ignore_index is not None and cls == ignore_index:
            continue
        color = _semantic_color(cls)
        rgba[mask == cls] = color
        legend.append({"cls": cls, "label": _label(label_map, cls), "color": color})
    encoded, png = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not encoded:
        raise ValueError("cannot encode semantic mask overlay")
    return {
        "mask_overlay": base64.b64encode(png.tobytes()).decode("ascii"),
        "mask_legend": legend,
    }


def format_trial_payload(
    image: np.ndarray,
    result: dict[str, Any],
    label_map: dict[int, str],
    top_k: int,
) -> dict[str, Any]:
    """Normalize every registry result into the public image-trial HTTP DTO."""
    kind = result["kind"]
    height, width = image.shape[:2]
    legacy_task = {
        "detection": "detect",
        "instance_segmentation": "segment",
        "semantic_segmentation": "segment",
        "classification": "classify",
    }[kind]
    payload = {
        "task": legacy_task,
        "kind": kind,
        "decoder": result["decoder"],
        "width": int(width),
        "height": int(height),
        "names": {str(key): value for key, value in label_map.items()},
    }
    if kind in ("detection", "instance_segmentation"):
        detections = result["detections"]
        payload["boxes"] = [
            _box_payload(detection, label_map) for detection in detections
        ]
        payload["probs"] = None
        if kind == "instance_segmentation":
            payload["polygons"] = _instance_polygons(detections, label_map)
        return payload
    if kind == "semantic_segmentation":
        payload.update(
            _semantic_payload(result["mask"], result.get("ignore_index"), label_map)
        )
        payload["boxes"] = []
        payload["polygons"] = []
        payload["probs"] = None
        return payload

    predictions = result["predictions"]
    if not predictions:
        from onnx_predict import DecoderMappingError

        raise DecoderMappingError("classification output contains no classes")
    best = predictions[0]
    best_cls = int(best["cls"])
    top5 = []
    for prediction in predictions[:5]:
        cls = int(prediction["cls"])
        top5.append(
            [
                _label(label_map, cls, prediction.get("name")),
                round(float(prediction["prob"]), 5),
            ]
        )
    topk = []
    for prediction in predictions[:top_k]:
        cls = int(prediction["cls"])
        topk.append(
            {
                "class_index": cls,
                "label": _label(label_map, cls, prediction.get("name")),
                "score": round(float(prediction["prob"]), 5),
            }
        )
    payload["boxes"] = []
    payload["probs"] = {
        "class_index": best_cls,
        "label": _label(label_map, best_cls, best.get("name")),
        "score": round(float(best["prob"]), 5),
        "top5": top5,
        "topk": topk,
    }
    return payload


def _label_list(label_map: dict[int, str]) -> list[str]:
    """Convert sparse label maps to the index-addressed registry input form."""
    if not label_map:
        return []
    return [label_map.get(index, "") for index in range(max(label_map) + 1)]


def register_try_routes(app) -> None:  # type: ignore[no-untyped-def]
    """Attach the inspection and decoder-registry image-trial Flask routes."""
    from flask import request

    @app.post("/try/inspect")
    def try_inspect() -> Any:  # noqa: ANN202
        body = request.get_json(force=True, silent=True) or {}
        model_ref = str(body.get("model_path") or "")
        if not model_ref:
            return {"error": "model_path required"}, 400
        try:
            session, path = _load_onnx_session(model_ref)
            from onnx_predict import inspect_decoders, read_onnx_manifest

            manifest = read_onnx_manifest(path)
            inspection = inspect_decoders(
                session,
                manifest,
                source=str(body.get("source") or ""),
                architecture=str(body.get("architecture") or ""),
            )
            label_map = _names(session, path, body, manifest)
            return {
                "artifact": model_ref,
                "inputs": [_tensor_info(tensor) for tensor in session.get_inputs()],
                "outputs": [_tensor_info(tensor) for tensor in session.get_outputs()],
                **inspection,
                "names": {str(key): value for key, value in label_map.items()},
            }
        except TrialRequestError as exc:
            return {"error": str(exc)}, 400
        except Exception as exc:  # noqa: BLE001
            logger.exception("try inspect failed")
            return {"error": f"inspection failed: {exc}"}, 500

    @app.post("/try/predict")
    def try_predict() -> Any:  # noqa: ANN202
        body = request.get_json(force=True, silent=True) or {}
        image = _decode_image(body)
        if image is None:
            return {"error": "image_url or image_base64 required"}, 400
        model_ref = str(body.get("model_path") or "")
        if not model_ref:
            return {"error": "model_path required"}, 400
        if bool(body.get("cam")):
            return {
                "error": "Grad-CAM is not available: ONNX Runtime has no gradients. "
                "See the evaluation report for per-class metrics and confusion data."
            }, 400
        try:
            conf = float(body.get("conf", 0.25))
            iou = float(body.get("iou", 0.45))
            if not np.isfinite(conf) or not np.isfinite(iou):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            return {"error": "invalid conf or iou"}, 400
        try:
            raw_top_k = body.get("top_k", 5)
            if isinstance(raw_top_k, bool) or not isinstance(
                raw_top_k, (str, int, np.integer)
            ):
                raise ValueError
            top_k = int(raw_top_k)
            if top_k < 1:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            return {"error": "invalid top_k"}, 400

        logger.info(
            "try/predict model=%s conf=%.2f %dx%d",
            model_ref,
            conf,
            getattr(image, "shape", [0, 0])[1],
            getattr(image, "shape", [0, 0])[0],
        )
        started = time.time()
        from onnx_predict import DecoderMappingError, DecoderMappingRequired

        try:
            session, path = _load_onnx_session(model_ref)
            from onnx_predict import read_onnx_manifest, run_onnx_trial

            manifest = read_onnx_manifest(path)
            label_map = _names(session, path, body, manifest)
            decoder = body.get("decoder") or {"mode": "auto"}
            if (
                isinstance(decoder, dict)
                and decoder.get("mode") == "custom"
                and isinstance(decoder.get("labels"), list)
            ):
                label_map = {
                    index: str(label) for index, label in enumerate(decoder["labels"])
                }
            result = run_onnx_trial(
                session,
                image,
                manifest=manifest,
                decoder=decoder,
                names=_label_list(label_map),
                conf_threshold=conf,
                iou_threshold=iou,
                top_k=max(top_k, 5),
                source=str(body.get("source") or ""),
                architecture=str(body.get("architecture") or ""),
                input_shape=body.get("input_shape"),
            )
            payload = format_trial_payload(image, result, label_map, top_k)
        except TrialRequestError as exc:
            return {"error": str(exc)}, 400
        except DecoderMappingRequired as exc:
            return {"error": str(exc), "code": exc.code}, 422
        except DecoderMappingError as exc:
            return {"error": str(exc), "code": exc.code}, 422
        except Exception as exc:  # noqa: BLE001
            logger.exception("try predict failed")
            return {"error": f"inference failed: {exc}"}, 500

        payload["time_ms"] = round((time.time() - started) * 1000, 1)
        logger.info(
            "try/predict done model=%s kind=%s boxes=%d probs=%s %.1fms",
            model_ref,
            payload.get("kind"),
            len(payload.get("boxes") or []),
            bool(payload.get("probs")),
            payload["time_ms"],
        )
        return payload
