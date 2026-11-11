# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Promptable segmentation endpoint for the NF ML backend (ONNX).

Mounted as an extra route (the LS ML backend app is FastAPI under the hood).
Consumed by the NF backend proxy, NOT by Label Studio:

    POST /sam/segment
    {"image_url": "http://s3.../img.jpg" | "image_base64": "...",
     "points": [[x, y], ...],        # positive clicks, absolute pixels
     "negative_points": [[x, y], ...],
     "box": [x1, y1, x2, y2],        # optional prompt box
     "model_path": "s3://bucket/models/sam/encoder.onnx",
     "decoder_path": "s3://bucket/models/sam/decoder.onnx"}  # optional

    -> {"polygon": [[x, y], ...], "bbox": [x1, y1, x2, y2]}

Runs ONNX only (no trainer framework). Two graph shapes are accepted:

1. Split encoder/decoder exports (the common SAM/SAM2 ONNX layout):
   encoder takes a 1024x1024 image and emits image embeddings; the decoder
   takes embeddings + point/box prompts. `decoder_path` may be omitted when
   the convention `<encoder-stem-without-.encoder>.decoder.onnx` resolves.
2. A single end-to-end graph that takes the image plus prompt inputs.

No weights are bundled in the image: the caller passes the s3:// reference of
the user-uploaded, user-selected model artifact.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

SAM_INPUT_SIZE = 1024
# SAM preprocessing constants (RGB, 0..1 scaled by 255 then standardised).
_PIXEL_MEAN = (123.675, 116.28, 103.53)
_PIXEL_STD = (58.395, 57.12, 57.375)


def _resolve_model_file(name: str) -> str:
    """Resolve an s3:// / MODEL_ROOT / absolute ONNX reference to a file."""
    from control_models.base import MODEL_ROOT, ControlModel

    if name.startswith("s3://"):
        base = os.path.basename(name.split("?", 1)[0]) or "model.onnx"
        local = os.path.join(MODEL_ROOT, f"sam-{base}")
        if os.path.exists(local) or ControlModel._download_s3_model(name, local):
            ControlModel._download_sidecar_manifest(name, local)
            return local
        raise FileNotFoundError(
            f"SAM model could not be downloaded from S3: {name}. Upload an ONNX "
            "model in the NF UI and select it for interactive segmentation."
        )
    path = name if os.path.isabs(name) else os.path.join(MODEL_ROOT, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SAM model not found: {path}. No model is bundled with the image; "
            "upload one in the NF UI and select it."
        )
    return path


def _decoder_for(encoder: str, explicit: str) -> str:
    """Resolve the decoder graph for a split export."""
    if explicit:
        return _resolve_model_file(explicit)
    stem, ext = os.path.splitext(encoder)
    for cand in (stem.replace(".encoder", "") + ".decoder" + ext, stem + ".decoder" + ext):
        if os.path.exists(cand):
            return cand
    return ""


def _preprocess(img, size: int = SAM_INPUT_SIZE):
    """SAM-style resize-longest-side + pad + RGB standardise, NCHW float32."""
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32)
    rgb = (rgb - np.asarray(_PIXEL_MEAN, np.float32)) / np.asarray(_PIXEL_STD, np.float32)
    return rgb.transpose(2, 0, 1)[None], scale, (pad_x, pad_y)


def _prompts(points, neg_points, box, scale, pad, size=SAM_INPUT_SIZE):
    """Convert absolute-pixel prompts to SAM's padded input coordinates."""
    import numpy as np

    coords, labels = [], []
    for x, y in points:
        coords.append([(float(x) * scale + pad[0]), (float(y) * scale + pad[1])])
        labels.append(1.0)
    for x, y in neg_points:
        coords.append([(float(x) * scale + pad[0]), (float(y) * scale + pad[1])])
        labels.append(0.0)
    if box:
        # Box prompt = two corner points with the "box" label convention:
        # the decoder treats labels 2/3 as top-left/bottom-right corners.
        x1, y1, x2, y2 = [float(v) for v in box]
        coords.append([x1 * scale + pad[0], y1 * scale + pad[1]])
        labels.append(2.0)
        coords.append([x2 * scale + pad[0], y2 * scale + pad[1]])
        labels.append(3.0)
    if not coords:
        return None, None
    return (
        np.asarray(coords, dtype=np.float32)[None],
        np.asarray(labels, dtype=np.float32)[None],
    )


def _run_onnx(encoder_path: str, decoder_path: str, img, points, neg_points, box):
    """Return a HxW uint8 mask (original geometry) or None."""
    import cv2
    import numpy as np

    from onnx_predict import get_session

    blob, scale, pad = _preprocess(img)
    enc_sess = get_session(encoder_path)
    enc_inputs = enc_sess.get_inputs()

    # End-to-end graph: image + prompts in one artifact.
    if decoder_path and not any(
        "embedding" in (i.name or "") for i in enc_sess.get_outputs()
    ):
        raise ValueError(
            "first graph does not look like a SAM image encoder (no embedding output)"
        )
    if not decoder_path:
        # A single graph: either an end-to-end promptable model, or a bare
        # encoder (which cannot segment without a prompt decoder).
        names = {i.name for i in enc_inputs}
        if not ({"point_coords", "point_coords_2"} & names):
            raise ValueError(
                "model has no decoder: pass decoder_path (or name the encoder "
                "'*.encoder.onnx' next to '*.decoder.onnx')"
            )
        sess = enc_sess
        coords, labels = _prompts(points, neg_points, box, scale, pad)
        if coords is None:
            return None
        feeds: dict[str, Any] = {}
        for i in enc_inputs:
            if i.name in ("image", "input_image", "images"):
                feeds[i.name] = blob
            elif i.name.startswith("point_coords"):
                feeds[i.name] = coords
            elif i.name.startswith("point_labels"):
                feeds[i.name] = labels
            elif i.name in ("orig_im_size", "original_size"):
                feeds[i.name] = np.asarray(img.shape[:2], dtype=np.float32)
            elif "mask" in i.name.lower() and "input" in i.name.lower():
                feeds[i.name] = np.zeros((1, 1, 256, 256), dtype=np.float32)
        outs = sess.run(None, feeds)
        masks = np.asarray(outs[0])
    else:
        emb = enc_sess.run(None, {enc_inputs[0].name: blob})[0]
        coords, labels = _prompts(points, neg_points, box, scale, pad)
        if coords is None:
            return None
        dec_sess = get_session(decoder_path)
        h, w = img.shape[:2]
        feeds = {}
        for i in dec_sess.get_inputs():
            n = i.name
            if "embedding" in n:
                feeds[n] = emb
            elif "point_coords" in n:
                feeds[n] = coords
            elif "point_labels" in n:
                feeds[n] = labels
            elif "orig" in n and "size" in n:
                feeds[n] = np.asarray([h, w], dtype=np.float32)
            elif "mask_input" in n:
                feeds[n] = np.zeros((1, 1, 256, 256), dtype=np.float32)
            elif "has_mask" in n:
                feeds[n] = np.zeros((1,), dtype=np.float32)
        outs = dec_sess.run(None, feeds)
        masks = np.asarray(outs[0])

    if masks.ndim == 4:
        masks = masks[0, 0]
    elif masks.ndim == 3:
        masks = masks[0]
    h, w = img.shape[:2]
    if masks.shape[:2] != (h, w):
        if masks.shape[:2] == (SAM_INPUT_SIZE, SAM_INPUT_SIZE):
            nh, nw = int(round(h * scale)), int(round(w * scale))
            masks = masks[pad[1]:pad[1] + nh, pad[0]:pad[0] + nw]
        masks = cv2.resize(masks.astype(np.float32), (w, h))
    return (masks > 0).astype(np.uint8) * 255


def register_sam_routes(app) -> None:  # type: ignore[no-untyped-def]
    """Attach /sam/segment to the ML backend Flask app."""

    from flask import request

    @app.post("/sam/segment")
    def sam_segment() -> Any:  # noqa: ANN202
        try:
            body = request.get_json(force=True, silent=True) or {}
        except Exception:
            return {"error": "invalid JSON"}, 400

        points = body.get("points") or []
        neg = body.get("negative_points") or []
        box = body.get("box")
        if not points and not neg and not box:
            return {"error": "points, negative_points or box required"}, 400
        model_name = str(body.get("model_path") or os.getenv("SAM_MODEL_PATH", ""))
        if not model_name:
            return {
                "error": "no SAM model selected: upload a SAM ONNX model in the NF "
                "UI and select it for interactive segmentation"
            }, 400
        if not model_name.lower().endswith(".onnx"):
            return {
                "error": "interactive segmentation requires an ONNX SAM artifact"
            }, 400

        import time

        logger.info(
            "sam/segment model=%s pos=%d neg=%d box=%s",
            model_name, len(points), len(neg), bool(box),
        )
        t0 = time.time()

        import base64
        import tempfile

        import cv2
        import numpy as np

        img_url = str(body.get("image_url") or "")
        b64 = str(body.get("image_base64") or "")
        tmp = None
        try:
            if b64:
                raw = base64.b64decode(b64)
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            else:
                if not img_url:
                    return {"error": "image_url or image_base64 required"}, 400
                import urllib.request

                tmp = tempfile.mktemp(suffix=".jpg")
                req = urllib.request.Request(img_url, headers={"User-Agent": "nf-ml-backend"})
                with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
                    f.write(r.read())
                img = cv2.imread(tmp)
            if img is None:
                return {"error": "cannot decode image"}, 400
        except Exception as e:  # noqa: BLE001
            return {"error": f"cannot fetch/decode image: {e}"}, 400
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

        try:
            encoder = _resolve_model_file(model_name)
            decoder = _decoder_for(encoder, str(body.get("decoder_path") or ""))
            mask = _run_onnx(encoder, decoder, img, points, neg, box)
        except Exception as e:  # noqa: BLE001
            logger.exception("sam predict failed model=%s", model_name)
            return {"error": f"sam predict failed: {e}"}, 500

        if mask is None:
            return {"polygon": None, "bbox": None, "message": "no prompt produced"}

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.info("sam/segment empty mask model=%s", model_name)
            return {"polygon": None, "bbox": None, "message": "empty mask"}
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, 0.005 * cv2.arcLength(contour, True), True)
        polygon = [[int(p[0][0]), int(p[0][1])] for p in approx]
        x, y, w, h = cv2.boundingRect(contour)
        logger.info(
            "sam/segment done model=%s points=%d %.0fms",
            model_name, len(polygon), (time.time() - t0) * 1000,
        )
        return {"polygon": polygon, "bbox": [int(x), int(y), int(x + w), int(y + h)]}
