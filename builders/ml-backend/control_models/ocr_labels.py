# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""OCR control model (backend_type=ocr) for the NF ML backend.

Detects text regions with PaddleOCR (ppocr) and returns them as
RectangleLabels regions carrying a `transcription` attribute, so a single
/predict call produces both boxes and the recognized text.

Label config contract (NF generates this for task_type=ocr):
  <View>
    <Image name="image" value="$image"/>
    <RectangleLabels name="label" toName="image" model_ocr="true">
      <Label value="text"/>
    </RectangleLabels>
  </View>

`model_ocr="true"` routes RectangleLabels to this model instead of the
YOLO detector. model_path may point to a PPOCR det/rec model dir (s3:// or
MODEL_ROOT); by default the bundled PP-OCRv4 mobile models are used.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from label_studio_sdk.label_interface.control_tags import ControlTag

from control_models.base import ControlModel, get_bool

logger = logging.getLogger(__name__)


class OcrRectangleModel(ControlModel):
    """RectangleLabels variant backed by PaddleOCR instead of YOLO."""

    type = "RectangleLabels"
    model_path = "paddleocr"  # marker: not a YOLO weight file

    @classmethod
    def is_control_matched(cls, control: ControlTag) -> bool:
        if control.objects[0].tag != "Image":
            return False
        if control.tag != "RectangleLabels":
            return False
        # only claim the control when OCR mode is explicitly requested
        return get_bool(control.attr, "model_ocr", "false")

    @classmethod
    def create(cls, mlbackend, control: ControlTag):
        """Build an OCR instance (bypasses the YOLO loader entirely)."""
        instance = cls(
            control=control,
            from_name=control.from_name[0] if hasattr(control, "from_name") else "label",
            to_name=control.to_name[0],
            value=control.objects[0].value_name,
            model=None,
            model_score_threshold=0.5,
            label_map={"text": "text"},
            label_studio_ml_backend=mlbackend,
            project_id=mlbackend.project_id,
        )
        instance._resolved_model_path = "paddleocr"  # type: ignore[attr-defined]
        return instance

    def predict_regions(self, path) -> List[Dict[str, Any]]:
        """Run OCR detection+recognition; one region per text line."""
        try:
            from paddleocr import PaddleOCR  # noqa: F401  # Probe the optional OCR runtime.
        except ImportError:
            raise ValueError(
                "paddleocr not installed in this backend; deploy the "
                "ml-backend-ocr service (builders/ml-backend-ocr)"
            )

        import cv2

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"cannot read image {path}")
        h, w = img.shape[:2]

        lines = run_ocr_lines(path)
        regions: List[Dict[str, Any]] = []
        for quad, text, score in lines:
            if score < self.model_score_threshold:
                continue
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
            x2, y2 = min(float(w), max(xs)), min(float(h), max(ys))
            regions.append({
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": ["text"],
                    "x": x1 / w * 100,
                    "y": y1 / h * 100,
                    "width": (x2 - x1) / w * 100,
                    "height": (y2 - y1) / h * 100,
                    # transcription rides along as a region attribute
                    "transcription": [str(text)],
                },
                "score": float(score),
            })
        return regions


_ocr_engine_cache: Any = None


def _ocr_engine() -> Any:
    """Lazily initialize the shared PaddleOCR engine (det+rec, mobile).

    paddleocr 3.x: PaddleOCR(lang=..) with .predict(); 2.x fallback uses
    .ocr(). The engine object is shared; per-call parsing lives in
    run_ocr_lines().
    """
    global _ocr_engine_cache
    if _ocr_engine_cache is None:
        from paddleocr import PaddleOCR

        # enable_mkldnn=False works around paddle 3.x onednn PIR attribute
        # bugs on some CPU models (NotImplementedError in onednn instruction)
        try:
            _ocr_engine_cache = PaddleOCR(lang="ch", enable_mkldnn=False)  # 3.x
        except TypeError:
            _ocr_engine_cache = PaddleOCR(use_angle_cls=True, lang="ch", enable_mkldnn=False)
    return _ocr_engine_cache


def run_ocr_lines(image_path: str) -> list[tuple[list[list[float]], str, float]]:
    """OCR one image -> [(quad_points(4x[x,y]), text, score), ...].

    Handles both paddleocr 3.x (predict → JSON dicts with rec_texts) and
    2.x (ocr → [box, (text, score)]) result shapes.
    """
    ocr = _ocr_engine()
    # 3.x path
    if hasattr(ocr, "predict"):
        try:
            out = ocr.predict(image_path)
            lines: list[tuple[list[list[float]], str, float]] = []
            for page in out or []:
                texts = page.get("rec_texts") if isinstance(page, dict) else getattr(page, "key_information", None)
                boxes = page.get("rec_boxes") if isinstance(page, dict) else None
                scores = page.get("rec_scores") if isinstance(page, dict) else None
                dt_polys = page.get("rec_polys") if isinstance(page, dict) else None
                if texts is None:
                    continue
                for i, text in enumerate(texts):
                    score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
                    if dt_polys is not None and i < len(dt_polys):
                        quad = [[float(p[0]), float(p[1])] for p in dt_polys[i]]
                    elif boxes is not None and i < len(boxes):
                        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
                        quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    else:
                        quad = [[0, 0], [1, 0], [1, 1], [0, 1]]
                    lines.append((quad, str(text), score))
            if lines:
                return lines
        except Exception as e:  # noqa: BLE001
            logger.warning("paddleocr 3.x predict failed, trying 2.x path: %s", e)
    # 2.x path (paddleocr 3.x ocr() also exists but rejects cls=)
    try:
        result = ocr.ocr(image_path)
    except TypeError:
        result = ocr.ocr(image_path, cls=True)
    lines = []
    for item in (result[0] or []) if result and result[0] else []:
        try:
            box, (text, score) = item[0], item[1]
            quad = [[float(p[0]), float(p[1])] for p in box]
            lines.append((quad, str(text), float(score)))
        except Exception:
            continue
    return lines
