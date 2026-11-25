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

        ocr = _ocr_engine()
        result = ocr.ocr(path, cls=True)
        regions: List[Dict[str, Any]] = []
        # result[0] = list of [box(4x2), (text, score)]
        for item in (result[0] or []) if result else []:
            try:
                box, (text, score) = item[0], item[1]
            except Exception:
                continue
            if score < self.model_score_threshold:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
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
    """Lazily initialize the shared PaddleOCR engine (det+rec, mobile)."""
    global _ocr_engine_cache
    if _ocr_engine_cache is None:
        from paddleocr import PaddleOCR

        _ocr_engine_cache = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    return _ocr_engine_cache
