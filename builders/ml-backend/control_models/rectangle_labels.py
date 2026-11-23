# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Dict

from control_models.base import ControlModel


logger = logging.getLogger(__name__)


class RectangleLabelsModel(ControlModel):
    """
    Class representing a RectangleLabels (bounding boxes) control tag for PP-YOLOE model.
    """

    type = "RectangleLabels"
    # MODEL_PATH env (s3:// supported) can override this in base.load_model
    model_path = ""

    @classmethod
    def is_control_matched(cls, control) -> bool:
        # check object tag type
        if control.objects[0].tag != "Image":
            return False
        return control.tag == cls.type

    def predict_regions(self, path) -> List[Dict]:
        """Run PP-YOLOE inference and return Label Studio regions."""
        # PP-YOLOE uses ONNX export
        mp = getattr(self, "_resolved_model_path", "")
        if isinstance(mp, str) and mp.lower().endswith(".onnx"):
            return self._predict_regions_onnx(path, mp)

        # Fallback for future native PaddleDetection support
        raise NotImplementedError("Only ONNX model path is supported for PP-YOLOE inference")

    def _predict_regions_onnx(self, path: str, model_file: str) -> List[Dict]:
        """Detection through onnxruntime for exported PP-YOLOE ONNX models."""
        import cv2

        from onnx_predict import detect_onnx, read_onnx_names

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"cannot read image {path}")
        dets = detect_onnx(model_file, img, conf_threshold=self.model_score_threshold)
        names = read_onnx_names(model_file)
        H, W = img.shape[:2]
        cfg_labels = [v for v in self.label_map.values()]
        regions: List[Dict] = []
        for d in dets:
            name = d["name"]
            if name and name in self.label_map:
                out_label = self.label_map[name]
            elif names is None and d["cls"] < len(cfg_labels):
                # no embedded names: class index -> config label order
                out_label = cfg_labels[d["cls"]]
            else:
                continue
            x1, y1, x2, y2 = d["xyxy"]
            regions.append({
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [out_label],
                    "x": max(0.0, x1) / W * 100,
                    "y": max(0.0, y1) / H * 100,
                    "width": (x2 - x1) / W * 100,
                    "height": (y2 - y1) / H * 100,
                },
                "score": d["score"],
            })
        return regions

    def create_rectangles(self, results, path):
        """Simple bounding boxes without rotation (for future native PaddleDetection support)."""
        logger.debug(f"create_rectangles: {self.from_name}")
        # This will be implemented when native PaddleDetection support is added
        raise NotImplementedError("Native PP-YOLOE inference not yet implemented; use ONNX export")
