# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Dict
from uuid import uuid4

from label_studio_sdk.converter.brush import mask2rle

from control_models.base import ControlModel

logger = logging.getLogger(__name__)


class BrushLabelsModel(ControlModel):
    """
    Class representing a BrushLabels control tag, backed by an ONNX
    instance-segmentation graph (boxes + masks).
    """

    type = "BrushLabels"
    # No default artifact: select a model version in the NF UI.
    model_path = ""

    @classmethod
    def is_control_matched(cls, control) -> bool:
        # check object tag type
        if control.objects[0].tag != "Image":
            return False
        return control.tag == cls.type

    def predict_regions(self, path) -> List[Dict]:
        import cv2

        from onnx_predict import segment_onnx

        model_file = getattr(self, "_resolved_model_path", "")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"cannot read image {path}")
        H, W = img.shape[:2]
        dets = segment_onnx(
            model_file, img, conf_threshold=self.model_score_threshold
        )
        return self.create_brush(dets, W, H)

    def create_brush(self, dets, width, height):
        logger.debug(f"create_brush: {self.from_name}")
        regions = []
        for det in dets:
            score = float(det["score"])
            if score < self.model_score_threshold:
                continue
            ls_label = self.ls_label(det.get("name"), det.get("cls", 0))
            if not ls_label:
                continue

            # segment_onnx already returns a full-resolution 0/255 mask.
            mask = det["mask"]
            if mask.shape[:2] != (height, width):
                import cv2

                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            rle = mask2rle(mask)

            logger.debug(
                "----------------------\n"
                f"type: {self.control}\n"
                f"rle length > {len(rle)}\n"
                f"model label > {det.get('name')}\n"
                f"score > {score}\n"
            )
            regions.append(
                {
                    "id": str(uuid4())[:9],
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "format": "rle",
                        "rle": rle,
                        "brushlabels": [ls_label],
                    },
                    "score": score,
                    "type": "brushlabels",
                }
            )
        return regions
