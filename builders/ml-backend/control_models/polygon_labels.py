# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import logging

from control_models.base import ControlModel
from typing import List, Dict


logger = logging.getLogger(__name__)


class PolygonLabelsModel(ControlModel):
    """
    Class representing a PolygonLabels control tag, backed by an ONNX
    instance-segmentation graph (boxes + masks).
    """

    type = "PolygonLabels"
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
        return self.create_polygons(dets, W, H)

    def create_polygons(self, dets, width, height):
        """Convert per-instance masks into Label Studio polygons."""
        import cv2

        logger.debug(f"create_polygons: {self.from_name}")
        regions = []
        for det in dets:
            score = float(det["score"])
            if score < self.model_score_threshold:
                continue
            ls_label = self.ls_label(det.get("name"), det.get("cls", 0))
            if not ls_label:
                # no mapping between model label and LS label
                continue

            contours, _ = cv2.findContours(
                det["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) < 1.0:
                continue
            approx = cv2.approxPolyDP(
                contour, 0.002 * cv2.arcLength(contour, True), True
            )
            if len(approx) < 3:
                continue
            points = [
                [float(p[0][0]) / width * 100, float(p[0][1]) / height * 100]
                for p in approx
            ]

            logger.debug(
                "----------------------\n"
                f"type: {self.control}\n"
                f"polygon points > {len(points)}\n"
                f"model label > {det.get('name')}\n"
                f"score > {score}\n"
            )
            regions.append(
                {
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "polygonlabels",
                    "value": {
                        "polygonlabels": [ls_label],
                        "points": points,
                        "closed": True,
                    },
                    "score": score,
                }
            )
        return regions
