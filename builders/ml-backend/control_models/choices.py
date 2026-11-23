# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import logging

from control_models.base import ControlModel
from typing import List, Dict


logger = logging.getLogger(__name__)


class ChoicesModel(ControlModel):
    """
    Class representing a Choices (classes) control tag, backed by an ONNX
    classifier (single-label or multi-label head).
    """

    type = "Choices"
    # No default artifact: nothing is bundled with the image. The project
    # selects a model version (ONNX export) in the NF UI.
    model_path = ""

    @classmethod
    def is_control_matched(cls, control) -> bool:
        # check object tag type
        if control.objects[0].tag != "Image":
            return False
        # support both Choices and Taxonomy because of their similarity
        return control.tag in [cls.type, "Taxonomy"]

    def predict_regions(self, path) -> List[Dict]:
        import cv2

        from onnx_predict import classify_onnx

        model_file = getattr(self, "_resolved_model_path", "")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"cannot read image {path}")
        preds = classify_onnx(model_file, img, topk=max(len(self.label_map or {}), 5))
        return self.create_choices(preds, path)

    def create_choices(self, preds, path):
        logger.debug(f"create_choices: {self.from_name}")
        mode = self.control.attr.get("choice", "single")

        # Resolve every prediction to (model label, LS label, prob) once, so
        # the single/multi branches only differ in how they filter.
        resolved = []
        for p in preds:
            model_label = self.model_label_name(p.get("name"), p.get("cls", 0))
            if not model_label or model_label not in (self.label_map or {}):
                continue
            resolved.append((model_label, self.label_map[model_label], float(p["prob"])))

        if mode in ["single", "single-radio"]:
            if not resolved:
                logger.debug("No choices found")
                return []
            # preds are sorted by probability; keep the best mapped class.
            _, ls_label, prob = resolved[0]
            probs, names = [prob], [ls_label]
        else:
            selected = [r for r in resolved if r[2] >= self.model_score_threshold]
            probs = [r[2] for r in selected]
            names = [r[1] for r in selected]

        if not probs:
            logger.debug("No choices found")
            return []

        score = sum(probs) / len(probs)
        logger.debug(
            "----------------------\n"
            f"task id > {path}\n"
            f"control: {self.control}\n"
            f"probs > {probs}\n"
            f"score > {score}\n"
            f"names > {names}\n"
        )

        if score < self.model_score_threshold:
            logger.debug(f"Score is too low for choice: {names[0]} = {probs[0]}")
            return []

        return [
            {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "choices",
                "value": {"choices": names},
                "score": float(score),
            }
        ]
