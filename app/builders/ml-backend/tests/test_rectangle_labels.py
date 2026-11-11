# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for PP-YOLOE ONNX rectangle prediction."""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onnx_predict  # noqa: E402
from control_models.rectangle_labels import RectangleLabelsModel  # noqa: E402


def test_onnx_regions_load_image_in_using_method(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ONNX method must own every import used by its inference path."""
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "imread", lambda _path: image)
    monkeypatch.setattr(
        onnx_predict,
        "detect_onnx",
        lambda *_args, **_kwargs: [
            {"name": "object", "cls": 0, "xyxy": [0, 0, 20, 10], "score": 0.9}
        ],
    )
    monkeypatch.setattr(onnx_predict, "read_onnx_names", lambda _path: ["object"])
    model = RectangleLabelsModel.model_construct(
        from_name="label",
        to_name="image",
        model_score_threshold=0.5,
        label_map={"object": "object"},
    )

    regions = model._predict_regions_onnx("image.jpg", "model.onnx")

    assert regions == [
        {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "rectanglelabels": ["object"],
                "x": 0.0,
                "y": 0.0,
                "width": 50.0,
                "height": 50.0,
            },
            "score": 0.9,
        }
    ]
