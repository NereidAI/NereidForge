# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""RectangleLabelsObbModel - NOT SUPPORTED (YOLO OBB removed due to AGPL-3.0).

Oriented bounding box detection with YOLO OBB has been removed due to AGPL-3.0 license.
Use PP-YOLOE (PaddleDetection) for object detection, and implement OBB detection
separately if needed using Apache-2.0 licensed alternatives.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class RectangleLabelsObbModel:
    """NOT SUPPORTED - YOLO OBB removed due to AGPL-3.0 license."""

    type = "RectangleLabels"

    @classmethod
    def is_control_matched(cls, control) -> bool:
        return False

    def predict_regions(self, path) -> List:
        raise NotImplementedError(
            "RectangleLabelsObbModel (YOLO OBB) removed due to AGPL-3.0 license. "
            "Use PP-YOLOE (PaddleDetection) for object detection and implement "
            "OBB detection separately with Apache-2.0 licensed alternatives."
        )
