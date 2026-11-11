# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""VideoRectangleModel - NOT SUPPORTED (YOLO/ultralytics removed due to AGPL-3.0).

Video tracking with YOLO has been removed due to AGPL-3.0 license incompatibility.
Use PP-YOLOE (PaddleDetection) for object detection, and implement video tracking
separately if needed using Apache-2.0 licensed alternatives.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class VideoRectangleModel:
    """NOT SUPPORTED - YOLO/ultralytics removed due to AGPL-3.0 license."""

    type = "VideoRectangle"

    @classmethod
    def is_control_matched(cls, control) -> bool:
        return False

    def predict_regions(self, path) -> List:
        raise NotImplementedError(
            "VideoRectangleModel (YOLO tracking) removed due to AGPL-3.0 license. "
            "Use PP-YOLOE (PaddleDetection) for object detection and implement "
            "video tracking separately with Apache-2.0 licensed alternatives."
        )
