# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""KeypointLabelsModel - NOT SUPPORTED (YOLO pose removed due to AGPL-3.0).

Keypoint detection with YOLO has been removed due to AGPL-3.0 license incompatibility.
Use PP-YOLOE (PaddleDetection) for object detection, and implement pose estimation
separately if needed using Apache-2.0 licensed alternatives (e.g., RTMPose, OpenPose).
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class KeypointLabelsModel:
    """NOT SUPPORTED - YOLO pose removed due to AGPL-3.0 license."""

    type = "KeyPointLabels"

    @classmethod
    def is_control_matched(cls, control) -> bool:
        return False

    def predict_regions(self, path) -> List:
        raise NotImplementedError(
            "KeypointLabelsModel (YOLO pose) removed due to AGPL-3.0 license. "
            "Use PP-YOLOE (PaddleDetection) for object detection and implement "
            "pose estimation separately with Apache-2.0 licensed alternatives "
            "(e.g., RTMPose, OpenPose)."
        )
