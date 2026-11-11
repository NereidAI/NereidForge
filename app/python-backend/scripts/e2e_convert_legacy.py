#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
DEPRECATED: Legacy YOLOv6/v7 ONNX conversion test removed.

This script was for testing legacy YOLOv6/v7 ONNX conversion, which used
AGPL-3.0 licensed ultralytics/YOLO code. These have been removed from
NereidForge due to AGPL-3.0 license incompatibility.

Use PP-YOLOE (PaddleDetection) for object detection instead, which is
Apache-2.0 licensed and fully supported.

See convert_step.py for the new PP-YOLOE (PaddleDetection) based conversion.
"""

import sys

def main() -> None:
    print("ERROR: e2e_convert_legacy.py is deprecated and removed.")
    print("Legacy YOLOv6/v7 ONNX conversion used AGPL-3.0 licensed ultralytics.")
    print("Use PP-YOLOE (PaddleDetection) with Apache-2.0 license instead.")
    print("See convert_step.py for PP-YOLOE (PaddleDetection) conversion.")
    sys.exit(1)

if __name__ == "__main__":
    main()
