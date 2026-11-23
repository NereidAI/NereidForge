# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Declared model-source routing contract tests.

Declared families must return before any S3 read so metadata remains the
authoritative conversion routing input.
"""

import pytest

from app.services.model_format import detect_family


def test_declared_generic_torch_source_uses_torch_family_without_s3():
    """test_declared_generic_torch_source_uses_torch_family_without_s3 covers all torch exporters."""
    for declared in ("torch", "torchvision", "custom"):
        assert detect_family(None, "bucket", "models/model.pt", declared=declared) == "torch"


def test_removed_torch_sources_fail_before_s3_access():
    """test_removed_torch_sources_fail_before_s3_access rejects unsupported exporters early."""
    for declared in ("ultralytics", "yolov5", "yolov6", "yolov7"):
        with pytest.raises(ValueError, match="unknown conversion source"):
            detect_family(None, "bucket", "models/model.pt", declared=declared)
