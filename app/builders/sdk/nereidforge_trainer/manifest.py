# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Artifact manifest (`<id>.metadata.json`) construction.

The manifest is the only thing the inference side trusts: it declares the task,
label order, input layout and normalisation, so an ONNX file exported by *any*
trainer can be served without the platform knowing the framework (ADR-014 D-04).
"""

from __future__ import annotations

import time
from typing import Any, Optional

MANIFEST_SCHEMA_VERSION = "1.0"


def build_manifest(
    *,
    task: str,
    architecture: str,
    framework: str,
    input_size: int,
    labels: list[str],
    metrics: dict[str, Any],
    dataset: dict[str, Any],
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the NF artifact manifest stored next to the ONNX export."""
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task": task,
        "framework": framework,
        "architecture": architecture,
        "labels": labels,
        "input": {
            "name": "images",
            "size": input_size,
            "layout": "NCHW",
            "color": "RGB",
        },
        "preprocess": {
            "scale": "1/255",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "outputs": manifest_outputs(task),
        "metrics": metrics,
        "dataset": dataset,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "license": {"code": "Apache-2.0", "weights": "NF-trained"},
    }
    if extra:
        manifest.update(extra)
    return manifest


def manifest_outputs(task: str) -> dict[str, Any]:
    """Default output contract per platform task (0-based labels, ADR-014 D-04)."""
    if task in ("detection", "instance_seg", "pose"):
        outputs: dict[str, Any] = {
            "boxes": {"layout": "xyxy", "space": "model-input", "dtype": "float32"},
            "scores": {"dtype": "float32"},
            "labels": {"dtype": "int64", "offset": 0},
        }
        if task in ("instance_seg", "pose"):
            outputs["masks" if task == "instance_seg" else "keypoints"] = {
                "space": "model-input",
            }
        return outputs
    if task == "semantic_seg":
        return {"logits": {"layout": "NCHW", "classes": "per-label"}}
    return {"logits": {"layout": "1xN", "classes": "per-label"}}
