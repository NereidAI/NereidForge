# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Capability profile of the NF vision trainer (`nf.trainer-profile/1`).

The image announces what it can do; the platform registers the announcement and
never keeps a second copy of this list (ADR-015). `trainer-profile.json` in this
directory is the committed output of `profile()` — the file is what the backend
reads when it seeds the image catalog, so it must stay in sync (guarded by
`tests/test_profile.py`).

    python nf_profile.py            # print the profile of this image
    NF_DEVICE=cuda python nf_profile.py   # same, with the device narrowed

`devices` in the committed file describes the component family (all vendors);
the running image narrows it through the `NF_DEVICE` build env because one set
of scripts is baked into every vendor image.
"""

from __future__ import annotations

import json
import os
from typing import Any

from models import DEFAULT_ARCH, TASK_ARCHS

SCHEMA = "nf.trainer-profile/1"

# The component family can run on all four vendors; each image picks one.
DEVICES = ["cpu", "cuda", "rocm", "xpu"]

TASK_LABELS = {
    "classification": "Image classification",
    "multilabel": "Multi-label classification",
    "detection": "Object detection",
    "instance_seg": "Instance segmentation",
    "semantic_seg": "Semantic segmentation",
    "pose": "Pose / keypoints",
}

# Tier per task (ADR-015 D-03). Everything this trainer implements today is the
# free default; heavy or specialised tasks added later declare "extended" here
# instead of being deleted or hardcoded in the platform.
TASK_TIERS: dict[str, str] = {}

# Architecture overrides: id -> tier. Empty means "same tier as the task".
ARCH_TIERS: dict[str, str] = {}

# Non-task features (optimisation, export variants, ...) announced separately so
# the platform can show/lock them without inventing a task id.
CAPABILITIES = [
    {
        "id": "export.onnx",
        "label": "ONNX export",
        "tier": "core",
        "entrypoint": "export",
    },
    {
        "id": "optimization.tensorrt",
        "label": "TensorRT optimisation (INT8/FP16)",
        "tier": "extended",
        "requires": ["extended.optimization"],
        "entrypoint": "optimize",
    },
]


# Requirement range, not the exact build: the file describes the component
# family and must not change when a vendor image ships a different torch build.
FRAMEWORK = {"name": "torch", "version": ">=2.9,<3"}


def _license_for(arch_id: str) -> str:
    """Code/weights licence of an architecture (ADR-014 §2.2)."""
    return "MIT" if arch_id.startswith("smp:") else "BSD-3-Clause"


def profile(device: str = "") -> dict[str, Any]:
    """Build the profile document for one image.

    `device` narrows the announced devices to what this image can actually use;
    it defaults to `NF_DEVICE` (baked per Dockerfile) and falls back to the whole
    family so a plain `python nf_profile.py` is still a valid announcement.
    """
    runtime_device = (device or os.environ.get("NF_DEVICE", "") or "").strip().lower()
    tasks = []
    for task, architectures in TASK_ARCHS.items():
        task_tier = TASK_TIERS.get(task, "core")
        tasks.append({
            "id": task,
            "label": TASK_LABELS.get(task, task),
            "tier": task_tier,
            "default_arch": DEFAULT_ARCH[task],
            "architectures": [
                {
                    **arch,
                    "tier": ARCH_TIERS.get(str(arch.get("id")), task_tier),
                    "license": _license_for(str(arch.get("id"))),
                }
                for arch in architectures
            ],
        })
    return {
        "schema": SCHEMA,
        "name": "vision/torch",
        "framework": dict(FRAMEWORK),
        "devices": [runtime_device] if runtime_device else list(DEVICES),
        "tier": "core",
        "tasks": tasks,
        "capabilities": CAPABILITIES,
        "entrypoints": {
            "train": "python /app/train.py",
            "evaluate": "python /app/evaluate.py",
            "export": "python /app/export.py",
            "catalog": "python /app/catalog.py",
            "profile": "python /app/nf_profile.py",
        },
        "env": ["INPUT_URI", "OUTPUT_DIR", "CONFIG_JSON", "ARTIFACT_ID"],
        "artifacts": [
            "<id>.onnx",
            "<id>.metadata.json",
            "<id>.metrics.json",
            "<id>.pt",
        ],
        "config_schema": {
            "task": {"type": "string", "enum": list(TASK_ARCHS)},
            "model": {"type": "string"},
            "epochs": {"type": "integer", "default": 50},
            "batch": {"type": "integer", "default": 16},
            "workers": {"type": "integer", "minimum": 0, "default": 2},
            "image_size": {"type": "integer"},
            "learning_rate": {"type": "number"},
            "base_weights_uri": {"type": "string"},
            "freeze_backbone": {"type": "boolean", "default": False},
        },
        "license": {"code": "Apache-2.0", "weights": "per-model"},
    }


if __name__ == "__main__":
    print(json.dumps(profile(), indent=2))
