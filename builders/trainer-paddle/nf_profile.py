# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Capability profile of the NF Paddle trainer (`nf.trainer-profile/1`).

Same announcement contract as the torch trainer (ADR-015): the image states what
it can do, the platform registers it. Only the tasks this trainer actually
implements are listed — the pre-2026-10 catalog advertised four Paddle tasks
while the implementation could only run one, which is exactly the drift this
mechanism exists to prevent.

    python nf_profile.py                  # print this image's profile
    NF_DEVICE=cuda python nf_profile.py   # device narrowed to the vendor image
"""

from __future__ import annotations

import json
import os
from typing import Any

from models import DEFAULT_ARCH, LEGACY_ARCHITECTURES, architecture_entries, input_size

SCHEMA = "nf.trainer-profile/1"

DEVICES = ["cpu", "cuda"]
# Verified band: 3.0.0. 3.1+ breaks the PaddleX3.0 repo dy2static export and
# paddle2onnx 2.1.0 cannot parse its PIR output (see pit 150/151).
FRAMEWORK = {"name": "paddle", "version": ">=3.0,<3.1"}

TASK_LABELS = {"detection": "Object detection"}

# Non-task features announced separately (the platform can show/lock them
# without inventing a task id).
CAPABILITIES = [
    {"id": "export.onnx", "label": "ONNX export (paddle2onnx)", "tier": "core",
     "entrypoint": "export"},
    {"id": "export.inference", "label": "Paddle inference model export", "tier": "core",
     "entrypoint": "export"},
]


def profile(device: str = "") -> dict[str, Any]:
    runtime_device = (device or os.environ.get("NF_DEVICE", "") or "").strip().lower()
    default_size = input_size(DEFAULT_ARCH)
    return {
        "schema": SCHEMA,
        "name": "vision/paddle",
        "framework": dict(FRAMEWORK),
        "devices": [runtime_device] if runtime_device else list(DEVICES),
        "tier": "core",
        "tasks": [
            {
                "id": "detection",
                "label": TASK_LABELS["detection"],
                "tier": "core",
                "default_arch": DEFAULT_ARCH,
                "architectures": architecture_entries(),
            }
        ],
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
            "<id>.pdparams",
        ],
        "config_schema": {
            "task": {"type": "string", "enum": ["detection"]},
            "model": {"type": "string", "default": DEFAULT_ARCH},
            "epochs": {"type": "integer", "default": 50},
            "batch": {"type": "integer", "default": 8},
            "image_size": {"type": "integer", "default": default_size},
            "learning_rate": {"type": "number", "default": 0.001},
            "num_classes": {"type": "integer"},
        },
        # Ids kept only so pre-ADR-014 jobs keep resolving; the UI must not use them.
        "compat": {"legacy_architectures": sorted(LEGACY_ARCHITECTURES)},
        "license": {"code": "Apache-2.0", "weights": "per-model"},
    }


if __name__ == "__main__":
    print(json.dumps(profile(), indent=2))
