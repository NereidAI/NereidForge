# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Catalog entrypoint for the NF Paddle trainer (`SCRIPT=catalog`).

Kept as the legacy, minimal view of the same facts `profile.py` announces (some
deployments and older tooling read `catalog`); both are generated from
`models.py`, so they cannot drift.
"""

from __future__ import annotations

import json

from models import ARCHITECTURES, DEFAULT_ARCH, LEGACY_ARCHITECTURES, architecture_entries


def catalog() -> dict:
    return {
        "framework": "paddle",
        "tasks": [
            {
                "id": "detection",
                "architectures": [a["id"] for a in architecture_entries()],
                "default": DEFAULT_ARCH,
            }
        ],
        "entrypoints": ["train", "evaluate", "export", "catalog", "profile"],
        "contract": {
            "env": ["INPUT_URI", "OUTPUT_DIR", "CONFIG_JSON", "ARTIFACT_ID"],
            "artifacts": ["<id>.onnx", "<id>.metadata.json", "<id>.metrics.json", "<id>.pdparams"],
            "progress": "progress.json",
        },
        "licenses": {arch: "Apache-2.0" for arch in ARCHITECTURES},
        "legacy": sorted(LEGACY_ARCHITECTURES),
    }


if __name__ == "__main__":
    print(json.dumps(catalog(), indent=2))
