# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""PaddleX model catalog for the NF Paddle trainer (Apache-2.0).

PaddleX names models by family and scale (`PP-YOLOE_plus-S`, `PicoDet-L`), and
the scale is part of the name, so NF exposes one architecture entry per scale
instead of a family with a size knob. `num_classes` is a *config* field: the
COCO head is built from the dataset, which is why no model needs rewriting.
"""

from __future__ import annotations

from typing import Any

# NF architecture id -> (PaddleX model name, human label, default input size)
ARCHITECTURES: dict[str, tuple[str, str, int]] = {
    "ppyoloe_plus_s": ("PP-YOLOE_plus-S", "PP-YOLOE+ Small", 640),
    "ppyoloe_plus_m": ("PP-YOLOE_plus-M", "PP-YOLOE+ Medium", 640),
    "ppyoloe_plus_l": ("PP-YOLOE_plus-L", "PP-YOLOE+ Large", 640),
    "ppyoloe_plus_x": ("PP-YOLOE_plus-X", "PP-YOLOE+ X-Large", 640),
    "picodet_s": ("PicoDet-S", "PicoDet Small (edge)", 320),
    "picodet_m": ("PicoDet-M", "PicoDet Medium (edge)", 416),
    "picodet_l": ("PicoDet-L", "PicoDet Large", 640),
    "rtdetr_l": ("RT-DETR-L", "RT-DETR Large", 640),
    "rtdetr_x": ("RT-DETR-X", "RT-DETR X-Large", 640),
}

DEFAULT_ARCH = "ppyoloe_plus_s"

#: Architecture ids the UI may offer, in display order.
MODEL_CHOICES = list(ARCHITECTURES)

#: Ids that still refer to the pre-ADR-014 naming (kept resolving so old jobs
#: and old presets keep working); never offered to the UI.
LEGACY_ARCHITECTURES: dict[str, str] = {
    "yolov8n": "ppyoloe_plus_s",
    "yolov8s": "ppyoloe_plus_s",
    "yolov8m": "ppyoloe_plus_m",
    "yolov8l": "ppyoloe_plus_l",
    "yolov8x": "ppyoloe_plus_x",
    "ppyoloe_s": "ppyoloe_plus_s",
    "ppyoloe_m": "ppyoloe_plus_m",
    "ppyoloe_l": "ppyoloe_plus_l",
    "ppyoloe_x": "ppyoloe_plus_x",
    "pp-yoloe-l": "ppyoloe_plus_l",
    "pp-yoloe-s": "ppyoloe_plus_s",
    "ppyoloe-l": "ppyoloe_plus_l",
    "ppyoloe-s": "ppyoloe_plus_s",
}


def canonical_arch(arch: str) -> str:
    """Map a legacy/alias id onto a supported architecture id."""
    value = (arch or "").strip().lower()
    if value in ARCHITECTURES:
        return value
    return LEGACY_ARCHITECTURES.get(value, DEFAULT_ARCH)


def architecture_entries() -> list[dict[str, Any]]:
    """Architecture list for the profile/catalog (announced by the component)."""
    return [
        {
            "id": arch_id,
            "label": label,
            "size": size,
            "tier": "core",
            "license": "Apache-2.0",
        }
        for arch_id, (_name, label, size) in ARCHITECTURES.items()
    ]


def model_name(arch: str) -> str:
    """PaddleX model name for an NF architecture id (e.g. `PP-YOLOE_plus-S`)."""
    return ARCHITECTURES[canonical_arch(arch)][0]


def resolve_model_name(arch: str, num_classes: int = 0) -> str:
    """PaddleX model name for a job.

    `num_classes` is accepted for API symmetry with the torch trainer (and for
    future catalogue-style variants); PaddleX derives the head from the dataset,
    so the name does not depend on it. A model name passed straight through
    (e.g. `PP-YOLOE_plus-M`) is honoured as-is.
    """
    value = (arch or "").strip()
    if value and value not in ARCHITECTURES and value not in LEGACY_ARCHITECTURES:
        if value[0].isupper():  # a PaddleX name, not an NF id
            return value
    return model_name(arch)


def input_size(arch: str) -> int:
    return ARCHITECTURES[canonical_arch(arch)][2]
