# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Augmentation policy shared by the trainer and the UI presets.

The UI sends a flat `aug_<key>` set which the backend groups into
`cfg["augmentation"]`. Keys mirror the presets in
`app/frontend/src/pages/Training.tsx`; the trainer only applies what it can
transform consistently across image + boxes + masks.
"""

from __future__ import annotations

from typing import Any

AUG_PRESETS: dict[str, dict[str, float]] = {
    "off": {"fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "shear": 0.0,
            "translate": 0.0, "scale": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0},
    "light": {"fliplr": 0.5, "flipud": 0.0, "degrees": 0.0, "shear": 0.0,
              "translate": 0.05, "scale": 0.3, "hsv_h": 0.01, "hsv_s": 0.4, "hsv_v": 0.3},
    "medium": {"fliplr": 0.5, "flipud": 0.0, "degrees": 0.0, "shear": 0.0,
               "translate": 0.1, "scale": 0.5, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4},
    "strong": {"fliplr": 0.5, "flipud": 0.0, "degrees": 5.0, "shear": 2.0,
               "translate": 0.15, "scale": 0.7, "hsv_h": 0.02, "hsv_s": 0.8, "hsv_v": 0.5},
}

# (min, max) per knob — same bounds the UI widgets enforce.
AUG_BOUNDS: dict[str, tuple[float, float]] = {
    "fliplr": (0.0, 1.0), "flipud": (0.0, 1.0), "degrees": (0.0, 15.0),
    "shear": (0.0, 10.0), "translate": (0.0, 0.3), "scale": (0.0, 0.9),
    "hsv_h": (0.0, 0.1), "hsv_s": (0.0, 1.0), "hsv_v": (0.0, 1.0),
}

# Keys the UI can send but the reference trainer does not implement yet: they
# need multi-image collation (mosaic/mixup) or mask-level edits (copy-paste).
AUG_UNSUPPORTED = ("mosaic", "mixup", "copy_paste", "close_mosaic", "perspective")


def resolve_augmentation(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve `cfg["augmentation"]` into clamped, typed augmentation knobs.

    Accepts a preset name (`"medium"`), a dict with a `preset` key plus
    overrides, or the flat `{"preset": ..., "fliplr": ...}` shape the backend
    produces. Missing/invalid values fall back to the preset; unknown keys are
    dropped; every number is clamped to the UI bounds. Never raises.
    """
    raw = cfg.get("augmentation") if isinstance(cfg, dict) else None
    preset = "medium"
    overrides: dict[str, Any] = {}
    if isinstance(raw, str):
        preset = raw
    elif isinstance(raw, dict):
        preset = str(raw.get("preset") or raw.get("level") or "medium")
        overrides = raw
    preset = preset.strip().lower()
    if preset not in AUG_PRESETS:
        print(f"[train] unknown augmentation preset {preset!r}; using 'medium'", flush=True)
        preset = "medium"

    resolved: dict[str, Any] = {"preset": preset, **AUG_PRESETS[preset]}
    for key, (low, high) in AUG_BOUNDS.items():
        if key not in overrides:
            continue
        try:
            value = float(overrides[key])
        except (TypeError, ValueError):
            continue  # malformed override: keep the preset value
        resolved[key] = min(max(value, low), high)
    return resolved


def augmentation_summary(aug: dict[str, Any]) -> str:
    """One-line, log-friendly description of the effective augmentation."""
    parts = [f"{k}={v:g}" for k, v in aug.items() if k != "preset" and float(v or 0) > 0]
    return f"preset={aug.get('preset', 'medium')} " + (" ".join(parts) or "geometric off")
