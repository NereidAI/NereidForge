# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Augmentation policy: presets, overrides, clamping and type validation.

The UI sends the same flat keys (`aug_fliplr`, `aug_hsv_s`, ...), the backend
groups them into `cfg["augmentation"]`, and this resolver is the single place
that turns them into knobs the datasets apply. Malformed values must never
break a training run.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import AUG_BOUNDS, AUG_PRESETS, augmentation_summary, resolve_augmentation


def test_missing_config_uses_medium_preset():
    out = resolve_augmentation(None)
    assert out["preset"] == "medium"
    assert out["fliplr"] == pytest.approx(0.5)


def test_preset_name_as_string():
    assert resolve_augmentation({"augmentation": "off"})["fliplr"] == 0.0
    assert resolve_augmentation({"augmentation": "strong"})["scale"] == pytest.approx(0.7)


def test_unknown_preset_falls_back_to_medium():
    assert resolve_augmentation({"augmentation": {"preset": "nonsense"}})["preset"] == "medium"


def test_overrides_replace_preset_values():
    out = resolve_augmentation(
        {"augmentation": {"preset": "medium", "fliplr": 0.2, "hsv_s": 0.1}}
    )
    assert out["fliplr"] == pytest.approx(0.2)
    assert out["hsv_s"] == pytest.approx(0.1)
    # Untouched keys keep the preset value.
    assert out["scale"] == pytest.approx(AUG_PRESETS["medium"]["scale"])


def test_values_are_clamped_to_bounds():
    out = resolve_augmentation(
        {"augmentation": {"preset": "medium", "fliplr": 9.0, "degrees": -5}}
    )
    assert out["fliplr"] == AUG_BOUNDS["fliplr"][1]
    assert out["degrees"] == AUG_BOUNDS["degrees"][0]


def test_malformed_value_keeps_preset_and_does_not_raise():
    out = resolve_augmentation({"augmentation": {"preset": "medium", "hsv_v": "banana"}})
    assert out["hsv_v"] == pytest.approx(AUG_PRESETS["medium"]["hsv_v"])


def test_unknown_keys_are_ignored():
    out = resolve_augmentation({"augmentation": {"preset": "medium", "inject": 1}})
    assert "inject" not in out


def test_every_preset_is_complete():
    for name, values in AUG_PRESETS.items():
        assert set(values) == set(AUG_BOUNDS), name


def test_summary_mentions_preset_and_skips_zero_knobs():
    summary = augmentation_summary(resolve_augmentation({"augmentation": "off"}))
    assert "preset=off" in summary
    assert "fliplr" not in summary
