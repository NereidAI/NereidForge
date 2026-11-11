# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""DataLoader worker selection preserves explicit zero and legacy defaults."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train


@pytest.mark.parametrize(
    ("config", "expected"),
    [({}, 2), ({"workers": 0}, 0), ({"workers": 4}, 4)],
)
def test_build_loaders_distinguishes_missing_and_explicit_workers(
    monkeypatch, config, expected,
):
    """build_loaders applies the legacy default only when workers is absent."""
    observed = []
    monkeypatch.setattr(train, "_safe_workers", lambda requested: requested)
    monkeypatch.setattr(
        train,
        "build_classification",
        lambda _root, _cfg, train: ([], ["class"]),
    )

    class _Loader:
        def __init__(self, _dataset, **kwargs):
            observed.append(kwargs["num_workers"])

    monkeypatch.setattr(train, "DataLoader", _Loader)
    train.build_loaders("classification", "/unused", config)

    assert observed == [expected, expected]
