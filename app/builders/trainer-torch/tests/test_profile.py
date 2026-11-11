# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""The committed profile must match what the component announces (ADR-015).

`trainer-profile.json` is what the platform reads when it seeds the image
catalog, so a stale file would make the UI offer capabilities the image does not
have (or hide ones it does). This test fails on drift.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nf_profile import SCHEMA, profile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIERS = {"core", "extended"}


def _committed():
    with open(os.path.join(HERE, "trainer-profile.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_committed_file_matches_generator():
    committed = _committed()
    generated = profile()
    # `devices` narrows per image (NF_DEVICE), so compare everything else.
    assert committed.pop("devices")
    generated.pop("devices")
    assert committed == generated


def test_schema_and_identity():
    data = _committed()
    assert data["schema"] == SCHEMA
    assert data["name"] == "vision/torch"
    assert data["tier"] in TIERS


def test_every_task_is_complete():
    for task in _committed()["tasks"]:
        assert task["tier"] in TIERS, task["id"]
        ids = [a["id"] for a in task["architectures"]]
        assert ids, task["id"]
        assert task["default_arch"] in ids, task["id"]
        assert len(ids) == len(set(ids)), task["id"]


def test_extended_capabilities_declare_their_entitlement():
    for cap in _committed().get("capabilities") or []:
        assert cap["tier"] in TIERS, cap["id"]
        if cap["tier"] == "extended":
            assert cap.get("requires"), cap["id"]


def test_entrypoints_and_contract_are_announced():
    data = _committed()
    for name in ("train", "evaluate", "export", "catalog", "profile"):
        assert data["entrypoints"][name].startswith("python "), name
    assert "CONFIG_JSON" in data["env"]
    assert any(a.endswith(".onnx") for a in data["artifacts"])
    assert data["license"]["code"] == "Apache-2.0"
