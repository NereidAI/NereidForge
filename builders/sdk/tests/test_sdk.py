# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the trainer SDK (ADR-016).

These pin the pieces a third-party trainer relies on: env parsing, manifest
shape, augmentation resolution, profile validation and the scaffold.
"""

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SDK_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.abspath(os.path.join(SDK_ROOT, "..", ".."))
sys.path.insert(0, SDK_ROOT)

import nereidforge_trainer as nf  # noqa: E402
from nereidforge_trainer import cli, runtime  # noqa: E402


def _torch_profile_path():
    return os.path.join(REPO_ROOT, "builders", "trainer-torch", "trainer-profile.json")


def test_version_is_the_pep440_dev_channel_default():
    # Tag-sourced version model (root treaty §19.5): no VERSION file exists.
    # Off-tag checkouts ship the rolling dev channel; the Python package uses
    # the PEP 440 spelling that the `0.0.0-dev` image tag normalizes to.
    assert nf.__version__ == "0.0.0.dev0"
    assert not nf.__version__.startswith("v")
    assert not os.path.exists(os.path.join(REPO_ROOT, "VERSION"))


def test_manifest_is_manifest_schema_10_and_labels_are_zero_based():
    manifest = nf.build_manifest(
        task="detection",
        architecture="fasterrcnn",
        framework="torch",
        input_size=640,
        labels=["cat", "dog"],
        metrics={"mAP": 0.5},
        dataset={"uri": "s3://b/d/"},
    )
    assert manifest["schema_version"] == nf.MANIFEST_SCHEMA_VERSION == "1.0"
    assert manifest["labels"] == ["cat", "dog"]
    assert manifest["outputs"]["labels"]["offset"] == 0
    assert manifest["input"]["size"] == 640
    assert manifest["license"]["code"] == "Apache-2.0"


def test_manifest_outputs_per_task():
    assert "boxes" in nf.build_manifest(
        task="pose", architecture="k", framework="torch", input_size=64,
        labels=[], metrics={}, dataset={},
    )["outputs"]
    assert nf.build_manifest(
        task="semantic_seg", architecture="s", framework="torch", input_size=64,
        labels=["a"], metrics={}, dataset={},
    )["outputs"]["logits"]["classes"] == "per-label"


def test_training_metrics_require_versioned_positive_duration():
    source = {"task": "detection", "mAP50": 0.8, "train_time_s": 1}
    document = nf.build_training_metrics(source, train_time_s=3465.0)

    assert document == {
        "schema": "nf.training-metrics/1",
        "task": "detection",
        "mAP50": 0.8,
        "train_time_s": 3465.0,
    }
    assert source["train_time_s"] == 1
    assert nf.validate_training_metrics(document) == []


def test_training_metrics_validation_rejects_missing_or_invalid_duration():
    assert "schema" in nf.validate_training_metrics({})[0]
    for value in (None, "12", True, 0, -1, float("nan"), float("inf")):
        document = {"schema": nf.TRAINING_METRICS_SCHEMA, "train_time_s": value}
        assert any("train_time_s" in error for error in nf.validate_training_metrics(document))
        with pytest.raises(ValueError, match="train_time_s"):
            nf.build_training_metrics({}, train_time_s=value)


def test_progress_json_replaces_non_finite_numbers_with_null(tmp_path, monkeypatch):
    """Live progress remains strict JSON even when a framework reports NaN."""
    destination = tmp_path / "progress.json"
    original_write_json = runtime.write_json
    monkeypatch.setattr(
        runtime, "write_json",
        lambda _path, data: original_write_json(str(destination), data),
    )
    progress = runtime.Progress("", 100)

    progress.log(10, {"train_loss": 3.8, "val_loss": float("nan")})

    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["val_loss"] is None
    assert document["history"][0]["val_loss"] is None
    assert document["train_loss"] == 3.8
    json.dumps(document, allow_nan=False)


def test_read_config_accepts_env_json_and_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CONFIG_JSON", '{"epochs": 3}')
    assert nf.read_config() == {"epochs": 3}
    path = tmp_path / "cfg.json"
    path.write_text('{"task": "detection"}', encoding="utf-8")
    assert nf.read_config(str(path)) == {"task": "detection"}
    monkeypatch.setenv("CONFIG_JSON", "")
    assert nf.read_config() == {}


def test_augmentation_falls_back_and_clamps():
    resolved = nf.resolve_augmentation({"augmentation": "nope"})
    assert resolved["preset"] == "medium"
    clamped = nf.resolve_augmentation({"augmentation": {"preset": "light", "scale": 99}})
    assert clamped["scale"] == 0.9
    assert nf.resolve_augmentation(None)["preset"] == "medium"


def test_valid_torch_profile_passes_validation():
    with open(_torch_profile_path(), encoding="utf-8") as fh:
        doc = json.load(fh)
    assert nf.validate_profile(doc) == []


def test_profile_validation_reports_contract_breaks():
    doc = {
        "schema": "nf.trainer-profile/1",
        "name": "x/y",
        "framework": {"name": "x"},
        "devices": ["gpu"],
        "tasks": [
            {
                "id": "detection",
                "label": "Detection",
                "default_arch": "missing",
                "architectures": [{"id": "a", "tier": "extended"}],
            }
        ],
        "entrypoints": {},
        "license": {},
    }
    errors = nf.validate_profile(doc)
    assert any("unknown device" in e for e in errors)
    assert any("default_arch" in e for e in errors)
    assert any("requires" in e for e in errors)
    assert any("license" in e for e in errors)
    assert any("entrypoints.train" in e for e in errors)


def test_all_builtin_training_entrypoints_emit_the_metrics_contract():
    entrypoints = (
        "trainer-torch/train.py",
        "trainer-paddle/train.py",
        "trainer-llm-cu/llm_train.py",
        "trainer-llm-rocm/llm_train.py",
        "trainer-llm-xpu/llm_train.py",
        "trainer-llm-xpu/vlm_train.py",
    )
    for relative in entrypoints:
        path = os.path.join(REPO_ROOT, "builders", relative)
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        assert "build_training_metrics(" in source, relative
        assert "time.monotonic()" in source, relative


def test_llm_training_images_package_the_shared_sdk():
    for vendor in ("cu", "rocm", "xpu"):
        path = os.path.join(REPO_ROOT, "builders", f"trainer-llm-{vendor}", "Dockerfile")
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        assert "COPY sdk /opt/nf-trainer-sdk" in source, vendor
        assert "pip install --no-cache-dir" in source, vendor
        assert "/opt/nf-trainer-sdk" in source, vendor
        assert "rm -rf /opt/nf-trainer-sdk" in source, vendor


def test_effective_tier_is_most_specific():
    doc = {
        "schema": nf.PROFILE_SCHEMA,
        "name": "x/y",
        "framework": {"name": "x"},
        "devices": ["cpu"],
        "tier": "core",
        "tasks": [
            {
                "id": "detection",
                "label": "Detection",
                "tier": "extended",
                "requires": ["extended.detection"],
                "default_arch": "a",
                "architectures": [
                    {"id": "a", "tier": "core", "license": "MIT"},
                    {"id": "b", "tier": "extended", "requires": ["extended.heavy"],
                     "license": "MIT"},
                ],
            }
        ],
        "capabilities": [
            {"id": "cap.core", "tier": "core"},
            {"id": "cap.ext", "tier": "extended", "requires": ["extended.cap"]},
        ],
        "entrypoints": {"train": "python /app/train.py"},
        "license": {"code": "Apache-2.0"},
    }
    assert nf.validate_profile(doc) == []
    assert nf.effective_tier(doc, task="detection") == "extended"
    assert nf.effective_tier(doc, task="detection", arch="a") == "core"
    assert nf.effective_tier(doc, task="detection", arch="b") == "extended"
    assert nf.effective_tier(doc, capability="cap.ext") == "extended"
    assert nf.effective_tier(doc, capability="cap.core") == "core"


def test_missing_entitlements_uses_most_specific_requirement():
    doc = {
        "schema": nf.PROFILE_SCHEMA,
        "name": "x/y",
        "framework": {"name": "x"},
        "devices": ["cpu"],
        "tasks": [
            {
                "id": "detection",
                "label": "Detection",
                "tier": "extended",
                "requires": ["extended.detection"],
                "default_arch": "a",
                "architectures": [
                    {"id": "a", "tier": "core", "license": "MIT"},
                ],
            }
        ],
        "entrypoints": {"train": "python /app/train.py"},
        "license": {"code": "Apache-2.0"},
    }
    assert nf.missing_entitlements(doc, [], task="detection") == ["extended.detection"]
    assert nf.missing_entitlements(doc, ["extended.detection"], task="detection") == []
    # The architecture announces core, so no entitlement is required for it.
    assert nf.missing_entitlements(doc, [], task="detection", arch="a") == []


def test_cli_init_scaffolds_a_valid_registrable_profile(tmp_path, capsys):
    target = tmp_path / "my-trainer"
    code = cli.main([
        "init", str(target), "--name", "acme/yolo", "--task", "detection",
        "--architecture", "acme-det", "--framework", "acme",
    ])
    assert code == 0
    for name in ("train.py", "nf_profile.py", "trainer_profile.py", "Dockerfile",
                 "README.md", "trainer-profile.json"):
        assert (target / name).is_file(), name

    document = json.loads((target / "trainer-profile.json").read_text(encoding="utf-8"))
    assert document["name"] == "acme/yolo"
    assert document["tasks"][0]["id"] == "detection"
    assert document["tasks"][0]["default_arch"] == "acme-det"
    assert nf.validate_profile(document) == []

    # The scaffold must contain no third-party framework import or weight.
    train_source = (target / "train.py").read_text(encoding="utf-8")
    assert "nereidforge_trainer" in train_source
    assert "build_training_metrics" in train_source
    assert "time.monotonic()" in train_source
    assert "ultralytics" not in train_source.lower()
    compile(train_source, str(target / "train.py"), "exec")

    # Generated builds must retain the same pristine/regional injection contract
    # as first-party images instead of silently pinning a package endpoint.
    dockerfile = (target / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG PYTHON_BASE_IMAGE=python:3.14-slim\nFROM ${PYTHON_BASE_IMAGE}" in dockerfile
    assert 'ARG PIP_INDEX_URL=""' in dockerfile
    assert 'ARG PIP_TRUSTED_HOST=""' in dockerfile
    assert "--index-url" not in dockerfile
    assert "https://" not in dockerfile

    assert cli.main(["validate", str(target / "trainer-profile.json")]) == 0


def test_cli_validate_metrics_checks_the_public_contract(tmp_path):
    path = tmp_path / "job.metrics.json"
    path.write_text(json.dumps({"schema": nf.TRAINING_METRICS_SCHEMA}), encoding="utf-8")
    assert cli.main(["validate-metrics", str(path)]) == 1

    path.write_text(json.dumps(nf.build_training_metrics({}, train_time_s=12.5)), encoding="utf-8")
    assert cli.main(["validate-metrics", str(path)]) == 0


def test_cli_validate_rejects_broken_profile(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema": "other"}', encoding="utf-8")
    assert cli.main(["validate", str(path)]) == 1


def test_trainer_torch_profile_validates_against_sdk(tmp_path):
    from nereidforge_trainer.profile import load_profile

    doc = load_profile(_torch_profile_path())
    assert nf.validate_profile(doc) == []
    assert nf.effective_tier(doc, task="detection") == "core"
    assert nf.effective_tier(doc, capability="optimization.tensorrt") == "extended"
    assert nf.missing_entitlements(doc, [], capability="optimization.tensorrt") == [
        "extended.optimization"
    ]
