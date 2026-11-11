# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Verify every builder and standalone KFP S3 credential boundary.

Modules are imported from their real paths, while boto3 and the XPU-only torch
import are replaced with inert test doubles. No object operation, network access,
or training path is executed.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import boto3
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDERS_ROOT = REPO_ROOT / "app" / "builders"
for import_root in (BUILDERS_ROOT, BUILDERS_ROOT / "sdk"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

SYNTHETIC_ACCESS_KEY = "SYNTHETIC_ROUTE_B_ACCESS_KEY"
SYNTHETIC_SECRET_KEY = "SYNTHETIC_ROUTE_B_SECRET_KEY"
S3_FACTORIES = (
    ("app/builders/common/s3_utils.py", "make_s3_client"),
    ("app/builders/common/model_package.py", "s3_client"),
    ("app/builders/evaluator/evaluate.py", "s3_client"),
    ("app/builders/optimizer/optimize.py", "s3_client"),
    ("app/builders/trainer-llm-cu/llm_train.py", "s3_client"),
    ("app/builders/trainer-llm-rocm/llm_train.py", "s3_client"),
    ("app/builders/trainer-llm-xpu/llm_train.py", "s3_client"),
    ("app/builders/trainer-llm-xpu/vlm_train.py", "s3_client"),
    ("app/builders/sdk/nereidforge_trainer/s3.py", "s3_client"),
    ("app/kfp-pipelines/docker/llm_train.py", "s3_client"),
    ("app/kfp-pipelines/docker/search_trial.py", "s3_client"),
    ("app/kfp-pipelines/docker/train.py", "s3_client"),
    ("app/kfp-pipelines/docker/evaluate.py", "s3_client"),
    ("app/kfp-pipelines/docker/optimize.py", "s3_client"),
)


def _load_s3_factory(relative_path: str, attribute: str, monkeypatch: pytest.MonkeyPatch):
    """_load_s3_factory imports an entrypoint without its accelerator runtime."""
    if "trainer-llm-xpu/" in relative_path:
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    module_name = "_route_b_" + relative_path.removesuffix(".py").replace("/", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return getattr(module, attribute)


@pytest.fixture
def boto3_client_spy(monkeypatch: pytest.MonkeyPatch):
    """boto3_client_spy records calls without constructing a real client."""
    calls = []
    sentinel = object()

    def fake_client(service, **kwargs):
        """fake_client returns an inert sentinel and records synthetic arguments."""
        calls.append((service, kwargs))
        return sentinel

    monkeypatch.setattr(boto3, "client", fake_client)
    return calls, sentinel


@pytest.mark.parametrize(("relative_path", "attribute"), S3_FACTORIES)
def test_s3_boundary_rejects_missing_and_blank_credentials(
    relative_path,
    attribute,
    monkeypatch,
    boto3_client_spy,
):
    """test_s3_boundary_rejects_missing_and_blank_credentials blocks client creation."""
    factory = _load_s3_factory(relative_path, attribute, monkeypatch)
    calls, _ = boto3_client_spy
    invalid_environments = (
        {},
        {"S3_ACCESS_KEY": SYNTHETIC_ACCESS_KEY},
        {"S3_SECRET_KEY": SYNTHETIC_SECRET_KEY},
        {"S3_ACCESS_KEY": "", "S3_SECRET_KEY": SYNTHETIC_SECRET_KEY},
        {"S3_ACCESS_KEY": SYNTHETIC_ACCESS_KEY, "S3_SECRET_KEY": " \t "},
    )
    for environment in invalid_environments:
        monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
        monkeypatch.delenv("S3_SECRET_KEY", raising=False)
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        with pytest.raises(RuntimeError) as exc_info:
            factory()
        message = str(exc_info.value)
        assert "S3_ACCESS_KEY" in message
        assert "S3_SECRET_KEY" in message
        assert "non-empty" in message
    assert calls == []


@pytest.mark.parametrize(("relative_path", "attribute"), S3_FACTORIES)
def test_s3_boundary_passes_only_explicit_synthetic_credentials(
    relative_path,
    attribute,
    monkeypatch,
    boto3_client_spy,
):
    """test_s3_boundary_passes_only_explicit_synthetic_credentials checks forwarding."""
    monkeypatch.setenv("S3_ACCESS_KEY", SYNTHETIC_ACCESS_KEY)
    monkeypatch.setenv("S3_SECRET_KEY", SYNTHETIC_SECRET_KEY)
    factory = _load_s3_factory(relative_path, attribute, monkeypatch)
    calls, sentinel = boto3_client_spy

    assert factory() is sentinel
    assert len(calls) == 1
    service, kwargs = calls[0]
    assert service == "s3"
    assert kwargs["aws_access_key_id"] == SYNTHETIC_ACCESS_KEY
    assert kwargs["aws_secret_access_key"] == SYNTHETIC_SECRET_KEY


def test_xpu_packager_image_layout_imports_without_source_tree(tmp_path: Path) -> None:
    """The XPU image ships every module imported by its copied packager entrypoint."""
    dockerfile = BUILDERS_ROOT / "evaluator-xpu" / "Dockerfile"
    instructions = dockerfile.read_text(encoding="utf-8")
    assert "COPY common/ /app/common/" in instructions
    assert "COPY common/model_package.py /app/model_package.py" in instructions

    # Reproduce the declared image layout, then launch outside the checkout with
    # no PYTHONPATH so only the copied entrypoint directory can supply `common`.
    image_app = tmp_path / "image" / "app"
    shutil.copytree(BUILDERS_ROOT / "common", image_app / "common")
    shutil.copy2(BUILDERS_ROOT / "common" / "model_package.py", image_app / "model_package.py")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(image_app / "model_package.py"), "--help"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--spec-uri" in completed.stdout
