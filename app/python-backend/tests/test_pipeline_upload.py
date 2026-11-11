# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for first-time and repeat NF pipeline registration."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[2] / "kfp-pipelines" / "upload_pipelines.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("upload_pipelines", MODULE_PATH)
assert SPEC and SPEC.loader
upload_pipelines = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(upload_pipelines)


class FakeCompiler:
    """Avoid reparsing pipeline DAGs when testing KFP/DB synchronization."""

    def compile(self, pipeline_func, package_path):
        Path(package_path).write_text("compiled")


class FakeKFP:
    """Small KFP fake exposing only the registration operations under test."""

    def __init__(self, existing_ids=False):
        self.created = []
        self.versions = []
        self.experiments = 0
        self.existing_ids = existing_ids

    def get_experiment(self, experiment_name, namespace):
        if not self.existing_ids:
            raise LookupError("missing")
        return SimpleNamespace(experiment_id="experiment-existing")

    def create_experiment(self, name, namespace):
        self.experiments += 1
        return SimpleNamespace(experiment_id="experiment-created")

    def list_pipelines(self, page_token="", page_size=100, namespace=None):
        return SimpleNamespace(pipelines=[], next_page_token="")

    def upload_pipeline(self, pipeline_package_path, pipeline_name, namespace):
        self.created.append(pipeline_name)
        short_name = pipeline_name.removeprefix("nereidforge-").removesuffix("-pipeline")
        return SimpleNamespace(pipeline_id=f"pid-{short_name}")

    def upload_pipeline_version(self, pipeline_package_path, pipeline_version_name, pipeline_id):
        self.versions.append((pipeline_id, pipeline_version_name))
        return SimpleNamespace(pipeline_version_id=f"version-{pipeline_id}")


def test_empty_config_creates_and_saves_all_registrations(monkeypatch):
    """A fresh install must not skip every upload and leave training unstartable."""
    monkeypatch.setattr(upload_pipelines.compiler, "Compiler", FakeCompiler)
    client = FakeKFP()
    config = {}
    saves = []

    upload_pipelines.synchronize_pipelines(client, config, on_update=lambda value: saves.append(value.copy()))

    assert client.experiments == 1
    assert client.created == [
        "nereidforge-training-pipeline",
        "nereidforge-optimization-pipeline",
        "nereidforge-evaluation-pipeline",
    ]
    assert config["kfp_experiment_id"] == "experiment-created"
    assert config["kfp_training_pipeline_id"] == "pid-training"
    assert config["kfp_optimization_pipeline_id"] == "pid-optimization"
    assert config["kfp_evaluation_pipeline_id"] == "pid-evaluation"
    assert len(saves) == 4


def test_existing_config_uploads_versions_without_duplicates(monkeypatch):
    """A rerun reuses persisted IDs and only adds new pipeline versions."""
    monkeypatch.setattr(upload_pipelines.compiler, "Compiler", FakeCompiler)
    client = FakeKFP(existing_ids=True)
    config = {
        "kfp_experiment_id": "experiment-existing",
        "kfp_training_pipeline_id": "pid-training",
        "kfp_optimization_pipeline_id": "pid-optimization",
        "kfp_evaluation_pipeline_id": "pid-evaluation",
    }

    upload_pipelines.synchronize_pipelines(client, config)

    assert client.experiments == 0
    assert client.created == []
    assert [item[0] for item in client.versions] == [
        "pid-training",
        "pid-optimization",
        "pid-evaluation",
    ]
