# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Kale notebook runtime (I-12, ADR-013 Case 2).

Covers the post-processing seam in app/services/notebook_runtime.py and the
worker's registration path:
  1. minimal kale tagging + forced approved step image (auto_tag),
  2. pip-clone bootstrap stripping + run-env injection + image allowlist,
  3. typed-artifact manifest extraction from a compiled pipeline,
  4. notebook model registration (declared typed output vs. fallback ranking).
"""

import types
from pathlib import Path

import nbformat
import yaml

from app.services import notebook_runtime as nrt
from app.worker import Worker

# A minimal sanitized fixture preserves the Kale/KFP post-processing contract.
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "notebook_pipeline.yaml"


def _docs() -> list[dict]:
    with open(_FIXTURE) as f:
        return list(yaml.safe_load_all(f))


# -- auto_tag ---------------------------------------------------------------

def test_auto_tag_marks_untagged_cells():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("import numpy"),
        nbformat.v4.new_code_cell("print('hi')"),
    ]
    path = Path("/tmp") / "auto-tag.ipynb"
    nbformat.write(nb, path)
    nrt.auto_tag(str(path), image="")
    reloaded = nbformat.read(path, as_version=4)
    tags = [list(c.metadata.get("tags") or []) for c in reloaded.cells]
    assert tags[0] == ["imports"]
    assert tags[1] == ["step:main"]


def test_auto_tag_forces_approved_image():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("x = 1", metadata={"tags": ["step:main", "image:evil/image:v9"]}),
    ]
    path = Path("/tmp") / "auto-tag-image.ipynb"
    nbformat.write(nb, path)
    nrt.auto_tag(str(path), image=nrt.DEFAULT_STEP_IMAGE)
    reloaded = nbformat.read(path, as_version=4)
    tags = reloaded.cells[0].metadata["tags"]
    assert "image:evil/image:v9" not in tags
    assert f"image:{nrt.DEFAULT_STEP_IMAGE}" in tags


# -- post-processing --------------------------------------------------------

def test_strip_pip_bootstrap_removes_pip_clone():
    container = _docs()[0]["deploymentSpec"]["executors"]["exec-main-step"]["container"]
    assert '"$0" "$@"' in container["command"][2]
    nrt.strip_pip_bootstrap(container)
    assert len(container["command"]) >= 3
    assert container["command"][0] == "sh"
    assert "pip install" not in yaml.safe_dump(container["command"])
    assert "executor_main" in container["command"][2]


def test_inject_env_adds_nf_vars_and_keeps_home():
    container = _docs()[0]["deploymentSpec"]["executors"]["exec-main-step"]["container"]
    nrt.inject_env(container, {
        "S3_ENDPOINT": "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333",
        "NF_JOB_ID": "job9",
        "NF_MODEL_DIR": "s3://nereidforge/models/job9",
    })
    names = [e["name"] for e in container["env"]]
    assert {"HOME", "S3_ENDPOINT", "NF_JOB_ID", "NF_MODEL_DIR"} <= set(names)
    assert container["env"][names.index("NF_JOB_ID")]["value"] == "job9"


def test_validate_images_rejects_non_approved():
    import pytest

    docs = _docs()
    container = docs[0]["deploymentSpec"]["executors"]["exec-main-step"]["container"]
    container["image"] = "docker.io/library/python:3.13"
    with pytest.raises(ValueError):
        nrt.validate_images(docs)


def test_validate_images_accepts_dual_use_image():
    docs = _docs()
    container = docs[0]["deploymentSpec"]["executors"]["exec-main-step"]["container"]
    container["image"] = nrt.DEFAULT_STEP_IMAGE
    nrt.validate_images(docs)  # must not raise


def test_extract_outputs_returns_typed_manifest():
    manifest = nrt.extract_outputs(_docs())
    assert {"artifact": "main_html_report", "type": "system.HTML"} in manifest


def test_compile_process_round_trips_the_fixture():
    from io import StringIO

    docs = _docs()
    for container in nrt._executors(docs):
        nrt.strip_pip_bootstrap(container)
        nrt.inject_env(container, {"NF_JOB_ID": "job9", "NF_MODEL_DIR": "s3://nereidforge/models/job9"})
    buf = StringIO()
    yaml.safe_dump_all(docs, buf)
    reloaded = list(yaml.safe_load_all(buf.getvalue()))
    executor = reloaded[0]["deploymentSpec"]["executors"]["exec-main-step"]["container"]
    assert "pip install" not in yaml.safe_dump(executor["command"])
    assert "NF_JOB_ID" in [e["name"] for e in executor["env"]]


# -- worker registration ----------------------------------------------------

class _FakeS3:
    """Minimal SeaweedFS client double for artifact discovery."""

    def __init__(self, files: list[str], sizes: dict[str, int] | None = None):
        self.files = files
        self.sizes = sizes or {}

    def walk(self, filer_path: str) -> list[str]:
        return list(self.files)

    def object_size(self, path: str) -> int:
        return self.sizes.get(path, 1)


def _job(params: dict | None = None, name: str = "job1") -> types.SimpleNamespace:
    return types.SimpleNamespace(id="job1", tenant_id="default", name=name, params=params or {})


def _register(files, sizes, params, metrics=None) -> tuple[object, dict]:
    calls = {}

    def _stub(job, metrics_in, source_model_id, task_type_override=None):
        calls["metrics"] = metrics_in
        calls["source_model_id"] = source_model_id
        calls["task_type_override"] = task_type_override
        return {"id": "m1", "name": job.name, "version": "v1"}

    worker = types.SimpleNamespace(
        s3=_FakeS3(files, sizes),
        register_trained_model=_stub,
    )
    result = Worker.register_notebook_model(worker, _job(params), metrics or {})
    return result, calls


def test_register_skips_when_no_model_artifact():
    result, calls = _register(
        ["/buckets/nereidforge/models/job1/metrics.json"],
        {},
        {},
    )
    assert result is None
    assert "metrics" not in calls


def test_register_uses_declared_typed_output():
    result, calls = _register(
        [
            "/buckets/nereidforge/models/job1/model.pt",
            "/buckets/nereidforge/models/job1/model.onnx",
            "/buckets/nereidforge/models/job1/metrics.json",
        ],
        sizes={
            "/buckets/nereidforge/models/job1/model.pt": 50,
            "/buckets/nereidforge/models/job1/model.onnx": 10,
        },
        params={"_kale_outputs": [{"artifact": "result", "type": "system.Model"}]},
    )
    assert result == {"id": "m1", "name": "job1", "version": "v1"}
    assert calls["task_type_override"] == "custom"
    assert calls["metrics"]["weights_uri"] == "s3://nereidforge/models/job1/model.pt"
    # With a declared Model output the largest blob wins over extension rank.
    assert "onnx_uri" not in calls["metrics"]


def test_register_falls_back_to_onnx_ranking():
    result, calls = _register(
        [
            "/buckets/nereidforge/models/job1/model.pt",
            "/buckets/nereidforge/models/job1/model.onnx",
        ],
        sizes={
            "/buckets/nereidforge/models/job1/model.pt": 9999,
            "/buckets/nereidforge/models/job1/model.onnx": 1,
        },
        params={},
    )
    # No typed Model output: the onnx file (top extension rank) wins.
    assert calls["metrics"]["onnx_uri"] == "s3://nereidforge/models/job1/model.onnx"


def test_register_honours_task_type_from_params():
    result, calls = _register(
        ["/buckets/nereidforge/models/job1/model.safetensors"],
        {},
        params={"task_type": "llm"},
    )
    assert calls["task_type_override"] == "llm"
    assert calls["metrics"]["weights_uri"] == "s3://nereidforge/models/job1/model.safetensors"


def test_register_merges_existing_weight_uri():
    result, calls = _register(
        ["/buckets/nereidforge/models/job1/model.pt"],
        {},
        params={},
        metrics={"weights_uri": "s3://nereidforge/models/job1/explicit.pt"},
    )
    assert calls["metrics"]["weights_uri"] == "s3://nereidforge/models/job1/explicit.pt"


def test_metric_candidates_notebook_use_bare_bucket():
    from app.clients.s3 import SeaweedFSClient

    worker = types.SimpleNamespace(s3=types.SimpleNamespace(
        tenant_filer_path=SeaweedFSClient.tenant_filer_path,
    ))
    nb = types.SimpleNamespace(
        id="job1", kfp_run_id="run1", tenant_id="nereidforge", batch_id=None, type="notebook"
    )
    tr = types.SimpleNamespace(
        id="job1", kfp_run_id="run1", tenant_id="nereidforge", batch_id=None, type="training"
    )
    # Notebook model dir is bare (ADR-013 D6: s3://<bucket>/models/{job_id}/),
    # so read_metrics must NOT tenant-prefix the candidates.
    nb_paths = Worker._metric_candidates(worker, nb, "models")
    assert all(not p.startswith("/buckets/nereidforge/nereidforge/") for p in nb_paths)
    assert "/buckets/nereidforge/models/job1/job1.metrics.json" in nb_paths
    # Trainer pipelines, by contrast, stay tenant-scoped.
    tr_paths = Worker._metric_candidates(worker, tr, "models")
    assert all("/nereidforge/models/" in p for p in tr_paths)
