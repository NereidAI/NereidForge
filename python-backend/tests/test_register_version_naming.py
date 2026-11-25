# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for Worker.register_trained_model version naming.

The registered version label must equal the source job id, so every model
version maps 1:1 to the training job that produced it.
"""

import types

from app.worker import Worker


def _job(job_id="job-1e9b7d4fcee3-11023", name="Mask Wearing Run", config=None, params=None):
    """Build a minimal Job-like object."""
    return types.SimpleNamespace(
        id=job_id, name=name, tenant_id="default",
        config=config or {"base_model": "yolov8n.pt", "task_type": "detection"},
        params=params or {}, kfp_run_id="run-abc",
    )


def _make_worker():
    """Build a Worker without __init__ and capture create_version calls."""
    worker = Worker.__new__(Worker)
    worker.models = types.SimpleNamespace(
        create_model=lambda tenant, name, desc, framework, tt, actor: {
            "id": "m-123", "name": name,
        },
        create_version=_capture,
        register_to_hub=lambda *a, **k: None,
    )
    worker.s3 = types.SimpleNamespace(object_size=lambda path: 100)
    return worker


_CAPTURED: list[dict] = []


def _capture(tenant, model_id, version, description, file_path, size_bytes, metadata, actor):
    """Record the create_version arguments and return a version dict."""
    out = {"version": version, "metadata": metadata}
    _CAPTURED.append(out)
    return out


def test_version_label_equals_job_id():
    _CAPTURED.clear()
    worker = _make_worker()
    job = _job()
    result = worker.register_trained_model(job, {"weights_uri": "s3://b/m.pt"}, source_model_id="")
    assert result["version"] == job.id
    assert _CAPTURED[0]["version"] == job.id


def test_metadata_records_job_id_and_job_name():
    _CAPTURED.clear()
    worker = _make_worker()
    job = _job()
    worker.register_trained_model(job, {"weights_uri": "s3://b/m.pt"}, source_model_id="")
    meta = _CAPTURED[0]["metadata"]
    assert meta["job_id"] == job.id
    assert meta["job_name"] == job.name
