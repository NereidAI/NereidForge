# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for Worker._source_model_name derivation.

A completed training job must be versioned under the model it was trained
from (base model stem / base-model pool directory), not under the job name.
"""

import types

from app.worker import Worker


def _job(config=None, params=None, name="some-job"):
    """Build a minimal Job-like object."""
    return types.SimpleNamespace(
        name=name, config=config or {}, params=params or {},
        tenant_id="default",
    )


def test_weight_filename_reduced_to_stem():
    assert Worker._source_model_name(_job({"base_model": "yolov8n.pt"})) == "yolov8n"
    assert Worker._source_model_name(_job({"base_model": "yolov8n-seg.pt"})) == "yolov8n-seg"


def test_bare_model_name_unchanged():
    assert Worker._source_model_name(_job({"base_model": "resnet50"})) == "resnet50"


def test_base_model_uri_uses_final_directory():
    job = _job({"base_model": "s3://nereidforge/base-models/qwen3/"})
    assert Worker._source_model_name(job) == "qwen3"


def test_base_model_uri_ending_on_weight_uses_parent_directory():
    job = _job({"base_model": "s3://nereidforge/base-models/qwen3/model.safetensors"})
    assert Worker._source_model_name(job) == "qwen3"


def test_registered_version_checkpoint_uses_model_directory():
    job = _job({
        "base_model": "s3://nereidforge/models/yolov8n/20260922-143000/best.pt"
    })
    assert Worker._source_model_name(job) == "yolov8n"


def test_params_model_id_is_fallback():
    job = _job(params={"model_id": "pooled-model"})
    assert Worker._source_model_name(job) == "pooled-model"


def test_job_name_is_last_resort():
    assert Worker._source_model_name(_job(name="my-job")) == "my-job"


def test_base_model_takes_precedence_over_params():
    job = _job(
        {"base_model": "yolo11n.pt"},
        params={"model_id": "other"},
    )
    assert Worker._source_model_name(job) == "yolo11n"


def _worker_with_models(get_model):
    """Build a Worker without __init__ carrying only a models service stub."""
    worker = Worker.__new__(Worker)
    worker.models = types.SimpleNamespace(get_model=get_model)
    return worker


def test_explicit_base_model_id_wins_over_checkpoint_path():
    worker = _worker_with_models(
        lambda tenant, ref: {"name": "mask-xpu-verify"} if ref == "m-34eca3a5d86d-77113" else None
    )
    job = _job({
        "base_model_id": "m-34eca3a5d86d-77113",
        "base_model": "s3://nereidforge/models/job-0ca4517f24c6-48594/job-0ca4517f24c6-48594.pt",
    })
    assert worker._resolve_source_model_name(job) == (
        "mask-xpu-verify", "m-34eca3a5d86d-77113"
    )


def test_unknown_base_model_id_falls_back_to_path_name():
    worker = _worker_with_models(lambda tenant, ref: None)
    job = _job({
        "base_model_id": "m-missing",
        "base_model": "s3://nereidforge/models/yolov8n/20260922-143000/best.pt",
    })
    assert worker._resolve_source_model_name(job) == ("yolov8n", "")
