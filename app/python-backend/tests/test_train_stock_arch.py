# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for architecture resolution when training starts from pool weights.

Pit 184: a base-model pool selection (``s3://.../yolov8n.pt``) used to reset
the vision architecture to the task default (``fasterrcnn_mobilenet_v3_large_fpn``),
so a customer ultralytics trainer image received
``model=fasterrcnn_mobilenet_v3_large_fpn`` plus ``base_weights_uri=...yolov8n.pt``
— the picked base model never took effect. A stock checkpoint file name must
instead provide the architecture id (``model=yolov8n``) while arbitrary
checkpoint names keep the old fallback behaviour.
"""

import json
import types

from app.services.jobs import JobService


class _FakeS3:
    """Minimal S3 double that records the uploaded train_config.json."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    @staticmethod
    def tenant_uri(uri: str, tenant_id: str) -> str:
        return uri

    @staticmethod
    def tenant_filer_path(path: str, tenant_id: str) -> str:
        return path

    def put_object(self, path: str, body: bytes) -> None:
        self.objects[path] = body


def _service() -> JobService:
    """Build a JobService double with trainer config pinned (CPU vendor)."""
    svc = JobService.__new__(JobService)
    svc.s3 = _FakeS3()

    class _DB:
        @staticmethod
        def scalar(stmt):
            return types.SimpleNamespace(uri="s3://nereidforge/datasets/ds/v1")

    svc.db = _DB()
    svc.dataset = None
    svc._trainer_cfg = lambda: ("", 0, "500m", "1", "1Gi", "2Gi", "")
    # Registered-version lookups must not interfere with these cases.
    svc._source_version_arch = lambda *args: ""
    return svc


def _train_config(svc: JobService, config: dict, params: dict | None = None) -> dict:
    """Run CV pipeline params and parse the uploaded train_config.json."""
    job = types.SimpleNamespace(
        id="j1", tenant_id="default",
        params=params or {"dataset_id": "ds"}, config=config,
    )
    svc.training_pipeline_params(
        "default", dict(job.params), job
    )
    key = "/buckets/nereidforge/train-input/j1/train_config.json"
    return json.loads(svc.s3.objects[key].decode())


def test_pool_stock_weights_derive_model_arch():
    """A pool S3 URI named like a stock checkpoint carries its arch id."""
    cfg = _train_config(_service(), {
        "accelerator": "cpu",
        "base_model": "s3://nereidforge/base-models/cv/yolov8n.pt",
    })

    assert cfg["model"] == "yolov8n"
    assert cfg["base_weights_uri"] == "s3://nereidforge/base-models/cv/yolov8n.pt"


def test_bare_legacy_stock_name_derives_model_arch():
    """A bare legacy stock file name (no URI slash) works the same way."""
    cfg = _train_config(_service(), {
        "accelerator": "cpu",
        "base_model": "yolo11x-seg.pt",
        "task_type": "segmentation",
    })

    assert cfg["model"] == "yolo11x-seg"
    assert cfg["legacy_base_model"] == "yolo11x-seg.pt"
    assert "base_weights_uri" not in cfg


def test_arbitrary_pool_weights_keep_task_default_arch():
    """Random checkpoint names still rebuild the task-default arch."""
    cfg = _train_config(_service(), {
        "accelerator": "cpu",
        "base_model": "s3://nereidforge/base-models/cv/my-finetuned.pt",
    })

    assert cfg["model"] == "fasterrcnn_mobilenet_v3_large_fpn"
    assert cfg["base_weights_uri"] == "s3://nereidforge/base-models/cv/my-finetuned.pt"


def test_bare_architecture_id_is_untouched():
    """A plain torchvision architecture id keeps meaning "arch, no weights"."""
    cfg = _train_config(_service(), {
        "accelerator": "cpu",
        "base_model": "fcos_resnet50_fpn",
    })

    assert cfg["model"] == "fcos_resnet50_fpn"
    assert "base_weights_uri" not in cfg
    assert "legacy_base_model" not in cfg


def test_stock_arch_flows_to_custom_yolo_trainer_image():
    """Reproduces the reported job: custom ultralytics image + pool yolov8n.pt."""
    svc = _service()
    cfg = _train_config(svc, {
        "accelerator": "intel",
        "gpu_count": 1,
        "base_model": "s3://nereidforge/base-models/cv/yolov8n.pt",
    }, params={
        "dataset_id": "ds",
        "custom_image": "registry.example.com/yiqisoft/yolov8-trainer-xpu:v1",
    })

    assert cfg["image"].endswith("yolov8-trainer-xpu:v1")
    # The custom trainer reads CONFIG_JSON["model"] and ignores base_weights_uri.
    assert cfg["model"] == "yolov8n"


def test_explicit_architecture_and_starting_weights_stay_independent():
    """The split UI controls resolve without overloading a checkpoint URI."""
    cfg = _train_config(_service(), {
        "accelerator": "cpu",
        "architecture": "fcos_resnet50_fpn",
        "base_model": "s3://nereidforge/base-models/cv/my-finetuned.pt",
        "base_weights_uri": "s3://nereidforge/base-models/cv/my-finetuned.pt",
        "pretrained": False,
    })

    assert cfg["model"] == "fcos_resnet50_fpn"
    assert cfg["base_weights_uri"] == "s3://nereidforge/base-models/cv/my-finetuned.pt"
    assert cfg["pretrained"] is False


def test_explicit_architecture_overrides_registered_source_metadata():
    """Transfer learning may replace the source arch without losing lineage."""
    svc = _service()
    svc._source_version_arch = lambda *args: "source_arch"
    cfg = _train_config(svc, {
        "accelerator": "cpu",
        "architecture": "override_arch",
        "base_model": "s3://nereidforge/models/source/v1.pt",
        "base_weights_uri": "s3://nereidforge/models/source/v1.pt",
        "base_model_id": "m-source",
        "pretrained": False,
    })

    assert cfg["model"] == "override_arch"
    assert cfg["base_arch"] == "source_arch"


def test_explicit_scratch_initialization_disables_upstream_weights():
    """Scratch is an explicit choice rather than an absent checkpoint URI."""
    cfg = _train_config(_service(), {
        "accelerator": "cpu",
        "task_type": "classification",
        "architecture": "resnet50",
        "base_model": "resnet50",
        "pretrained": False,
    })

    assert cfg["model"] == "resnet50"
    assert cfg["pretrained"] is False
    assert "base_weights_uri" not in cfg


def test_checkpoint_without_explicit_architecture_keeps_stock_inference():
    """An empty override preserves registered/stock/default arch resolution."""
    cfg = _train_config(_service(), {
        "accelerator": "cpu",
        "base_model": "s3://nereidforge/base-models/cv/yolov8n.pt",
        "base_weights_uri": "s3://nereidforge/base-models/cv/yolov8n.pt",
        "pretrained": False,
    })

    assert cfg["model"] == "yolov8n"
    assert cfg["base_weights_uri"] == "s3://nereidforge/base-models/cv/yolov8n.pt"
