# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for multi-GPU count propagation in CV training pipeline params.

The TrainJob ``torch-distributed`` runtime wraps the entrypoint with torchrun
using ``spec.trainer.numProcPerNode``; that field (and the NPROC env / GPU
resource count) must reflect the form-selected GPU count instead of a
hardcoded single process.
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

    @staticmethod
    def s3_uri_to_filer_path(uri: str) -> str:
        return "/buckets/" + uri[len("s3://"):].rstrip("/")

    def put_object(self, path: str, body: bytes) -> None:
        self.objects[path] = body

    def read_object(self, path: str) -> bytes:
        if path in self.objects:
            return self.objects[path]
        raise FileNotFoundError(path)


def _service() -> JobService:
    """Build a JobService double with trainer config pinned."""
    svc = JobService.__new__(JobService)
    svc.s3 = _FakeS3()

    class _DB:
        @staticmethod
        def scalar(stmt):
            return types.SimpleNamespace(uri="s3://nereidforge/datasets/ds/v1")

    svc.db = _DB()
    svc.dataset = None
    svc._trainer_cfg = lambda: ("", 0, "500m", "1", "1Gi", "2Gi", "img")
    return svc


def _train_config(svc: JobService, config: dict) -> dict:
    """Run CV pipeline params and parse the uploaded train_config.json."""
    job = types.SimpleNamespace(id="j1", params={}, config=config)
    svc.training_pipeline_params(
        "default", {"dataset_id": "ds"}, job
    )
    key = "/buckets/nereidforge/train-input/j1/train_config.json"
    return json.loads(svc.s3.objects[key].decode())


def test_nvidia_multi_gpu_count_reaches_trainjob():
    """numProcPerNode / resources / NPROC all carry the requested GPU count."""
    cfg = _train_config(
        _service(),
        {"accelerator": "nvidia", "gpu_count": 2, "epochs": 2},
    )

    assert cfg["numProcPerNode"] == 2
    assert cfg["resourcesPerNode"]["requests"]["nvidia.com/gpu"] == "2"
    assert cfg["resourcesPerNode"]["limits"]["nvidia.com/gpu"] == "2"
    assert cfg["device"] == "0,1"
    env = {e["name"]: e["value"] for e in cfg["env"]}
    assert env["NPROC"] == "2"


def test_cpu_job_keeps_single_process():
    """A CPU job must never ask torchrun for zero processes."""
    cfg = _train_config(_service(), {"accelerator": "cpu", "gpu_count": 0})

    assert cfg["numProcPerNode"] == 1
    assert "nvidia.com/gpu" not in cfg["resourcesPerNode"]["requests"]


def test_explicit_workers_reach_train_config():
    """Zero and positive DataLoader worker counts survive the CV config boundary."""
    for workers in (0, 6):
        cfg = _train_config(_service(), {"accelerator": "cpu", "workers": workers})
        assert cfg["workers"] == workers


def test_omitted_workers_preserve_trainer_default():
    """An absent optional field must not become a platform-owned default."""
    cfg = _train_config(_service(), {"accelerator": "cpu"})

    assert "workers" not in cfg
