# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for I-11 custom training scripts end-to-end.

Covers the three code paths added for user-supplied ``custom_image`` +
``custom_script_uri`` jobs:
  1. the API body forwards both selectors into job params,
  2. the KFP train step turns ``custom_script_uri`` into a fetch-and-run
     trainer command (and honours explicit command/args / rejects bad URIs),
  3. the worker discovers a produced artifact when a custom script did not
     follow the trainer's metrics.json contract.
"""

import json
import sys
import types
from pathlib import Path

import pytest

from app.api.routers.jobs import JobCreate, JobUpdate, _build_params
from app.core.images import nf_image
from app.services.accelerators import get_profile
from app.services.jobs import JobService
from app.worker import Worker

# The KFP components live in a sibling app (app/kfp-pipelines) and are not on
# the backend's import path; add it so the train_step tests can import them.
_KFP_PIPELINES = Path(__file__).resolve().parents[2] / "kfp-pipelines"
if str(_KFP_PIPELINES) not in sys.path:
    sys.path.insert(0, str(_KFP_PIPELINES))


# -- API body forwarding ----------------------------------------------------

def test_build_params_forwards_custom_selectors():
    """custom_image/custom_script_uri must reach job params untouched."""
    req = JobCreate(
        name="job", custom_image="hub/your/trainer:v1",
        custom_script_uri="s3://nereidforge/scripts/train.py",
    )
    params = _build_params(req)
    assert params["custom_image"] == "hub/your/trainer:v1"
    assert params["custom_script_uri"] == "s3://nereidforge/scripts/train.py"


def test_build_params_omits_absent_selectors():
    """No custom selectors => no keys (keeps the built-in trainer path clean)."""
    params = _build_params(JobCreate(name="job"))
    assert "custom_image" not in params
    assert "custom_script_uri" not in params


def test_job_update_accepts_custom_selectors():
    """An edit body may carry the selectors so they round-trip."""
    assert "custom_image" in JobUpdate.model_fields
    assert "custom_script_uri" in JobUpdate.model_fields


def test_stale_builtin_cpu_image_is_corrected_for_xpu():
    """A cloned CPU built-in image must not run against an Intel XPU profile."""
    selected = nf_image("trainer/torch-cpu")
    expected = nf_image("trainer/torch-xpu")

    resolved = JobService._resolve_builtin_trainer_image(selected, expected)

    assert resolved == expected


def test_unknown_custom_image_is_preserved_for_xpu():
    """Only known built-in images are automatically vendor-corrected."""
    selected = "registry.example.com/team/custom-yolo:20260920"
    expected = nf_image("trainer/torch-xpu")

    resolved = JobService._resolve_builtin_trainer_image(selected, expected)

    assert resolved == selected


def test_matching_builtin_image_is_preserved_for_vendor():
    """The vendor's own built-in image remains selected."""
    selected = nf_image("trainer/torch-xpu")
    expected = nf_image("trainer/torch-xpu")

    resolved = JobService._resolve_builtin_trainer_image(selected, expected)

    assert resolved == expected


def test_nvidia_gpu_count_is_forwarded():
    """An NVIDIA job may request multiple GPUs per node."""
    profile = get_profile("nvidia")

    assert JobService._requested_gpu_count({"gpu_count": 2}, {}, profile, 1) == 2


def test_gpu_vendor_defaults_to_one_gpu():
    """A GPU vendor without a count receives one GPU."""
    profile = get_profile("amd")

    assert JobService._requested_gpu_count({}, {}, profile, 0) == 1


def test_intel_shared_device_count_is_not_capped():
    """Shared vGPU devices are allowed; the count passes through unclamped."""
    profile = get_profile("intel")

    assert JobService._requested_gpu_count({"gpu_count": 4}, {}, profile, 1) == 4
    assert JobService._requested_gpu_count({"gpu_count": 20}, {}, profile, 1) == 20


def test_multi_gpu_device_strings_match_vendor_contracts():
    """Vendor device strings match what Ultralytics expects."""
    assert JobService._trainer_device("nvidia", get_profile("nvidia"), 2) == "0,1"
    assert JobService._trainer_device("amd", get_profile("amd"), 3) == "0,1,2"
    assert JobService._trainer_device("intel", get_profile("intel"), 2) == "xpu:0"


# -- KFP train step: custom script -> command/args --------------------------

class _FakeCustomObjectsApi:
    """Capture the TrainJob body and immediately report success."""

    def __init__(self):
        self.created = None

    def create_namespaced_custom_object(self, **kwargs):
        self.created = kwargs["body"]

    def get_namespaced_custom_object(self, **kwargs):
        return {
            "status": {
                "conditions": [{"type": "Succeeded", "status": "True"}],
                "modelUri": "s3://nereidforge/models/job/",
                "metrics": {},
            }
        }


def _run_train_step(monkeypatch, train_config: dict) -> dict:
    """Invoke the train_step component body with Kubernetes faked out.

    Returns the captured TrainJob body so tests can assert on the trainer
    command/args. SystemExit(0) is expected on the simulated success.
    """
    import kubernetes
    import kubernetes.client.rest

    import importlib

    ts = importlib.import_module("components.train_step")

    api = _FakeCustomObjectsApi()
    monkeypatch.setattr(kubernetes.client, "CustomObjectsApi", lambda: api)
    monkeypatch.setattr(kubernetes.config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(kubernetes.client.rest, "ApiException", Exception)

    with pytest.raises(SystemExit) as exc:
        ts.train_step.python_func(
            input_uri="s3://nereidforge/data/",
            output_dir="s3://nereidforge/models/job/",
            config_json=json.dumps(train_config),
            artifact_id="job",
            trainjob_name="train-job",
            namespace="nereidforge",
            poll_interval=0,
            # The pipeline passes the resolved image explicitly (pit 192); the
            # body must never fall back to a module-level constant.
            trainer_image="ghcr.io/nereidai/trainer-torch-cpu:0.0.1",
        )
    assert exc.value.code == 0
    return api.created


def test_train_step_runs_custom_script_from_filer(monkeypatch):
    """custom_script_uri -> python fetch-and-run of the filer HTTP URL."""
    body = _run_train_step(monkeypatch, {
        "image": "hub/your/trainer:v1",
        "custom_script_uri": "s3://nereidforge/scripts/train.py",
        "resourcesPerNode": {"requests": {}, "limits": {}},
    })
    trainer = body["spec"]["trainer"]
    assert trainer["command"] == ["python"]
    assert trainer["args"][0] == "-c"
    assert "runpy.run_path" in trainer["args"][1]
    assert trainer["args"][2].startswith("http://")
    assert trainer["args"][2].endswith("/buckets/nereidforge/scripts/train.py")
    # The custom image must be what the TrainJob actually runs.
    assert trainer["image"] == "hub/your/trainer:v1"


def test_train_step_accepts_http_script_uri(monkeypatch):
    """A plain http(s) script URI is used verbatim (no filer translation)."""
    body = _run_train_step(monkeypatch, {
        "custom_script_uri": "https://example.com/train.py",
        "resourcesPerNode": {"requests": {}, "limits": {}},
    })
    assert body["spec"]["trainer"]["args"][2] == "https://example.com/train.py"


def test_train_step_explicit_command_wins_over_script(monkeypatch):
    """Explicit command/args (custom entrypoint) override script handling."""
    body = _run_train_step(monkeypatch, {
        "custom_script_uri": "s3://nereidforge/scripts/train.py",
        "command": ["/usr/bin/mytrain"],
        "args": ["--epochs", "3"],
        "resourcesPerNode": {"requests": {}, "limits": {}},
    })
    trainer = body["spec"]["trainer"]
    assert trainer["command"] == ["/usr/bin/mytrain"]
    assert trainer["args"] == ["--epochs", "3"]


def test_train_step_rejects_non_s3_http_script(monkeypatch):
    """A selector that is neither s3:// nor http(s):// fails fast."""
    import kubernetes
    import kubernetes.client.rest

    import importlib

    ts = importlib.import_module("components.train_step")

    monkeypatch.setattr(kubernetes.client, "CustomObjectsApi", lambda: _FakeCustomObjectsApi())
    monkeypatch.setattr(kubernetes.config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(kubernetes.client.rest, "ApiException", Exception)

    with pytest.raises(SystemExit) as exc:
        ts.train_step.python_func(
            input_uri="s3://nereidforge/data/",
            output_dir="s3://nereidforge/models/job/",
            config_json=json.dumps({
                "custom_script_uri": "ftp://host/train.py",
                "resourcesPerNode": {"requests": {}, "limits": {}},
            }),
            artifact_id="job",
            trainjob_name="train-job",
            poll_interval=0,
            trainer_image="ghcr.io/nereidai/trainer-torch-cpu:0.0.1",
        )
    assert exc.value.code == 1


def test_train_step_falls_back_to_trainer_image_parameter(monkeypatch):
    """A config without ``image`` uses the component's resolved parameter."""
    body = _run_train_step(monkeypatch, {
        "resourcesPerNode": {"requests": {}, "limits": {}},
    })
    assert body["spec"]["trainer"]["image"] == (
        "ghcr.io/nereidai/trainer-torch-cpu:0.0.1"
    )


def test_train_step_rejects_missing_trainer_image(monkeypatch):
    """Neither config image nor parameter: fail before creating the TrainJob."""
    import kubernetes
    import kubernetes.client.rest

    import importlib

    ts = importlib.import_module("components.train_step")

    api = _FakeCustomObjectsApi()
    monkeypatch.setattr(kubernetes.client, "CustomObjectsApi", lambda: api)
    monkeypatch.setattr(kubernetes.config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(kubernetes.client.rest, "ApiException", Exception)

    with pytest.raises(SystemExit) as exc:
        ts.train_step.python_func(
            input_uri="s3://nereidforge/data/",
            output_dir="s3://nereidforge/models/job/",
            config_json=json.dumps({"resourcesPerNode": {"requests": {}, "limits": {}}}),
            artifact_id="job",
            trainjob_name="train-job",
            poll_interval=0,
        )
    assert exc.value.code == 1
    assert not api.created, "an image-less TrainJob must never be submitted"


# -- worker artifact discovery ----------------------------------------------

class _FakeS3:
    """Minimal SeaweedFS client double for artifact discovery."""

    def __init__(self, files: list[str], sizes: dict[str, int] | None = None):
        self.files = files
        self.sizes = sizes or {}

    @staticmethod
    def s3_uri_to_filer_path(uri: str) -> str:
        return "/buckets/" + uri[len("s3://"):].rstrip("/")

    def walk(self, filer_path: str) -> list[str]:
        return list(self.files)

    def object_size(self, path: str) -> int:
        return self.sizes.get(path, 1)

    @staticmethod
    def tenant_uri(uri: str, tenant_id: str) -> str:
        return uri


def _job(params: dict | None = None) -> types.SimpleNamespace:
    """Build a minimal Job-like object carrying id/tenant/params."""
    return types.SimpleNamespace(id="job1", tenant_id="default", params=params or {})


def _discover(files: list[str], sizes: dict[str, int] | None = None,
              params: dict | None = None) -> str:
    """Run Worker._discover_artifact_uri with a fake S3 client."""
    fake = types.SimpleNamespace(s3=_FakeS3(files, sizes))
    return Worker._discover_artifact_uri(fake, _job(params))


def test_discover_prefers_onnx_and_ignores_metrics_and_tensorboard():
    uri = _discover([
        "/buckets/nereidforge/models/job1/job1.metrics.json",
        "/buckets/nereidforge/models/job1/tensorboard/results.csv",
        "/buckets/nereidforge/models/job1/job1.pt",
        "/buckets/nereidforge/models/job1/job1.onnx",
    ], sizes={"/buckets/nereidforge/models/job1/job1.pt": 9999})
    assert uri == "s3://nereidforge/models/job1/job1.onnx"


def test_discover_falls_back_to_largest_weight():
    uri = _discover([
        "/buckets/nereidforge/models/job1/small.pt",
        "/buckets/nereidforge/models/job1/big.pt",
    ], sizes={
        "/buckets/nereidforge/models/job1/small.pt": 10,
        "/buckets/nereidforge/models/job1/big.pt": 100,
    })
    assert uri == "s3://nereidforge/models/job1/big.pt"


def test_discover_honours_explicit_output_uri():
    seen = {}

    class _RecordingS3(_FakeS3):
        def walk(self, filer_path: str) -> list[str]:
            seen["path"] = filer_path
            return ["/buckets/nereidforge/custom/run/out/model.safetensors"]

    fake = types.SimpleNamespace(s3=_RecordingS3([]))
    uri = Worker._discover_artifact_uri(
        fake, _job({"output_uri": "s3://nereidforge/custom/run/out/"})
    )
    assert seen["path"] == "/buckets/nereidforge/custom/run/out"
    assert uri == "s3://nereidforge/custom/run/out/model.safetensors"


def test_discover_returns_empty_when_nothing_useful():
    assert _discover(["/buckets/nereidforge/models/job1/job1.metrics.json"]) == ""
