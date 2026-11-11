# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO IR serving support: artifact detection, ISVC manifest, IR versioning.

Covers phase 1 of the vendor-native artifact work (Intel/OpenVINO): the
optimization pipeline produces (.xml, .bin) IR versions, deployment must
download the pair into OVMS, and such versions are named/marked as IR.
"""

import types

import pytest

from app.clients.k8s import InferenceServiceSpec, K8sService
from app.services.jobs import register_optimized_artifact
from app.services.model_format import (
    FORMAT_ONNX,
    FORMAT_OPENVINO_IR,
    FORMAT_TENSORRT_PLAN,
    ir_companion_bin,
    serving_format,
)


# -- format detection -------------------------------------------------------

def test_serving_format_classifies_artifacts():
    assert serving_format("s3://b/m/model.onnx") == FORMAT_ONNX
    assert serving_format("s3://b/m/model.optimized.xml") == FORMAT_OPENVINO_IR
    assert serving_format("s3://b/m/model.plan") == FORMAT_TENSORRT_PLAN
    assert serving_format("s3://b/m/model.engine") == FORMAT_TENSORRT_PLAN
    # Source/training formats are not serving artifacts.
    assert serving_format("s3://b/m/best.pt") is None


def test_ir_companion_bin_same_basename():
    assert ir_companion_bin("s3://b/o/m.optimized.xml") == "s3://b/o/m.optimized.bin"
    assert ir_companion_bin("s3://b/o/m.xml?token=x") == "s3://b/o/m.bin?token=x"
    assert ir_companion_bin("s3://b/o/m.onnx") is None


# -- ISVC naming --------------------------------------------------------------

def test_serving_isvc_name_short_pair_unchanged():
    from app.clients.k8s import serving_isvc_name

    assert serving_isvc_name("m-abc", "v1") == "model-m-abc-v1"


def test_serving_isvc_name_bounded_to_53_and_predictor_dns_ok():
    from app.clients.k8s import ISVC_NAME_MAX, serving_isvc_name

    mid = "m-7b26452669c5-10796"
    ver = "job-c25c5ab16221-32035-ir-fp32"
    name = serving_isvc_name(mid, ver)
    assert len(name) <= ISVC_NAME_MAX
    # Knative appends "-predictor"; the Configuration must stay DNS-1035 (<=63).
    assert len(f"{name}-predictor") <= 63
    # recognizable + deterministic across deploy/undeploy/serving-status calls
    assert name.endswith("32035-ir-fp32") or "ir-fp32" in name
    assert name == serving_isvc_name(mid, ver)


# -- ISVC manifest -----------------------------------------------------------

def _manifest(model_uri, runtime="kserve-openvino", gpu="cpu"):
    svc = K8sService("nereidforge")
    spec = InferenceServiceSpec(
        name="model-test",
        namespace="nereidforge",
        model_uri=model_uri,
        runtime_type=runtime,
        gpu=gpu,
    )
    return svc._isvc_manifest(spec)


def _command(manifest):
    return manifest["spec"]["predictor"]["containers"][0]["command"][2]


def test_ir_manifest_downloads_xml_and_bin_pair():
    m = _manifest("s3://nereidforge/optimized/x.optimized.xml")
    cmd = _command(m)
    assert "model.xml" in cmd and "model.bin" in cmd
    assert "x.optimized.xml" in cmd and "x.optimized.bin" in cmd
    assert "/ovms/bin/ovms" in cmd
    # IR must not be saved as onnx
    assert "model.onnx" not in cmd


def test_onnx_manifest_unchanged_single_file():
    cmd = _command(_manifest("s3://nereidforge/models/j/model.onnx"))
    assert "model.onnx" in cmd
    assert "model.xml" not in cmd


def test_ir_rejected_for_triton_runtime():
    with pytest.raises(ValueError, match="OpenVINO IR"):
        _manifest("s3://nereidforge/optimized/x.optimized.xml", runtime="kserve-tritonserver")


# -- optimization source validation ------------------------------------------

def _source_svc(row):
    from app.services.jobs import JobService

    svc = JobService.__new__(JobService)
    svc.db = types.SimpleNamespace(
        scalars=lambda q: types.SimpleNamespace(first=lambda: row))
    return svc


def test_validate_optimization_source_rejects_ir():
    from app.services.jobs import OptimizationSourceError

    row = types.SimpleNamespace(
        version="job-x-ir-fp32",
        file_path="s3://b/o/x.optimized.xml",
        metadata_={"framework": "openvino"},
    )
    svc = _source_svc(row)
    with pytest.raises(OptimizationSourceError, match="OpenVINO IR"):
        svc.validate_optimization_source(
            "default", {"source_model_id": "m1", "source_version": "job-x-ir-fp32"})


def test_validate_optimization_source_accepts_onnx():
    row = types.SimpleNamespace(
        version="job-x", file_path="s3://b/m/x.onnx", metadata_={})
    svc = _source_svc(row)
    svc.validate_optimization_source(
        "default", {"source_model_id": "m1", "source_version": "job-x"})


def test_validate_optimization_source_unknown_version_passes():
    # Resolution error is surfaced later with full context, not at create time.
    svc = _source_svc(None)
    svc.validate_optimization_source(
        "default", {"source_model_id": "m1", "source_version": "ghost"})


def test_validate_optimization_source_accepts_capability_bearing_onnx():
    """An explicit optimize capability is sufficient for a new ONNX upload."""
    row = types.SimpleNamespace(
        version="v1", file_path="s3://b/m/x.onnx",
        metadata_={"artifact_format": "onnx", "artifact_capabilities": ["optimize"]},
    )
    _source_svc(row).validate_optimization_source(
        "default", {"source_model_id": "m1", "source_version": "v1"})


@pytest.mark.parametrize("format_id", ["tflite", "tensorrt-plan", "ascend-om", "gguf", "openvino-ir"])
def test_validate_optimization_source_enforces_recorded_capabilities(format_id):
    """Registration-only and IR uploads cannot enter the ONNX optimizer."""
    from app.services.jobs import OptimizationSourceError

    row = types.SimpleNamespace(
        version="v1", file_path=f"s3://b/m/{format_id}.artifact",
        metadata_={"artifact_format": format_id, "artifact_capabilities": ["download"]},
    )
    with pytest.raises(OptimizationSourceError, match="does not support optimization"):
        _source_svc(row).validate_optimization_source(
            "default", {"source_model_id": "m1", "source_version": "v1"})


def test_katib_start_revalidates_source_before_launch():
    """The Katib branch cannot bypass capabilities on stale jobs or retries."""
    from app.services.jobs import JobService, OptimizationSourceError

    row = types.SimpleNamespace(
        version="v1", file_path="s3://b/m/model.om",
        metadata_={"artifact_format": "ascend-om", "artifact_capabilities": ["download"]},
    )
    svc = _source_svc(row)
    job = types.SimpleNamespace(
        status="pending", kfp_run_id="",
        params={
            "source_model_id": "m1", "source_version": "v1", "search_mode": True,
        },
    )
    svc.get_job = lambda *_args: job
    svc.kfp = object()
    svc._launch_katib_search = lambda *_args: pytest.fail("Katib launch must not be reached")
    with pytest.raises(OptimizationSourceError, match="does not support optimization"):
        JobService.start_job(svc, "optimization", "default", "job-1")


# -- optimized artifact registration -----------------------------------------

def _fake_models(captured):
    def create_version(tenant, model_id, version, description, file_path, size, metadata, actor):
        # Mirror ModelService._version_dict: id is the mv-* version row, while
        # model_id is the parent m-* model required by package enqueue.
        out = {
            "id": "mv-1", "model_id": model_id, "version": version,
            "file_path": file_path, "size": size, "metadata": metadata,
        }
        captured["versions"].append(out)
        return out

    return types.SimpleNamespace(
        db=types.SimpleNamespace(scalar=lambda q: None),
        s3=types.SimpleNamespace(object_size=lambda uri: 200 if uri.endswith(".bin") else 100),
        get_model=lambda tenant, mid: {"id": mid, "name": "mask"},
        create_model=lambda *a, **k: {"id": "m-new", "name": a[1]},
        create_version=create_version,
        register_to_hub=lambda name, uri, version, fmt, meta: captured["hubs"].append((uri, fmt)),
    )


def _opt_job(target_device, quant_method, source_version="job-base-1"):
    return types.SimpleNamespace(
        id="job-opt-1",
        name="opt",
        tenant_id="default",
        kfp_run_id="run-opt",
        params={
            "target_device": target_device,
            "quant_method": quant_method,
            "source_version": source_version,
            "source_model_id": "m-1",
        },
    )


def test_openvino_graph_artifact_registers_ir_version():
    captured = {"versions": [], "hubs": []}
    metrics = {
        "optimized_uri": "s3://nereidforge/optimized/x.optimized.xml",
        "extra_artifacts": ["s3://nereidforge/optimized/x.optimized.bin"],
    }
    out = register_optimized_artifact(_fake_models(captured), _opt_job("intel", "graph"), metrics)
    assert out["version"] == "job-base-1-ir-fp32"
    v = captured["versions"][0]
    assert v["metadata"]["framework"] == "openvino"
    assert v["metadata"]["ir_bin_uri"].endswith("x.optimized.bin")
    assert v["size"] == 300  # xml + bin
    assert captured["hubs"][0][1] == "openvino"


def test_ir_extra_artifacts_are_complete_uris_not_characters():
    captured = {"versions": [], "hubs": []}
    metrics = {
        "optimized_uri": "s3://nereidforge/optimized/x.optimized.xml",
        # Legacy optimizer runs emitted no list here; the companion is derived.
        "extra_artifacts": [],
    }
    register_optimized_artifact(_fake_models(captured), _opt_job("intel", "graph"), metrics)
    extra = captured["versions"][0]["metadata"]["extra_artifacts"]
    assert extra == ["s3://nereidforge/optimized/x.optimized.bin"]
    # Every entry is a complete s3:// URI, never a character fragment from
    # star-unpacking a string (the old '-', '.', '/', '0', ... bug).
    assert all(isinstance(u, str) and u.startswith("s3://") for u in extra)
    assert all(len(u) > 8 for u in extra)


def test_openvino_int8_suffix():
    captured = {"versions": [], "hubs": []}
    out = register_optimized_artifact(
        _fake_models(captured), _opt_job("openvino", "int8-static"),
        {"optimized_uri": "s3://b/x.xml"},
    )
    assert out["version"] == "job-base-1-ir-int8"


def _metrics_worker():
    from app.worker import Worker
    w = Worker.__new__(Worker)
    w.s3 = types.SimpleNamespace(tenant_filer_path=lambda p, t: p)
    return w


def _metrics_job(jtype="optimization", batch=""):
    import types as _t
    return _t.SimpleNamespace(
        id="job-abc", kfp_run_id="run-xyz", tenant_id="default",
        type=jtype, batch_id=batch or None,
    )


def test_auto_package_uses_parent_model_id_not_version_row_id():
    """The registration result carries both ids; enqueue needs the parent m-*."""
    captured = {"versions": [], "hubs": []}
    metrics = {"optimized_uri": "s3://nereidforge/optimized/x.optimized.xml"}
    job = _opt_job("intel", "fp16")
    registered = register_optimized_artifact(_fake_models(captured), job, metrics)
    assert registered["id"] == "mv-1"
    assert registered["model_id"] == "m-1"

    worker = _metrics_worker()
    calls = []
    worker.packages = types.SimpleNamespace(
        enqueue=lambda *args: calls.append(args) or {"task_id": "task-1"}
    )
    worker._maybe_auto_package(job, metrics, {"gate_result": "PASS"}, registered)

    assert len(calls) == 1
    tenant, model_ref, version, precision, actor = calls[0]
    assert (tenant, model_ref, version, precision, actor) == (
        "default", "m-1", "job-base-1-ir-fp16", "fp16", "nereidforge-worker-auto"
    )


def test_optimization_metrics_candidates_include_double_nested_dir():
    w = _metrics_worker()
    paths = w._metric_candidates(_metrics_job(), "optimized")
    assert "/buckets/nereidforge/optimized/job-abc/job-abc/job-abc.metrics.json" in paths


def test_optimization_batch_candidate_single_nested_under_batch():
    w = _metrics_worker()
    paths = w._metric_candidates(_metrics_job(batch="batch-1"), "optimized")
    assert "/buckets/nereidforge/optimized/batch-1/job-abc/job-abc.metrics.json" in paths


def test_training_candidates_have_no_double_nested_dir():
    w = _metrics_worker()
    paths = w._metric_candidates(_metrics_job(jtype="training"), "models")
    assert "/buckets/nereidforge/models/job-abc/job-abc/job-abc.metrics.json" not in paths


def test_openvino_uri_double_slash_is_normalized():
    captured = {"versions": [], "hubs": []}
    metrics = {
        # A trailing-slash output_uri historically produced "opt//id/id.xml".
        "optimized_uri": "s3://nereidforge/optimized/x//x/x.optimized.xml",
        "extra_artifacts": ["s3://nereidforge/optimized/x//x/x.optimized.bin"],
    }
    out = register_optimized_artifact(
        _fake_models(captured), _opt_job("intel", "graph"), metrics
    )
    v = captured["versions"][0]
    assert v["file_path"].count("//") == 1  # only the s3:// scheme separator
    assert v["metadata"]["ir_bin_uri"].endswith("x/x.optimized.bin")
    assert v["metadata"]["ir_bin_uri"].count("//") == 1
    assert out["version"] == "job-base-1-ir-fp32"


def test_cpu_graph_artifact_keeps_onnx_naming():
    captured = {"versions": [], "hubs": []}
    out = register_optimized_artifact(
        _fake_models(captured), _opt_job("cpu", "graph"),
        {"optimized_uri": "s3://b/x.optimized.onnx"},
    )
    assert out["version"] == "job-base-1-onnx-graph"
    assert captured["versions"][0]["metadata"]["framework"] == "onnx"
    assert captured["hubs"][0][1] == "onnx"

# -- device selection (CPU / Intel GPU) -------------------------------------

def _container(manifest):
    return manifest["spec"]["predictor"]["containers"][0]


def test_cpu_manifest_stays_cpu_only():
    m = _manifest("s3://nereidforge/x.onnx", gpu="cpu")
    c = _container(m)
    assert "gpu.intel.com/xe" not in c["resources"]["limits"]
    assert "--target_device GPU" not in c["command"][2]
    assert c.get("securityContext") is None
    assert "openvino-xpu" not in c["image"]


def test_legacy_zero_gpu_value_means_cpu():
    m = _manifest("s3://nereidforge/x.onnx", gpu="0")
    c = _container(m)
    assert "gpu.intel.com/xe" not in c["resources"]["limits"]
    assert c.get("securityContext") is None


def test_intel_gpu_manifest_wires_resource_device_and_image():
    m = _manifest("s3://nereidforge/x.onnx", gpu="intel")
    c = _container(m)
    assert c["resources"]["limits"]["gpu.intel.com/xe"] == "1"
    assert "--target_device GPU" in c["command"][2]
    assert "openvino-xpu" in c["image"]
    assert c["securityContext"] == {"runAsUser": 0, "runAsGroup": 0}


def test_intel_gpu_aliases_xpu_and_resource_string():
    for gpu in ("xpu", "gpu.intel.com/xe"):
        c = _container(_manifest("s3://nereidforge/x.onnx", gpu=gpu))
        assert c["resources"]["limits"].get("gpu.intel.com/xe") == "1"
        assert "--target_device GPU" in c["command"][2]


def test_intel_gpu_with_non_openvino_runtime_does_not_add_device_flag():
    m = _manifest("s3://nereidforge/x.onnx", runtime="kserve-tritonserver", gpu="intel")
    c = _container(m)
    assert c["resources"]["limits"].get("gpu.intel.com/xe") == "1"
    assert "--target_device" not in c["command"][2]
