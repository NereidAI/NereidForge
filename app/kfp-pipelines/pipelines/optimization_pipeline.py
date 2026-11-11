# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NereidForge Optimization Pipeline v2 (single-run multi-target DAG).

One KFP run per optimization batch:
  parse_targets   materializes the targets JSON into a loopable list
  baseline_step   evaluates the source model once (real mAP + latency)
  per target (parallel):
    optimize_step quantizes / graph-optimizes + benchmarks, uploads artifact
    gate_step     evaluates the optimized model against the shared baseline
                  metrics, writes the gate report, and FAILS the branch when
                  a hard gate is violated (artifact not registered; soft
                  gates only warn)

All per-target values (job_id, device, method, dtype, calib_n) are read at
runtime from the `target` dict — the graph never inspects loop items.
"""

from typing import List

from kfp import dsl

from images import nf_image

OPTIMIZER_IMAGE = nf_image("train/optimizer")
EVALUATOR_IMAGE = nf_image("train/evaluator")
# NOTE: lightweight components serialize ONLY the function body — module-level
# constants are invisible inside them; keep the endpoint literal inline.
S3_ENDPOINT = "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"


@dsl.component(base_image=EVALUATOR_IMAGE)
def parse_targets(
    targets_json: str,
    targets: dsl.OutputPath(List[dict]),
) -> None:
    """Parse the targets JSON string into a list parameter for ParallelFor."""
    import json

    with open(targets, "w") as f:
        json.dump(json.loads(targets_json or "[]"), f)


@dsl.component(base_image=EVALUATOR_IMAGE)
def baseline_step(
    model_uri: str,
    dataset_uri: str,
    output_uri: str,
    config_json: str,
    s3_access_key: str,
    s3_secret_key: str,
    metrics: dsl.OutputPath(str),
) -> None:
    """Evaluate the source model once; metrics JSON forwarded to every gate."""
    if not (s3_access_key or "").strip() or not (s3_secret_key or "").strip():
        raise ValueError("s3_access_key and s3_secret_key must be non-empty")

    S3_ENDPOINT = "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"

    import os
    import subprocess

    env = {
        "SCRIPT": "evaluate",
        "INPUT_URI": model_uri,
        "DATASET_URI": dataset_uri,
        "OUTPUT_URI": output_uri,
        "ARTIFACT_ID": "baseline",
        "CONFIG_JSON": config_json,
        "S3_ENDPOINT": S3_ENDPOINT,
        "S3_ACCESS_KEY": s3_access_key,
        "S3_SECRET_KEY": s3_secret_key,
    }
    r = subprocess.run(["python", "/app/evaluate.py"], env={**os.environ, **env})
    if r.returncode != 0:
        raise RuntimeError(f"baseline eval failed with code {r.returncode}")
    with open("/tmp/baseline.metrics.json") as src, open(metrics, "w") as dst:
        dst.write(src.read())


@dsl.component(base_image=OPTIMIZER_IMAGE)
def optimize_step(
    target: dict,
    model_uri: str,
    output_uri: str,
    config_json: str,
    s3_access_key: str,
    s3_secret_key: str,
    status: dsl.OutputPath(str),
    dataset_uri: str = "",
) -> None:
    """Quantize / graph-optimize the model and upload artifact + metrics."""
    if not (s3_access_key or "").strip() or not (s3_secret_key or "").strip():
        raise ValueError("s3_access_key and s3_secret_key must be non-empty")

    S3_ENDPOINT = "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"

    import json
    import os
    import subprocess

    artifact_id = str(target.get("job_id", ""))
    cfg = json.loads(config_json or "{}")
    opt_cfg = {
        "target_device": str(target.get("target_device", "cpu")),
        "quant_method": str(target.get("quant_method", "graph")),
        "dtype": str(target.get("dtype", "")),
        "calib_n": int(target.get("calib_n", 200) or 200),
        "val_n": int(target.get("val_n", 100) or 100),
        "max_map_drop_abs": target.get("max_map_drop_abs"),
        "imgsz": cfg.get("imgsz", 640),
        # int8-static calibration sampling draws from the evaluation dataset.
        "calib_uri": dataset_uri,
    }
    env = {
        "SCRIPT": "optimize",
        "INPUT_URI": model_uri,
        "OUTPUT_DIR": f"{output_uri.rstrip('/')}/{artifact_id}",
        "ARTIFACT_ID": artifact_id,
        "CONFIG_JSON": json.dumps(opt_cfg),
        "S3_ENDPOINT": S3_ENDPOINT,
        "S3_ACCESS_KEY": s3_access_key,
        "S3_SECRET_KEY": s3_secret_key,
    }
    r = subprocess.run(["python", "/app/optimize.py"], env={**os.environ, **env})
    if r.returncode != 0:
        raise RuntimeError(f"optimize step failed with code {r.returncode}")
    with open(status, "w") as f:
        f.write("ok")


@dsl.component(base_image=EVALUATOR_IMAGE)
def gate_step(
    target: dict,
    baseline_metrics_json: str,
    dataset_uri: str,
    output_uri: str,
    config_json: str,
    gates_json: str,
    s3_access_key: str,
    s3_secret_key: str,
    status: dsl.OutputPath(str),
) -> None:
    """Evaluate the optimized model against baseline metrics + gates.

    Exits non-zero on hard-gate failure so the branch (and thus the target
    job) fails; the report is still written for post-mortem/override.
    """
    if not (s3_access_key or "").strip() or not (s3_secret_key or "").strip():
        raise ValueError("s3_access_key and s3_secret_key must be non-empty")

    S3_ENDPOINT = "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"

    import json
    import os
    import subprocess

    artifact_id = str(target.get("job_id", ""))
    device = str(target.get("target_device", "cpu"))
    ext = ".xml" if device == "openvino" else ".onnx"
    optimized_uri = f"{output_uri.rstrip('/')}/{artifact_id}/{artifact_id}.optimized{ext}"
    report_uri = f"s3://nereidforge/evaluation-reports/{artifact_id}/report.json"
    cfg = json.loads(config_json or "{}")
    cfg["baseline_metrics"] = json.loads(baseline_metrics_json or "{}")
    cfg["gates"] = json.loads(gates_json or "{}")
    cfg["job_type"] = "optimization"
    env = {
        "SCRIPT": "evaluate",
        "INPUT_URI": optimized_uri,
        "DATASET_URI": dataset_uri,
        "REPORT_URI": report_uri,
        "ARTIFACT_ID": artifact_id,
        "CONFIG_JSON": json.dumps(cfg),
        "S3_ENDPOINT": S3_ENDPOINT,
        "S3_ACCESS_KEY": s3_access_key,
        "S3_SECRET_KEY": s3_secret_key,
    }
    r = subprocess.run(["python", "/app/evaluate.py"], env={**os.environ, **env})
    if r.returncode == 3:
        raise RuntimeError(
            "GATE_FAIL: hard gate violated (see gate report); artifact rejected"
        )
    if r.returncode != 0:
        raise RuntimeError(f"gate eval failed with code {r.returncode}")
    with open(status, "w") as f:
        f.write("ok")


def _res(step, cpu: str = "4", mem: str = "8Gi") -> None:
    """Standardized resource envelope so benchmarks stay comparable."""
    step.set_cpu_request(cpu)
    step.set_cpu_limit(cpu)
    step.set_memory_request(mem)
    step.set_memory_limit(mem)


@dsl.pipeline(
    name="nereidforge-optimization-pipeline",
    description="Quantize/optimize per target with baseline gate (multi-target DAG)",
    pipeline_root="s3://nereidforge-pipelines/{{workflow.uid}}",
)
def optimization_pipeline(
    s3_access_key: str,
    s3_secret_key: str,
    model_uri: str = "",
    dataset_uri: str = "",
    output_uri: str = "s3://nereidforge/optimized/",
    targets_json: str = "[]",
    gates_json: str = "{}",
    eval_config_json: str = "{}",
):
    # Baseline runs once for the whole batch (the point of the single DAG).
    base = baseline_step(
        model_uri=model_uri,
        dataset_uri=dataset_uri,
        output_uri=output_uri,
        config_json=eval_config_json,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    base.set_caching_options(False)
    _res(base)

    # Dynamic per-target loop: targets arrive as JSON (KFP params are
    # strings); a parse component materializes the list for ParallelFor.
    parsed = parse_targets(targets_json=targets_json)
    parsed.set_caching_options(False)

    with dsl.ParallelFor(items=parsed.outputs["targets"]) as tgt:
        opt = optimize_step(
            target=tgt,
            model_uri=model_uri,
            output_uri=output_uri,
            config_json=eval_config_json,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            dataset_uri=dataset_uri,
        )
        opt.set_caching_options(False)

        gate = gate_step(
            target=tgt,
            baseline_metrics_json=base.outputs["metrics"],
            dataset_uri=dataset_uri,
            output_uri=output_uri,
            config_json=eval_config_json,
            gates_json=gates_json,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
        )
        gate.set_caching_options(False)
        gate.after(opt)
