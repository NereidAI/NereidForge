# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NereidForge Evaluation Pipeline v2.

A single step running the evaluator script: real detection metrics
(NMS + IoU + mAP@50/@50:95) or classification metrics, with optional
baseline model evaluated in-process for delta reporting. No gates here —
standalone evaluation is a report, not a blocker.
"""

from kfp import dsl

from images import nf_image

EVALUATOR_IMAGE = nf_image("train/evaluator")
# Lightweight components serialize only the function body; the
# endpoint literal must live inside the component.
S3_ENDPOINT = "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"


@dsl.component(base_image=EVALUATOR_IMAGE)
def evaluate_step(
    model_uri: str,
    baseline_model_uri: str,
    dataset_uri: str,
    output_uri: str,
    report_uri: str,
    artifact_id: str,
    config_json: str,
    s3_access_key: str,
    s3_secret_key: str,
    status_output_path: dsl.OutputPath(str),
) -> None:
    """Run the model evaluation script (real metrics, optional baseline)."""
    if not (s3_access_key or "").strip() or not (s3_secret_key or "").strip():
        raise ValueError("s3_access_key and s3_secret_key must be non-empty")

    S3_ENDPOINT = "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"

    import os
    import subprocess

    env = {
        "SCRIPT": "evaluate",
        "INPUT_URI": model_uri,
        "BASELINE_URI": baseline_model_uri,
        "DATASET_URI": dataset_uri,
        "OUTPUT_URI": output_uri,
        "REPORT_URI": report_uri,
        "ARTIFACT_ID": artifact_id,
        "CONFIG_JSON": config_json,
        "S3_ENDPOINT": S3_ENDPOINT,
        "S3_ACCESS_KEY": s3_access_key,
        "S3_SECRET_KEY": s3_secret_key,
    }
    r = subprocess.run(["python", "/app/evaluate.py"], env={**os.environ, **env})
    if r.returncode != 0:
        raise RuntimeError(f"evaluate step failed with code {r.returncode}")
    with open(status_output_path, "w") as f:
        f.write("ok")


@dsl.pipeline(
    name="nereidforge-evaluation-pipeline",
    description="Real model evaluation (detection/classification, optional baseline delta)",
    pipeline_root="s3://nereidforge-pipelines/{{workflow.uid}}",
)
def evaluation_pipeline(
    s3_access_key: str,
    s3_secret_key: str,
    model_uri: str = "",
    baseline_model_uri: str = "",
    dataset_uri: str = "",
    output_uri: str = "s3://nereidforge/eval/",
    report_uri: str = "",
    artifact_id: str = "{{workflow.uid}}",
    config_json: str = "{}",
    cpu_request: str = "4",
    cpu_limit: str = "4",
    memory_request: str = "8Gi",
    memory_limit: str = "8Gi",
):
    ev = evaluate_step(
        model_uri=model_uri,
        baseline_model_uri=baseline_model_uri,
        dataset_uri=dataset_uri,
        output_uri=output_uri,
        report_uri=report_uri,
        artifact_id=artifact_id,
        config_json=config_json,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    ev.set_caching_options(False)
    ev.set_cpu_request(cpu_request)
    ev.set_cpu_limit(cpu_limit)
    ev.set_memory_request(memory_request)
    ev.set_memory_limit(memory_limit)
