# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Run Katib Experiment Step (in-cluster)

Runs inside a KFP pipeline pod. Replaces the old "NF kubectl-applies a Katib
Experiment" flow: the pipeline pod creates the Experiment, waits for it to
reach a terminal state, resolves the best succeeded trial, and writes
best_trial.json to S3 for the worker to consume. Trial scoring prefers the
S3 .trial.json uploaded by search_trial.py (authoritative), then the S3
metrics.json, then Katib observation metrics.
"""

import logging

from images import nf_image
from kfp import dsl

logger = logging.getLogger(__name__)

# nf-runtime preinstalls kubernetes/boto3 + NJU pip index (see apply step).
BASE_IMAGE = nf_image("runtime")
PACKAGES = ["kubernetes>=28.1,<32.0", "boto3>=1.34.0", "botocore>=1.34.0"]


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def run_katib_experiment(
    experiment_json: str,
    s3_access_key: str,
    s3_secret_key: str,
    job_id: str = "",
    best_trial_s3_uri: str = "s3://nereidforge/optimized/best_trial.json",
    s3_endpoint: str = "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333",
    timeout_minutes: int = 360,
) -> str:
    """Create + watch a Katib Experiment and record the best trial."""
    if not (s3_access_key or "").strip() or not (s3_secret_key or "").strip():
        raise ValueError("s3_access_key and s3_secret_key must be non-empty")

    import json
    import logging
    import time

    import boto3
    from botocore.client import Config
    from kubernetes import client, config
    from kubernetes.dynamic import DynamicClient
    from kubernetes.dynamic.exceptions import ResourceNotFoundError

    logger = logging.getLogger(__name__)

    # KFP serializes only this body, so terminal states must not come from a
    # module-level constant that is absent in the executor.
    terminal_states = {"Succeeded", "Failed", "Completed", "Completed (Capped)"}

    config.load_incluster_config()
    dyn = DynamicClient(client.ApiClient())

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint or None,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )

    def resource(kind: str, api_version: str) -> object:
        return dyn.resources.get(kind=kind, api_version=api_version)

    experiment = json.loads(experiment_json or "{}")
    name = experiment.get("metadata", {}).get("name") or ""
    namespace = experiment.get("metadata", {}).get("namespace") or "nereidforge"
    if not name:
        raise ValueError("experiment_json missing metadata.name")

    res = resource("Experiment", "kubeflow.org/v1beta1")
    try:
        res.get(name=name, namespace=namespace)
        res.replace(experiment, name=name, namespace=namespace)
    except ResourceNotFoundError:
        res.create(body=experiment, namespace=namespace)
    logger.info("experiment %s created in %s", name, namespace)

    end = time.time() + int(timeout_minutes) * 60
    phase = ""
    while time.time() < end:
        item = res.get(name=name, namespace=namespace).to_dict()
        conds = (item.get("status") or {}).get("conditions") or []
        phase = next(
            (c.get("type", "") for c in conds
             if c.get("type") in terminal_states and c.get("status") == "True"),
            "",
        )
        if phase:
            break
        time.sleep(30)

    if phase == "Failed":
        raise RuntimeError(f"katib experiment {name} failed")
    if not phase:
        raise RuntimeError(f"katib experiment {name} did not finish within {timeout_minutes} min")

    tres = resource("Trial", "kubeflow.org/v1beta1")
    trials = tres.list(
        namespace=namespace,
        label_selector=f"katib.kubeflow.org/experiment={name}",
    ).to_dict().get("items") or []

    def score(trial: dict) -> float:
        """Best latency score for a trial (None when no valid metric)."""
        tname = trial.get("metadata", {}).get("name", "")
        if job_id:
            for suffix, latch in (
                (f"{tname}.trial.json", "trial"),
                (f"{tname}.metrics.json", "metrics"),
            ):
                key = f"optimized/{job_id}/{tname}/{suffix}"
                try:
                    data = json.loads(
                        s3.get_object(Bucket="nereidforge", Key=key)["Body"].read()
                    )
                    lat = data.get("latency_ms") if latch == "trial" else (
                        data.get("avg_latency_ms") or data.get("latency_ms")
                    )
                    if isinstance(lat, (int, float)) and lat > 0:
                        return float(lat)
                except Exception:  # noqa: BLE001
                    continue
        status = trial.get("status") or {}
        conds = status.get("conditions") or []
        if not any(
            c.get("type") == "Succeeded" and c.get("status") == "True" for c in conds
        ):
            return float("inf")
        obs = status.get("observation") or {}
        for m in obs.get("metrics") or []:
            if m.get("name") == "latency_ms":
                try:
                    return float(m.get("value"))
                except (TypeError, ValueError):
                    continue
        return float("inf")

    best = ""
    best_score = float("inf")
    for t in trials:
        cand = score(t)
        if cand < best_score:
            best_score = cand
            best = t.get("metadata", {}).get("name", "")

    if not best or best_score == float("inf"):
        raise RuntimeError(f"katib experiment {name} has no valid trial result")

    out = {"experiment": name, "best_trial": best, "latency_ms": best_score}
    parsed = best_trial_s3_uri.replace("s3://", "", 1).split("/", 1)
    bkt = parsed[0] or "nereidforge"
    key = parsed[1] if len(parsed) > 1 else "best_trial.json"
    s3.put_object(Bucket=bkt, Key=key, Body=json.dumps(out).encode())
    logger.info("best trial %s (latency %s) -> %s", best, best_score, best_trial_s3_uri)
    return best
