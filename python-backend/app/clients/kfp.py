# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Kubeflow Pipelines (KFP) v2 API client.

Mirrors the Go KFPService (internal/service/kfp.go):
- submits runs via the v2 REST API (/apis/v2beta1/runs)
- filters parameters through per-pipeline allowlists
- maps v2 run states to the legacy v1 names used across the app
- reads dynamic config (host/experiment/pipeline ids) from system_config
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from app.core.config import config, require_s3_credentials

# Per-pipeline allowed parameter keys (matches Go pipelineParams).
_PIPELINE_PARAMS: dict[str, set[str]] = {
    "training-pipeline": {
        "artifact_id", "checkpoint_output_dir", "common_config", "cpu_limit", "cpu_request",
        "dataset_name", "dataset_output_dir", "dataset_registry_url", "dataset_version",
        "event_bus_url", "event_output_dir", "export_formats", "export_output_dir",
        "gpu_limit", "gpu_request", "hub_api_url", "hub_token", "memory_limit", "memory_request",
        "model_name", "model_version", "namespace", "quantization", "register_output_dir",
        "seaweedfs_url", "train_config", "train_output_dir", "trainjob_name",
    },
    "optimization-pipeline": {
        "model_uri", "dataset_uri", "output_uri", "targets_json", "gates_json",
        "eval_config_json", "cpu_request", "cpu_limit", "memory_request",
        "memory_limit", "s3_access_key", "s3_secret_key",
    },
    "evaluation-pipeline": {
        "model_uri", "baseline_model_uri", "dataset_uri", "output_uri",
        "report_uri", "artifact_id", "config_json", "cpu_request", "cpu_limit",
        "memory_request", "memory_limit", "s3_access_key", "s3_secret_key",
    },
}

_V2_TO_V1 = {
    "SUCCEEDED": "Succeeded",
    "FAILED": "Failed",
    "CANCELED": "Cancelled",
    "CANCELLING": "Cancelled",
    "PENDING": "Pending",
    "SKIPPED": "Skipped",
}

_TERMINAL_STATES = {"Succeeded", "Failed", "Cancelled", "Error"}


def map_v2_state(state: str) -> str:
    """Map a KFP v2 run state to the legacy v1 naming used by the app."""
    if not state:
        return "Running"
    return _V2_TO_V1.get(state, "Running")


def pipeline_param_allowed(pipeline_name: str, key: str) -> bool:
    """Return True when the parameter is allowed for the given pipeline."""
    if not pipeline_name:
        return True
    allowed = _PIPELINE_PARAMS.get(pipeline_name)
    if allowed is None:
        return True  # unknown pipeline: do not filter
    return key in allowed


@dataclass
class PipelineRunSpec:
    """Parameters for submitting a pipeline run."""
    pipeline_name: str
    display_name: str
    parameters: dict[str, str] = field(default_factory=dict)
    pipeline_id: str = ""
    pipeline_version_id: str = ""
    experiment_id: str = ""
    user_id: str = ""


@dataclass
class PipelineRunResult:
    """Result of a submitted or queried pipeline run."""
    run_id: str
    status: str = "Pending"
    display_name: str = ""
    created_at: Optional[str] = None
    finished_at: Optional[str] = None
    parameters: dict[str, str] = field(default_factory=dict)
    # KFP task states drive live optimization milestones; no image-side
    # progress producer or timing estimate is required.
    tasks: list[dict[str, Any]] = field(default_factory=list)


class KFPClient:
    """KFP v2 REST client with dynamic config from system_config."""

    def __init__(self, config_fn: Any) -> None:
        self.config_fn = config_fn
        self.timeout = httpx.Timeout(config.kfp.timeout or 300)

    def _cfg(self) -> dict[str, str]:
        """Fetch the current KFP config (host/experiment/pipeline ids)."""
        return self.config_fn()

    def _latest_version(self, pipeline_id: str, userid: str) -> str:
        """Resolve the newest uploaded pipeline version id for a pipeline."""
        cfg = self._cfg()
        host = cfg.get("kfp_host") or config.kfp.host
        url = f"{host.rstrip('/')}/apis/v2beta1/pipelines/{pipeline_id}/versions?page_size=50"
        resp = httpx.get(
            url, headers={"kubeflow-userid": userid}, timeout=self.timeout,
        )
        if resp.status_code >= 400:
            return ""
        data = resp.json()
        versions = data.get("pipeline_versions") or []
        if not versions:
            return ""
        # Pick the newest by created_at (KFP may return versions oldest-first).
        best = ""
        best_ts = ""
        for v in versions:
            vid = v.get("pipeline_version_id") or ""
            ts = v.get("created_at") or ""
            if vid and ts >= best_ts:
                best, best_ts = vid, ts
        return best

    def submit_run(self, spec: PipelineRunSpec) -> PipelineRunResult:
        """Submit a pipeline run via the KFP v2 API."""
        cfg = self._cfg()
        host = cfg.get("kfp_host") or config.kfp.host
        if not host:
            raise ValueError("kfp host not configured")
        userid = spec.user_id or cfg.get("kfp_userid") or "admin@nereidai.local"
        experiment = spec.experiment_id or cfg.get("kfp_experiment_id") or ""

        pipeline_id = spec.pipeline_id or cfg.get(
            f"kfp_{spec.pipeline_name.replace('-pipeline', '').replace('-', '_')}_pipeline_id", ""
        )
        if not pipeline_id:
            raise ValueError(f"no pipeline id for name {spec.pipeline_name}")

        # Pin the run to the newest uploaded version. Without an explicit
        # version, KFP v2 falls back to the pipeline's default version which
        # may predate pipeline edits (e.g. missing cpu/memory parameters).
        version_id = spec.pipeline_version_id or self._latest_version(pipeline_id, userid)
        version_ref = (
            {"pipeline_id": pipeline_id, "pipeline_version_id": version_id}
            if version_id
            else {"pipeline_id": pipeline_id}
        )

        # v2 only accepts parameters declared by the pipeline; strip internal keys.
        params = {
            k: str(v)
            for k, v in spec.parameters.items()
            if pipeline_param_allowed(spec.pipeline_name, k)
        }

        body: dict[str, Any] = {
            "experiment_id": experiment,
            "display_name": spec.display_name,
            "pipeline_version_reference": version_ref,
            "runtime_config": {"parameters": params},
        }

        url = f"{host.rstrip('/')}/apis/v2beta1/runs"
        resp = httpx.post(
            url,
            json=body,
            headers={"Content-Type": "application/json", "kubeflow-userid": userid},
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"kfp submit error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return PipelineRunResult(
            run_id=data.get("run_id", ""),
            status=map_v2_state(data.get("state", "")),
            display_name=data.get("display_name", spec.display_name),
            created_at=data.get("created_at"),
            finished_at=data.get("finished_at"),
            parameters=params,
        )

    def get_run(self, run_id: str) -> PipelineRunResult:
        """Fetch a pipeline run's current state."""
        cfg = self._cfg()
        host = cfg.get("kfp_host") or config.kfp.host
        if not host:
            raise ValueError("kfp host not configured")
        userid = cfg.get("kfp_userid") or "admin@nereidai.local"
        url = f"{host.rstrip('/')}/apis/v2beta1/runs/{run_id}"
        resp = httpx.get(
            url,
            headers={"kubeflow-userid": userid},
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            raise ValueError(f"pipeline run {run_id} not found")
        if resp.status_code >= 400:
            raise RuntimeError(f"KFP API error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        # The v2 run endpoint returns the run object at the top level.
        run = data.get("run", data)
        return PipelineRunResult(
            run_id=run_id,
            status=map_v2_state(run.get("state", "")),
            display_name=run.get("display_name", ""),
            created_at=run.get("created_at"),
            finished_at=run.get("finished_at"),
            tasks=list((run.get("run_details") or {}).get("task_details") or []),
        )

    def cancel_run(self, run_id: str) -> None:
        """Terminate a pipeline run."""
        cfg = self._cfg()
        host = cfg.get("kfp_host") or config.kfp.host
        if not host:
            raise ValueError("kfp host not configured")
        userid = cfg.get("kfp_userid") or "admin@nereidai.local"
        url = f"{host.rstrip('/')}/apis/v2beta1/runs/{run_id}:terminate"
        resp = httpx.post(
            url,
            headers={"kubeflow-userid": userid},
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"kfp cancel failed: {resp.status_code}: {resp.text[:300]}")

    def get_run_logs(self, run_id: str, tail: int = 400) -> str:
        """Fetch a run's aggregated executor logs from the KFP object store.

        KFP v2 stores per-step executor logs under
        private-artifacts/{namespace}/v2/artifacts/{pipeline}/{run_id}/{step}/*/executor-logs-0.
        """
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError:
            return ""
        endpoint = config.kfp.object_store_endpoint
        if not endpoint:
            return ""
        access_key, secret_key = require_s3_credentials(
            config.kfp.object_store_access_key,
            config.kfp.object_store_secret_key,
            scope="kfp.object_store",
        )
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
            config=BotoConfig(connect_timeout=5, read_timeout=10, retries={"max_attempts": 1}),
        )
        bucket = config.kfp.object_store_bucket or "mlpipeline"
        ns = config.kfp.object_store_namespace or "kubeflow-user-example-com"
        prefix = f"private-artifacts/{ns}/v2/artifacts/"
        logs: list[str] = []
        try:
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []) or []:
                    key = obj["Key"]
                    if f"/{run_id}/" not in key or not key.endswith("executor-logs-0"):
                        continue
                    try:
                        body = client.get_object(Bucket=bucket, Key=key)["Body"].read().decode(errors="replace")
                        logs.append(f"===== {key.split('/')[-3]}/{key.split('/')[-4]} =====\n{body}")
                    except Exception:
                        continue
        except Exception:
            return "\n\n".join(logs)
        logs.sort()
        if tail and tail > 0:
            joined = "\n\n".join(logs)
            lines = joined.splitlines()
            return "\n".join(lines[-tail:])
        return "\n\n".join(logs)

    def is_terminal(self, status: str) -> bool:
        """Return True when the run status is terminal."""
        return status in _TERMINAL_STATES
