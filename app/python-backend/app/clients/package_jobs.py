# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes Batch Job adapter for the single-container edge packager."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.core.config import config, require_s3_credentials
from app.core.images import nf_image
from app.services.package_contract import EDGE_VERIFIED, ONNX_STRUCTURAL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PackageJobState:
    status: str  # pending | running | succeeded | failed
    reason: str = ""
    message: str = ""
    log_tail: str = ""


def _sanitize(name: str) -> str:
    """Reduce an arbitrary name to a RFC-1123-safe Job name fragment."""
    return re.sub(r"[^a-z0-9.-]+", "-", name.lower()).strip(".-")[:53] or "package"


class PackageJobRunner:
    """Tiny submit/state facade; all package logic stays in the container."""

    # IR retains the edge-verified CPU/GPU profile. Direct ONNX is structural
    # and runs on CPU without reserving a device it never executes on.
    PROFILES = {
        EDGE_VERIFIED: {
            "image": nf_image("train/evaluator-xpu"),
            # The validated XPU base uses a venv rather than Debian system Python.
            "command": ["/opt/xpu/bin/python", "/app/model_package.py"],
            "resources": {
                "requests": {"cpu": "2", "memory": "4Gi", "gpu.intel.com/xe": "1"},
                "limits": {"cpu": "2", "memory": "4Gi", "gpu.intel.com/xe": "1"},
            },
            # NEO enumerates zero Arc GPUs for the image's non-root uid (pit 174).
            "security_context": {"runAsUser": 0, "runAsGroup": 0},
        },
        ONNX_STRUCTURAL: {
            "image": nf_image("train/evaluator"),
            "command": ["python", "/app/model_package.py"],
            "resources": {
                "requests": {"cpu": "2", "memory": "4Gi"},
                "limits": {"cpu": "2", "memory": "4Gi"},
            },
            "security_context": None,
        },
    }

    def __init__(
        self,
        namespace: str = "nereidforge",
        *,
        batch_api: Any = None,
        core_api: Any = None,
        s3_endpoint: str = "",
        s3_access_key: str = "",
        s3_secret_key: str = "",
    ) -> None:
        self.namespace = namespace
        self._batch_api = batch_api
        self._core_api = core_api
        self.s3_endpoint = s3_endpoint or (
            config.storage.s3_cluster_endpoint
            or config.storage.s3_endpoint
            or "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"
        )
        self.s3_access_key, self.s3_secret_key = require_s3_credentials(
            s3_access_key or config.storage.s3_access_key,
            s3_secret_key or config.storage.s3_secret_key,
            scope="storage.s3",
        )

    def _apis(self) -> tuple[Any, Any]:
        if self._batch_api is not None and self._core_api is not None:
            return self._batch_api, self._core_api
        from kubernetes import client as kclient
        from kubernetes import config as kconfig

        try:
            kconfig.load_incluster_config()
        except Exception:
            kconfig.load_kube_config()
        if self._batch_api is None:
            self._batch_api = kclient.BatchV1Api()
        if self._core_api is None:
            self._core_api = kclient.CoreV1Api()
        return self._batch_api, self._core_api

    def job_manifest(
        self, spec_uri: str, profile: str, job_name: str, request_id: str = ""
    ) -> dict[str, Any]:
        if not spec_uri.startswith("s3://"):
            raise ValueError("spec_uri must be an s3:// URI")
        if profile not in self.PROFILES:
            raise ValueError(f"unknown package hardware profile {profile!r}")
        profile_cfg = self.PROFILES[profile]
        name = _sanitize(job_name)
        container: dict[str, Any] = {
            "name": "packager",
            "image": profile_cfg["image"],
            "imagePullPolicy": config.image_pull_policy if hasattr(config, "image_pull_policy") else "Always",
            "command": profile_cfg["command"],
            "args": ["--spec-uri", spec_uri],
            "env": [
                {"name": "S3_ENDPOINT", "value": self.s3_endpoint},
                {"name": "S3_ACCESS_KEY", "value": self.s3_access_key},
                {"name": "S3_SECRET_KEY", "value": self.s3_secret_key},
            ],
            "resources": profile_cfg["resources"],
        }
        if profile_cfg["security_context"]:
            container["securityContext"] = profile_cfg["security_context"]
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": name,
                "namespace": self.namespace,
                "labels": {
                    "app.kubernetes.io/name": "nereidforge-packager",
                    "app.kubernetes.io/managed-by": "nereidforge-worker",
                    "nereidforge/hardware-profile": profile,
                    "nereidforge/request-id": _sanitize(request_id) if request_id else name,
                },
            },
            "spec": {
                "backoffLimit": 0,
                "completions": 1,
                "parallelism": 1,
                "activeDeadlineSeconds": 2400,
                "ttlSecondsAfterFinished": 3600,
                "template": {
                    "metadata": {
                        "labels": {
                            "app.kubernetes.io/name": "nereidforge-packager",
                            "app.kubernetes.io/managed-by": "nereidforge-worker",
                            "nereidforge/hardware-profile": profile,
                            "job-name": name,
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [container],
                    },
                },
            },
        }

    def submit(self, spec_uri: str, profile: str, *, job_name: str) -> str:
        body = self.job_manifest(spec_uri, profile, job_name)
        batch, _ = self._apis()
        obj = batch.create_namespaced_job(self.namespace, body)
        name = str(obj.metadata.name)
        logger.info("submitted package job %s/%s for %s", self.namespace, name, profile)
        return name

    def delete(self, job_name: str) -> None:
        """Delete a job and its pod; used when the async task is cancelled."""
        batch, _ = self._apis()
        try:
            batch.delete_namespaced_job(
                job_name,
                self.namespace,
                propagation_policy="Background",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to delete package job %s: %s", job_name, e)

    def get_state(self, job_name: str) -> PackageJobState:
        """Map Job/pod conditions to pending|running|succeeded|failed."""
        batch, core = self._apis()
        job = batch.read_namespaced_job(job_name, self.namespace)
        status = job.status or {}
        if getattr(status, "succeeded", 0):
            return PackageJobState("succeeded")
        if getattr(status, "active", 0):
            return PackageJobState("running")
        if getattr(status, "failed", 0):
            return self._failure_state(core, job_name)
        return PackageJobState("pending")

    def _failure_state(self, core: Any, job_name: str) -> PackageJobState:
        """Collect reason/message and the last 100 pod log lines on failure."""
        reason, message, log_tail = "JobFailed", "package job failed", ""
        try:
            pods = core.list_namespaced_pod(
                self.namespace, label_selector=f"job-name={job_name}"
            ).items
            for pod in pods:
                statuses = list(getattr(pod.status, "container_statuses", None) or [])
                for cs in statuses:
                    terminated = getattr(cs.state, "terminated", None)
                    if terminated:
                        reason = str(terminated.reason or reason)
                        message = str(terminated.message or message)
                        try:
                            log_tail = core.read_namespaced_pod_log(
                                name=pod.metadata.name,
                                namespace=self.namespace,
                                container=cs.name,
                                tail_lines=100,
                            )
                        except Exception as e:  # noqa: BLE001
                            log_tail = f"unavailable logs: {e}"
        except Exception as e:  # noqa: BLE001
            message = f"{message}; pod inspection failed: {e}"
        return PackageJobState("failed", reason, message, log_tail[-2000:])
