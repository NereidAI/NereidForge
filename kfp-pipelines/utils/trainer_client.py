# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Trainer Client

Client for kubeflow.trainer v1alpha1 TrainJob CRUD operations.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class TrainJobSpec(BaseModel):
    trainer: str = "pytorch"
    replicas: int = 1
    template: Dict[str, Any] = Field(default_factory=dict)


class TrainJobStatus(BaseModel):
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    model_uri: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)
    failure_reason: str = ""


class TrainerClient:
    """Client for kubeflow.trainer TrainJob CRD."""

    GROUP = "trainer.kubeflow.org"
    VERSION = "v1alpha1"
    PLURAL = "trainjobs"

    def __init__(
        self,
        namespace: str = "nereidforge",
        kubeconfig: Optional[str] = None,
    ):
        """
        Initialize Trainer client.

        Args:
            namespace: Kubernetes namespace
            kubeconfig: Path to kubeconfig (None for in-cluster)
        """
        self.namespace = namespace

        if kubeconfig:
            config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()

        self.custom_api = client.CustomObjectsApi()
        self.core_api = client.CoreV1Api()

    def _build_trainjob(
        self,
        name: str,
        spec: TrainJobSpec,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Build TrainJob resource dict."""
        job = {
            "apiVersion": f"{self.GROUP}/{self.VERSION}",
            "kind": "TrainJob",
            "metadata": {
                "name": name,
                "namespace": self.namespace,
                "labels": labels or {},
                "annotations": annotations or {},
            },
            "spec": spec.model_dump(exclude_none=True),
        }
        return job

    def create_trainjob(
        self,
        name: str,
        spec: TrainJobSpec,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a TrainJob."""
        job = self._build_trainjob(name, spec, labels, annotations)
        try:
            result = self.custom_api.create_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=self.namespace,
                plural=self.PLURAL,
                body=job,
            )
            logger.info(f"TrainJob {name} created")
            return result
        except ApiException as e:
            if e.status == 409:
                logger.info(f"TrainJob {name} already exists")
                return self.get_trainjob(name)
            raise

    def get_trainjob(self, name: str) -> Dict[str, Any]:
        """Get TrainJob by name."""
        return self.custom_api.get_namespaced_custom_object(
            group=self.GROUP,
            version=self.VERSION,
            namespace=self.namespace,
            plural=self.PLURAL,
            name=name,
        )

    def delete_trainjob(self, name: str) -> None:
        """Delete a TrainJob."""
        self.custom_api.delete_namespaced_custom_object(
            group=self.GROUP,
            version=self.VERSION,
            namespace=self.namespace,
            plural=self.PLURAL,
            name=name,
        )
        logger.info(f"TrainJob {name} deleted")

    def list_trainjobs(
        self,
        label_selector: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List TrainJobs."""
        result = self.custom_api.list_namespaced_custom_object(
            group=self.GROUP,
            version=self.VERSION,
            namespace=self.namespace,
            plural=self.PLURAL,
            label_selector=label_selector,
            limit=limit,
        )
        return result.get("items", [])

    def patch_trainjob(self, name: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Patch a TrainJob."""
        return self.custom_api.patch_namespaced_custom_object(
            group=self.GROUP,
            version=self.VERSION,
            namespace=self.namespace,
            plural=self.PLURAL,
            name=name,
            body=patch,
        )

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def watch_trainjob(
        self,
        name: str,
        timeout_seconds: int = 86400,
        poll_interval: int = 30,
    ) -> Dict[str, Any]:
        """
        Watch TrainJob until completion.

        Returns final TrainJob with status.
        Raises TimeoutError if timeout exceeded.
        Raises RuntimeError if job fails.
        """
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                trainjob = self.get_trainjob(name)
                status = trainjob.get("status", {})
                conditions = status.get("conditions", [])

                # Check for terminal conditions
                for cond in conditions:
                    if cond.get("type") == "Succeeded" and cond.get("status") == "True":
                        logger.info(f"TrainJob {name} succeeded")
                        return trainjob
                    elif cond.get("type") == "Failed" and cond.get("status") == "True":
                        reason = status.get("failureReason", "Unknown")
                        logger.error(f"TrainJob {name} failed: {reason}")
                        raise RuntimeError(f"TrainJob failed: {reason}")

                logger.debug(f"TrainJob {name} still running...")

            except ApiException as e:
                if e.status == 404:
                    logger.warning(f"TrainJob {name} not found")
                    raise
                logger.error(f"Error watching TrainJob: {e}")
                raise

            time.sleep(poll_interval)

        raise TimeoutError(f"TrainJob {name} timed out after {timeout_seconds} seconds")

    def get_trainjob_logs(
        self,
        name: str,
        container: str = "trainer",
        tail_lines: int = 100,
    ) -> str:
        """Get logs from TrainJob pods."""
        # Find pods for this TrainJob
        pods = self.core_api.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"trainjob-name={name}",
        )

        logs = []
        for pod in pods.items:
            try:
                pod_logs = self.core_api.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=self.namespace,
                    container=container,
                    tail_lines=tail_lines,
                )
                logs.append(f"--- {pod.metadata.name} ---\n{pod_logs}")
            except ApiException as e:
                logger.warning(f"Could not get logs for pod {pod.metadata.name}: {e}")

        return "\n\n".join(logs)

    def wait_for_trainjob(
        self,
        name: str,
        timeout_seconds: int = 86400,
        poll_interval: int = 30,
    ) -> Dict[str, Any]:
        """Alias for watch_trainjob."""
        return self.watch_trainjob(name, timeout_seconds, poll_interval)


def create_trainer_client_from_env() -> TrainerClient:
    """Create Trainer client from environment variables."""
    import os

    namespace = os.getenv("TRAINER_NAMESPACE", "nereidforge")
    kubeconfig = os.getenv("KUBECONFIG")
    return TrainerClient(namespace=namespace, kubeconfig=kubeconfig)
