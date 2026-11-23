# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""K8s ML-Backend Deployer (via KFP cluster-op pipelines).

The NereidForge container has no cluster access, so custom ML-backend
Deployment + Service objects are applied by a KFP pipeline pod running with
its own service account (mirrors the old in-cluster provisioner).
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional

from app.clients import cluster_ops
from app.core.images import nf_image

logger = logging.getLogger(__name__)

NAMESPACE = "nereidforge"
IMAGE_BASE = nf_image("ml-backend/ml-backend-custom")
S3_ENDPOINT = "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"
S3_CREDENTIALS_SECRET = "seaweedfs-s3-credentials"
LABEL_STUDIO_HOST = "http://labelstudio.nereidforge.svc.cluster.local:8080"


def generate_service_name(model_name: str, version: str) -> str:
    """
    Generate K8s Service name (max 63 characters).

    Rule: {model_name}-{version}-ml-backend
    Example: ppyoloe-seg-v1.0-ml-backend
    """
    name = f"{model_name}-{version}-ml-backend"

    # If exceeds 63 chars, truncate model_name and add hash
    if len(name) > 63:
        hash_suffix = hashlib.md5(f"{model_name}-{version}".encode()).hexdigest()[:8]
        name = f"{model_name[:50]}-{hash_suffix}-ml-backend"

    return name[:63]


def _deployment_manifest(
    service_name: str,
    model_name: str,
    version: str,
    s3_path: str,
    backend_type: str,
    framework: str,
) -> dict:
    """Build the Deployment manifest for a custom ML-backend service."""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": service_name,
            "namespace": NAMESPACE,
            "labels": {
                "app": "ml-backend",
                "type": "custom",
                "model": model_name,
                "version": version,
            },
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "ml-backend", "service": service_name}},
            "template": {
                "metadata": {"labels": {"app": "ml-backend", "service": service_name}},
                "spec": {
                    "containers": [
                        {
                            "name": "ml-backend",
                            "image": IMAGE_BASE,
                            "ports": [{"containerPort": 9090}],
                            "env": [
                                {"name": "MODEL_PATH", "value": s3_path},
                                {"name": "BACKEND_TYPE", "value": backend_type},
                                {"name": "FRAMEWORK", "value": framework},
                                {"name": "S3_ENDPOINT", "value": S3_ENDPOINT},
                                {
                                    "name": "S3_ACCESS_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": S3_CREDENTIALS_SECRET,
                                            "key": "accessKey",
                                            "optional": False,
                                        }
                                    },
                                },
                                {
                                    "name": "S3_SECRET_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": S3_CREDENTIALS_SECRET,
                                            "key": "secretKey",
                                            "optional": False,
                                        }
                                    },
                                },
                                {"name": "LABEL_STUDIO_HOST", "value": LABEL_STUDIO_HOST},
                                {
                                    "name": "LABEL_STUDIO_API_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "ml-backend-ls-token",
                                            "key": "token",
                                        }
                                    },
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "512Mi"},
                                "limits": {"cpu": "1000m", "memory": "2Gi"},
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 9090},
                                "initialDelaySeconds": 120,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 9090},
                                "initialDelaySeconds": 120,
                                "periodSeconds": 20,
                                "timeoutSeconds": 5,
                            },
                        }
                    ],
                    "imagePullSecrets": [{"name": "harbor-registry-secret"}],
                },
            },
        },
    }


def _service_manifest(service_name: str, model_name: str, version: str) -> dict:
    """Build the Service manifest for a custom ML-backend."""
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": service_name,
            "namespace": NAMESPACE,
            "labels": {"app": "ml-backend", "type": "custom"},
        },
        "spec": {
            "type": "ClusterIP",
            "ports": [{"port": 9090, "targetPort": 9090, "protocol": "TCP", "name": "http"}],
            "selector": {"app": "ml-backend", "service": service_name},
        },
    }


def create_ml_backend_deployment(
    model_name: str,
    version: str,
    s3_path: str,
    backend_type: str,
    framework: str,
    service_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create K8s Deployment + Service for a custom ML-Backend via KFP.

    Args:
        model_name: Model name
        version: Model version
        s3_path: S3 path to model (s3://bucket/key)
        backend_type: Backend type (det/seg/ocr/etc)
        framework: ML framework (paddlex/pytorch/etc)
        service_name: K8s Service name (auto-generated if not provided)

    Returns:
        Dict with deployment and service info
    """
    if not service_name:
        service_name = generate_service_name(model_name, version)

    manifests = [
        _deployment_manifest(service_name, model_name, version, s3_path, backend_type, framework),
        _service_manifest(service_name, model_name, version),
    ]
    logger.info("custom ml-backend deploy via KFP: %s (model=%s:%s)", service_name, model_name, version)
    run_id = cluster_ops.submit_pipeline(
        "cluster_deploy_pipeline",
        params={"manifests_json": json.dumps(manifests), "wait_isvc_name": "", "timeout_minutes": 20},
        run_name=f"mlb-deploy-{service_name}",
        namespace="nereidforge",
        wait=False,
    )
    return {
        "service_name": service_name,
        "url": f"http://{service_name}.{NAMESPACE}.svc.cluster.local:9090",
        "status": "created",
        "kfp_run_id": run_id,
    }


def delete_ml_backend_deployment(service_name: str) -> bool:
    """
    Delete K8s Deployment + Service for a custom ML-Backend via KFP.

    Args:
        service_name: K8s Service name

    Returns:
        True after the undeploy run is submitted
    """
    manifests = [
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": service_name, "namespace": NAMESPACE},
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": service_name, "namespace": NAMESPACE},
        },
    ]
    logger.info("custom ml-backend undeploy via KFP: %s", service_name)
    cluster_ops.submit_pipeline(
        "cluster_undeploy_pipeline",
        params={"manifests_json": json.dumps(manifests)},
        run_name=f"mlb-undeploy-{service_name}",
        namespace="nereidforge",
        wait=True,
        timeout=300,
        poll_interval=15,
    )
    return True


def get_ml_backend_status(service_name: str) -> Dict[str, Any]:
    """
    Return status for a custom ML-Backend service (degraded, no cluster reads).

    Live readiness/health is tracked by the NF health checker against the
    service URL; the NereidForge container cannot query the cluster.
    """
    return {
        "service_name": service_name,
        "url": f"http://{service_name}.{NAMESPACE}.svc.cluster.local:9090",
        "status": "unknown",
        "note": "cluster status unavailable from NereidForge (use the KFP run / cluster)",
    }
