# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Delete Manifests Step (in-cluster)

Runs inside a KFP pipeline pod. Deletes a JSON array of Kubernetes manifests
using the pod's service account. Missing objects / missing resource types are
tolerated so undeploy is idempotent.
"""

import json
import logging

from kfp import dsl

logger = logging.getLogger(__name__)

from images import nf_image

# nf-runtime preinstalls kubernetes/boto3 + NJU pip index (see apply step).
BASE_IMAGE = nf_image("runtime")
PACKAGES = ["kubernetes>=28.1,<32.0", "boto3>=1.34.0", "botocore>=1.34.0"]


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def delete_manifests(manifests_json: str) -> str:
    """Delete a JSON array of Kubernetes manifests; missing = ok."""
    import json
    import logging
    logger = logging.getLogger(__name__)
    from kubernetes import client, config
    from kubernetes.dynamic import DynamicClient
    from kubernetes.dynamic.exceptions import NotFoundError, ResourceNotFoundError

    config.load_incluster_config()
    dyn = DynamicClient(client.ApiClient())

    manifests = json.loads(manifests_json or "[]")
    if not isinstance(manifests, list):
        raise ValueError("manifests_json must be a JSON array")

    deleted = 0
    for obj in manifests:
        name = obj["metadata"]["name"]
        namespace = obj.get("metadata", {}).get("namespace")
        try:
            res = dyn.resources.get(kind=obj["kind"], api_version=obj["apiVersion"])
        except (ResourceNotFoundError, Exception):  # noqa: BLE001
            # Unknown resource type (e.g. no IngressRoute CRD) is not fatal.
            logger.warning("delete of %s/%s skipped: resource type missing", obj["kind"], name)
            continue
        try:
            res.get(name=name, namespace=namespace)
        except (NotFoundError, Exception):  # noqa: BLE001
            logger.info("no %s/%s to delete", obj["kind"], name)
            continue
        try:
            res.delete(name=name, namespace=namespace)
            deleted += 1
            logger.info("deleted %s/%s in %s", obj["kind"], name, namespace)
        except Exception as e:  # noqa: BLE001
            logger.warning("delete %s/%s failed: %s", obj["kind"], name, e)
    return f"deleted:{deleted}"
