# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Apply Manifests Step (in-cluster)

Runs inside a KFP pipeline pod. Applies a JSON array of Kubernetes manifests
using the pod's service account. NereidForge never talks to the cluster
directly: it only submits these pipelines through the KFP API, and whatever
the manifests need (RBAC, CRDs, namespaces) is provisioned by ops beforehand.
"""

import json
import logging
import time

from kfp import dsl

logger = logging.getLogger(__name__)

from images import nf_image

# nf-runtime already ships kubernetes/boto3 and the NJU pip index: the
# component must not pip-install from files.pythonhosted.org at run time
# (times out on CN/offline clusters). Keep packages_to_install as a no-op
# version assertion.
BASE_IMAGE = nf_image("runtime")
PACKAGES = ["kubernetes>=28.1,<32.0", "boto3>=1.34.0", "botocore>=1.34.0"]


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def apply_manifests(
    manifests_json: str,
    wait_isvc_name: str = "",
    wait_isvc_namespace: str = "nereidforge",
    timeout_minutes: int = 30,
) -> str:
    """Apply manifests, optionally waiting for an InferenceService to be Ready."""
    import json
    import logging
    logger = logging.getLogger(__name__)
    import time
    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException
    from kubernetes.dynamic import DynamicClient
    from kubernetes.dynamic.exceptions import (
        NotFoundError,
        ResourceNotFoundError,
    )

    config.load_incluster_config()
    dyn = DynamicClient(client.ApiClient())

    def resource(kind: str, api_version: str) -> object:
        return dyn.resources.get(kind=kind, api_version=api_version)

    def apply_one(obj: dict) -> None:
        name = obj["metadata"]["name"]
        namespace = obj.get("metadata", {}).get("namespace")
        res = resource(obj["kind"], obj["apiVersion"])
        try:
            if namespace:
                res.create(body=obj, namespace=namespace)
            else:
                res.create(body=obj)
            logger.info("created %s/%s in %s", obj["kind"], name, namespace)
        except ApiException as e:
            if e.status != 409:
                raise
            res.patch(
                body=obj,
                name=name,
                namespace=namespace,
                content_type="application/merge-patch+json",
            )
            logger.info("patched %s/%s in %s", obj["kind"], name, namespace)

    manifests = json.loads(manifests_json or "[]")
    if not isinstance(manifests, list):
        raise ValueError("manifests_json must be a JSON array")
    for obj in manifests:
        apply_one(obj)

    def _active_revision(obj: dict) -> str:
        """latestRevision traffic target; first weighted target as fallback."""
        status = obj.get("status") or {}
        traffic = (
            (status.get("components") or {}).get("predictor", {}).get("traffic")
            or status.get("traffic") or []
        )
        weighted = ""
        for t in traffic:
            if t.get("latestRevision"):
                return t.get("revisionName") or ""
            if t.get("percent", 0) > 0 and not weighted:
                weighted = t.get("revisionName") or weighted
        return weighted

    def _find_private_service(revision_name: str, namespace: str) -> str:
        """The revision's private Knative ServerlessService (label lookup)."""
        svc_res = resource("Service", "v1")
        selector = (
            f"serving.knative.dev/revision={revision_name},"
            "networking.internal.knative.dev/serviceType=Private"
        )
        resp = svc_res.get(namespace=namespace, label_selector=selector)
        items = (resp.to_dict() if hasattr(resp, "to_dict") else {}).get(
            "items", []
        )
        if items:
            return items[0]["metadata"]["name"]
        # Legacy Knative releases did name it <revision>-private.
        legacy = f"{revision_name}-private"
        try:
            svc_res.get(name=legacy, namespace=namespace)
            return legacy
        except (ResourceNotFoundError, NotFoundError):
            return ""

    def _private_service_has_endpoints(svc_name: str, namespace: str) -> bool:
        """True once the private SKS service lists ready pod addresses."""
        try:
            ep = resource("Endpoints", "v1").get(
                name=svc_name, namespace=namespace,
            ).to_dict()
        except (ResourceNotFoundError, NotFoundError):
            return False
        for subset in ep.get("subsets") or []:
            if subset.get("addresses"):
                return True
        return False

    def reconcile_ingress_targets(isvc_name: str, ns: str, traffic: list,
                                  target: str = "") -> int:
        """Point applied IngressRoutes at the active revision service.

        Every KServe redeploy creates a new Knative revision
        (<name>-predictor-0000N) backed by a private Knative ServerlessService,
        while an IngressRoute manifest can only be generated before the
        revision exists (so it carries the deterministic 00001 target). After
        readiness we read the traffic split and repoint every IngressRoute
        named after the InferenceService to the current active revision's
        private service, so redeploys never keep serving the stale revision.

        The private service must be located by labels
        (serving.knative.dev/revision + serviceType=Private): current Knative
        names it after the configuration template hash
        (<prefix>-<template-hash>-private), not <revision-name>-private.

        ``target`` is passed when the caller has already verified the active
        revision (generation match + Ready + backend endpoints); otherwise it
        is resolved here from the traffic split.
        """
        revision = ""
        for t in traffic or []:
            if t.get("latestRevision") or t.get("percent", 0) > 0:
                revision = t.get("revisionName") or revision
                if t.get("latestRevision"):
                    break
        if not revision:
            # Fallback: read the same split from the Knative Route status.
            try:
                route = resource("Route", "serving.knative.dev/v1").get(
                    name=f"{isvc_name}-predictor", namespace=ns,
                ).to_dict()
                for t in (route.get("status") or {}).get("traffic") or []:
                    if t.get("latestRevision") or t.get("percent", 0) > 0:
                        revision = t.get("revisionName") or revision
                        if t.get("latestRevision"):
                            break
            except Exception:  # noqa: BLE001 - fallback best-effort
                pass
        if not revision:
            logger.warning("no active revision found for %s", isvc_name)
            return 0

        def find_private_service(revision_name: str) -> str:
            """Return the revision's private Knative ServerlessService name."""
            return _find_private_service(revision_name, ns)

        target = target or find_private_service(revision)
        if not target:
            logger.warning(
                "no private service found for revision %s", revision
            )
            return 0
        ingress_res = resource("IngressRoute", "traefik.io/v1alpha1")
        patched = 0
        for ing in manifests:
            if ing.get("kind") != "IngressRoute" or ing["metadata"].get("name") != isvc_name:
                continue
            # Rebuild the route list from the original manifest so entryPoints,
            # host match and TLS settings are preserved; only the backend
            # service is swapped to the active revision.
            routes = [
                {
                    **route,
                    "services": [{"name": target, "port": 80}],
                }
                for route in ing.get("spec", {}).get("routes", [])
            ]
            try:
                ingress_res.patch(
                    name=ing["metadata"]["name"],
                    namespace=ing["metadata"].get("namespace") or ns,
                    body={"spec": {"routes": routes}},
                    content_type="application/merge-patch+json",
                )
                logger.info("ingressroute %s -> %s", ing["metadata"]["name"], target)
                patched += 1
            except (ResourceNotFoundError, NotFoundError):
                logger.warning("ingressroute %s missing, skip reconcile", ing["metadata"]["name"])
        return patched

    if wait_isvc_name:
        end = time.time() + int(timeout_minutes) * 60
        last_msg = ""
        ns = wait_isvc_namespace or "nereidforge"
        res = resource("InferenceService", "serving.kserve.io/v1beta1")
        rev_res = resource("Revision", "serving.knative.dev/v1")
        # ISVC Ready stays True throughout a rolling redeploy (the previous
        # revision still serves traffic), so checking the top-level condition
        # returns on the first poll with the OLD revision's traffic target —
        # ingress reconcile then pinned the stale private service. Wait for
        # the NEW generation's revision: observedGeneration catches up, the
        # active traffic revision belongs to that generation, its Revision CR
        # is Ready, and its private service already lists pod endpoints.
        while time.time() < end:
            try:
                obj = res.get(name=wait_isvc_name, namespace=ns).to_dict()
                want_gen = int((obj.get("metadata") or {}).get("generation") or 0)
                status = obj.get("status") or {}
                try:
                    obs_gen = int(status.get("observedGeneration") or 0)
                except (TypeError, ValueError):
                    obs_gen = 0
                rev_name = _active_revision(obj)
                new_revision_ready = False
                if want_gen and obs_gen >= want_gen and rev_name:
                    try:
                        rev = rev_res.get(name=rev_name, namespace=ns).to_dict()
                        rlabels = (rev.get("metadata") or {}).get("labels") or {}
                        if str(rlabels.get(
                            "serving.knative.dev/configurationGeneration", ""
                        )) == str(want_gen):
                            for c in (rev.get("status") or {}).get("conditions") or []:
                                if c.get("type") == "Ready":
                                    new_revision_ready = c.get("status") == "True"
                                    if not new_revision_ready and c.get("message"):
                                        last_msg = c["message"]
                    except (ResourceNotFoundError, NotFoundError):
                        last_msg = f"revision {rev_name} not found yet"
                if new_revision_ready:
                    target = _find_private_service(rev_name, ns)
                    if target and _private_service_has_endpoints(target, ns):
                        logger.info(
                            "inference service %s ready at revision %s (%s)",
                            wait_isvc_name, rev_name, target,
                        )
                        try:
                            reconcile_ingress_targets(
                                wait_isvc_name, ns,
                                (
                                    (status.get("components") or {})
                                    .get("predictor", {})
                                    .get("traffic")
                                    or status.get("traffic") or []
                                ),
                                target=target,
                            )
                        except Exception as e:  # noqa: BLE001 - ingress fixup non-fatal
                            logger.warning("ingress reconcile failed: %s", e)
                        return f"ready:{wait_isvc_name}:{rev_name}"
                    last_msg = (
                        f"revision {rev_name} ready but private service "
                        f"{target or '?'} has no endpoints yet"
                    )
                else:
                    for c in status.get("conditions") or []:
                        if c.get("type") == "Ready" and c.get("status") == "False":
                            last_msg = c.get("message") or last_msg
            except (ResourceNotFoundError, NotFoundError):
                last_msg = "inference service not found yet"
            time.sleep(10)
        raise RuntimeError(
            f"inference service {wait_isvc_name} not ready within "
            f"{timeout_minutes} min: {(last_msg or '')[:300]}"
        )
    return f"applied:{len(manifests)}"
