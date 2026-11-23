# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Cluster GPU capacity discovery from node allocatable extended resources.

Device plugins advertise physical GPUs AND shared/vGPU devices (e.g. the
Intel GPU plugin's ``shared-dev-num`` is reflected in the allocatable count
of ``gpu.intel.com/xe``). Reading node objects therefore gives the real
upper bound a training job may request, without a hardcoded limit.

The module is intentionally tolerant: when the pod lacks RBAC to list
nodes or the in-cluster config is unavailable, callers receive an empty
result and the UI falls back to an unbounded GPU-count field.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.accelerators import ACCELERATOR_PROFILES

logger = logging.getLogger(__name__)


def _resource_count(quantity: Any) -> int:
    """Convert a Kubernetes resource quantity object/string to an int."""
    if isinstance(quantity, (int, float)):
        return int(quantity)
    if quantity is None:
        return 0
    try:
        return int(str(quantity))
    except (TypeError, ValueError):
        return 0


def gpu_capacity() -> dict[str, Any]:
    """Return per-vendor shared-device capacity across schedulable nodes.

    Shape::

        {
          "vendors": {
            "nvidia": {
              "resource": "nvidia.com/gpu",
              "total": 8,
              "max_per_node": 4,
              "nodes": [{"name": "node-a", "devices": 4}],
            },
          }
        }

    Vendors without any advertised devices are omitted. Returns an empty
    ``vendors`` map when the Kubernetes API cannot be reached.
    """
    try:
        from kubernetes import client as kclient
        from kubernetes import config as kconfig
    except ImportError:
        return {"vendors": {}}

    try:
        kconfig.load_incluster_config()
    except Exception:
        try:
            kconfig.load_kube_config()
        except Exception as e:
            logger.info("gpu capacity unavailable: %s", e)
            return {"vendors": {}}

    try:
        core = kclient.CoreV1Api()
        nodes = core.list_node().items
    except Exception as e:
        logger.info("gpu capacity node list denied: %s", e)
        return {"vendors": {}}

    # Vendor id -> collected node entries.
    collected: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        # Skip cordoned nodes: their devices cannot receive new pods.
        if getattr(node.spec, "unschedulable", False):
            continue
        allocatable = getattr(node.status, "allocatable", None) or {}
        for vendor, profile in ACCELERATOR_PROFILES.items():
            resource = profile.get("resource")
            if not resource:
                continue
            count = _resource_count(allocatable.get(resource))
            if count <= 0:
                continue
            collected.setdefault(vendor, []).append(
                {"name": node.metadata.name, "devices": count}
            )

    vendors: dict[str, Any] = {}
    for vendor, entries in collected.items():
        entries.sort(key=lambda e: e["name"])
        vendors[vendor] = {
            "resource": ACCELERATOR_PROFILES[vendor]["resource"],
            "total": sum(e["devices"] for e in entries),
            "max_per_node": max(e["devices"] for e in entries),
            "nodes": entries,
        }
    return {"vendors": vendors}
