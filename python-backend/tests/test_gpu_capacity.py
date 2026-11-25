# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cluster GPU capacity discovery client."""

import types

from app.clients import gpu_capacity


def _node(name, capacities, unschedulable=False):
    """Build a minimal Kubernetes Node-like object."""
    return types.SimpleNamespace(
        metadata=types.SimpleNamespace(name=name),
        spec=types.SimpleNamespace(unschedulable=unschedulable),
        status=types.SimpleNamespace(allocatable=capacities),
    )


def _patch_k8s(monkeypatch, nodes):
    """Replace the kubernetes config loader and node API with stubs."""
    fake_k8s = types.ModuleType("kubernetes")
    fake_config = types.ModuleType("kubernetes.config")
    fake_config.load_incluster_config = lambda: None
    fake_client = types.ModuleType("kubernetes.client")

    class FakeCoreV1Api:
        def list_node(self):
            return types.SimpleNamespace(items=nodes)

    fake_client.CoreV1Api = FakeCoreV1Api
    fake_k8s.config = fake_config
    fake_k8s.client = fake_client

    import sys

    monkeypatch.setitem(sys.modules, "kubernetes", fake_k8s)
    monkeypatch.setitem(sys.modules, "kubernetes.config", fake_config)
    monkeypatch.setitem(sys.modules, "kubernetes.client", fake_client)


def test_capacity_counts_shared_devices(monkeypatch):
    nodes = [
        _node("xpu-node-a", {"gpu.intel.com/xe": 20}),
        _node("xpu-node-b", {"gpu.intel.com/xe": 0}),
    ]
    _patch_k8s(monkeypatch, nodes)
    result = gpu_capacity.gpu_capacity()
    intel = result["vendors"]["intel"]
    assert intel["total"] == 20
    assert intel["max_per_node"] == 20
    assert intel["resource"] == "gpu.intel.com/xe"


def test_multiple_vendors_and_nodes(monkeypatch):
    nodes = [
        _node("gpu-a", {"nvidia.com/gpu": 4}),
        _node("gpu-b", {"nvidia.com/gpu": 8}),
    ]
    _patch_k8s(monkeypatch, nodes)
    result = gpu_capacity.gpu_capacity()
    nvidia = result["vendors"]["nvidia"]
    assert nvidia["total"] == 12
    assert nvidia["max_per_node"] == 8
    assert [n["name"] for n in nvidia["nodes"]] == ["gpu-a", "gpu-b"]


def test_cordoned_nodes_are_skipped(monkeypatch):
    nodes = [_node("drained", {"gpu.intel.com/xe": 20}, unschedulable=True)]
    _patch_k8s(monkeypatch, nodes)
    assert gpu_capacity.gpu_capacity()["vendors"] == {}


def test_quantity_string_parsed(monkeypatch):
    nodes = [_node("gpu-a", {"nvidia.com/gpu": "4"})]
    _patch_k8s(monkeypatch, nodes)
    assert gpu_capacity.gpu_capacity()["vendors"]["nvidia"]["total"] == 4
