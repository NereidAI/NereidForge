# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Tests for the worker's model_undeploy verification task: the DB row may
# only leave "deployed" after the live InferenceService is confirmed gone,
# and a deletion that fails must surface undeploy_failed instead of silent
# DB/cluster divergence (pit 170).
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.worker import Worker


def _task() -> SimpleNamespace:
    return SimpleNamespace(id="task-1")


_PARAMS = {
    "tenant": "default",
    "model_id": "m-1",
    "version": "v1",
    "service": "model-m-1-v1",
    "namespace": "nereidforge",
    "run_id": "run-1",
}


@pytest.fixture
def patched_reader(monkeypatch):
    """Patch serving_reader + cluster_ops.get_run_state for one test."""
    import app.clients.cluster_ops as cluster_ops
    import app.clients.k8s as k8s_mod

    state: dict = {"presence": iter(()), "run_state": ("", "")}

    class _Reader:
        available = True

        @staticmethod
        def isvc_presence(name, ns):
            return next(state["presence"])

    monkeypatch.setattr(k8s_mod, "serving_reader", _Reader)
    monkeypatch.setattr(cluster_ops, "get_run_state", lambda rid: state["run_state"])
    # The task sleeps 5s between polls for up to 300s; never wait in tests.
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    # Make the 300s deadline elapse instantly: time.time() advances 60s per
    # call while the loop checks it.
    clock = {"t": 0.0}

    def _fake_time():
        clock["t"] += 60.0
        return clock["t"]

    monkeypatch.setattr("time.time", _fake_time)
    return state


def _worker() -> tuple[Worker, dict]:
    w = Worker.__new__(Worker)
    calls = {"cleared": 0, "failed": 0, "completed": 0, "progress": 0}

    class _Models:
        @staticmethod
        def mark_undeployed(*_a):
            calls["cleared"] += 1

        @staticmethod
        def mark_undeploy_failed(*_a):
            calls["failed"] += 1

    class _Dataset:
        @staticmethod
        def set_async_task_progress(*_a):
            calls["progress"] += 1

        @staticmethod
        def complete_async_task(*_a):
            calls["completed"] += 1

    w.models = _Models()
    w.dataset = _Dataset()
    w._task_cancelled = lambda _tid: False
    return w, calls


def test_undeploy_clears_db_only_after_confirmed_404(patched_reader):
    patched_reader["presence"] = iter([True, True, False])
    w, calls = _worker()
    w._undeploy_model(_task(), dict(_PARAMS))
    assert calls["cleared"] == 1
    assert calls["failed"] == 0
    assert calls["completed"] == 1


def test_undeploy_fails_fast_when_pipeline_run_failed(patched_reader):
    patched_reader["presence"] = iter([True, True])
    patched_reader["run_state"] = ("FAILED", "delete rejected")
    w, calls = _worker()
    with pytest.raises(RuntimeError, match="FAILED"):
        w._undeploy_model(_task(), dict(_PARAMS))
    assert calls["cleared"] == 0
    # The worker's task-error hook records undeploy_failed on the version.
    w._undeploy_model_failure(_task(), dict(_PARAMS), RuntimeError("boom"))
    assert calls["failed"] == 1


def test_undeploy_fails_when_object_survives_deadline(patched_reader):
    patched_reader["presence"] = iter(lambda: True, None)  # never 404
    w, calls = _worker()
    with pytest.raises(RuntimeError, match="still exists"):
        w._undeploy_model(_task(), dict(_PARAMS))
    assert calls["cleared"] == 0


def test_undeploy_without_reader_trusts_run(monkeypatch):
    import app.clients.k8s as k8s_mod

    class _Reader:
        available = False

    monkeypatch.setattr(k8s_mod, "serving_reader", _Reader)
    w, calls = _worker()
    w._undeploy_model(_task(), dict(_PARAMS))
    assert calls["cleared"] == 1
    assert calls["completed"] == 1
