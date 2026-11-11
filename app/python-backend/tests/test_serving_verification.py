# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Tests for the post-mutation serving verification gates (pit 170 root cause 6:
# a KFP run reporting success is not proof the InferenceService exists).
from __future__ import annotations

import time

import pytest

from app.clients import k8s as k8s_mod
from app.clients.k8s import (
    K8sServingReader,
    K8sServingUnavailable,
    K8sServingVerifyError,
)


def _reader() -> K8sServingReader:
    r = K8sServingReader()
    # Pretend kube config was loaded successfully without touching a cluster.
    r._loaded = True
    r._available = True
    return r


def test_wait_until_ready_succeeds(monkeypatch):
    r = _reader()
    monkeypatch.setattr(r, "isvc_presence", lambda name, ns: True)
    monkeypatch.setattr(
        r, "describe", lambda name, ns, force=False: {"ready": True, "error": ""}
    )
    out = r.wait_until_ready("svc", "ns", timeout=2, interval=0.01)
    assert out["ready"] is True


def test_wait_until_ready_fails_on_missing_object(monkeypatch):
    r = _reader()
    monkeypatch.setattr(r, "isvc_presence", lambda name, ns: False)
    with pytest.raises(K8sServingVerifyError, match="does not exist"):
        r.wait_until_ready("svc", "ns", timeout=0.1, interval=0.01)


def test_wait_until_ready_fails_when_not_ready(monkeypatch):
    r = _reader()
    polls = {"n": 0}

    def presence(name, ns):
        polls["n"] += 1
        return True

    monkeypatch.setattr(r, "isvc_presence", presence)
    monkeypatch.setattr(
        r, "describe",
        lambda name, ns, force=False: {"ready": False, "error": "Predictor failed: ImagePullBackOff"},
    )
    with pytest.raises(K8sServingVerifyError, match="not Ready"):
        r.wait_until_ready("svc", "ns", timeout=0.1, interval=0.01)
    assert polls["n"] >= 1


def test_wait_until_absent_succeeds_on_confirmed_404(monkeypatch):
    r = _reader()
    monkeypatch.setattr(r, "isvc_presence", lambda name, ns: False)
    r.wait_until_absent("svc", "ns", timeout=2, interval=0.01)  # must not raise


def test_wait_until_absent_fails_while_object_lives(monkeypatch):
    r = _reader()
    monkeypatch.setattr(r, "isvc_presence", lambda name, ns: True)
    with pytest.raises(K8sServingVerifyError, match="still exists"):
        r.wait_until_absent("svc", "ns", timeout=0.1, interval=0.01)


def test_unreachable_reader_blocks_verification():
    r = K8sServingReader()
    r._loaded = True
    r._available = False
    with pytest.raises(K8sServingUnavailable):
        r.wait_until_ready("svc", "ns", timeout=0.1, interval=0.01)
    with pytest.raises(K8sServingUnavailable):
        r.wait_until_absent("svc", "ns", timeout=0.1, interval=0.01)


def test_transient_api_errors_do_not_count_as_absent(monkeypatch):
    r = _reader()
    seq = [None, None, False]  # two unreachable polls, then confirmed 404
    monkeypatch.setattr(r, "isvc_presence", lambda name, ns: seq.pop(0))
    r.wait_until_absent("svc", "ns", timeout=5, interval=0.01)


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    yield


def test_module_exposes_shared_reader():
    assert isinstance(k8s_mod.serving_reader, K8sServingReader)
