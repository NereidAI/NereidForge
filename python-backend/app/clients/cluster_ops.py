# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Shared submit helper for the small in-cluster KFP pipelines.

NereidForge never touches the Kubernetes cluster directly. Cluster mutations
(deploy/undeploy an InferenceService + ingress, run a Katib grid search) are
executed by KFP pipeline pods that use their own service account; this module
compiles and submits those pipelines and waits for their result.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import sys
import tempfile
import time
from typing import Any

from app.core.config import config

log = logging.getLogger(__name__)


def _resolve_pipelines_dir() -> str:
    """Locate the kfp-pipelines directory (image or dev checkout).

    Inside the container the dir is /app/kfp-pipelines; in a dev checkout
    (backend runs on the host) fall back to <repo>/kfp-pipelines.
    NF_KFP_PIPELINES_DIR always wins as an explicit override.
    """
    env = os.environ.get("NF_KFP_PIPELINES_DIR")
    if env:
        return env
    if os.path.isdir("/app/kfp-pipelines"):
        return "/app/kfp-pipelines"
    here = pathlib.Path(__file__).resolve()
    return str(here.parents[3] / "kfp-pipelines")


# Path of the kfp-pipelines dir (see _resolve_pipelines_dir).
PIPELINES_DIR = _resolve_pipelines_dir()

# Pipeline modules and their components do `from images import ...` at module
# top level (compile-time image resolution). When the module is loaded below
# via importlib from an explicit file path, Python does not add its parent to
# sys.path, so a dev-mode worker on the host fails with "No module named
# 'images'" (the baked image works only because /app/kfp-pipelines is already
# on PYTHONPATH there). Make both layouts identical.
if PIPELINES_DIR not in sys.path:
    sys.path.insert(0, PIPELINES_DIR)

# pipeline_name (== module + function name in pipelines/) -> compiled YAML path.
_PACKAGE_CACHE: dict[str, str] = {}


def _pipeline_package(pipeline_name: str) -> str:
    """Compile the named pipeline to a YAML package path (cached)."""
    cached = _PACKAGE_CACHE.get(pipeline_name)
    if cached and os.path.exists(cached):
        return cached
    mod_path = os.path.join(PIPELINES_DIR, "pipelines", f"{pipeline_name}.py")
    if not os.path.exists(mod_path):
        raise RuntimeError(f"kfp pipeline module not found: {mod_path}")
    spec = importlib.util.spec_from_file_location(f"nf_pipeline_{pipeline_name}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, pipeline_name)
    fd, out = tempfile.mkstemp(prefix=f"nf_{pipeline_name}_", suffix=".yaml")
    os.close(fd)
    try:
        import kfp.compiler

        kfp.compiler.Compiler().compile(fn, out)
    except Exception:
        try:
            os.remove(out)
        except OSError:
            pass
        raise
    _PACKAGE_CACHE[pipeline_name] = out
    return out


def _kfp_client():
    """KFP SDK client with the required user-identity header.

    The KFP v2 API is behind istio authorization and refuses requests without
    a kubeflow-userid header. The SDK client has no headers parameter, so we
    inject the header into the generated swagger ApiClient's default headers.
    """
    from kfp import Client

    # Same default as app/clients/kfp.py: the admin/default tenant's KFP user.
    userid = "admin@nereidforge.local"
    client = Client(host=config.kfp.host, namespace="kubeflow")
    for svc in ("_experiment_api", "_run_api", "_pipelines_api", "_upload_api"):
        api = getattr(client, svc, None)
        if api is not None and hasattr(api, "api_client"):
            api.api_client.set_default_header("kubeflow-userid", userid)
    return client


def submit_pipeline(
    pipeline_name: str,
    params: dict[str, Any],
    run_name: str,
    namespace: str = "nereidforge",
    wait: bool = True,
    timeout: int = 1800,
    poll_interval: int = 20,
) -> str:
    """Compile + submit a cluster-op pipeline run via KFP.

    Returns the run id. When wait is True it blocks until the run reaches a
    terminal state and raises on failure; otherwise it returns immediately
    and the caller (the worker) polls the run like any other KFP run.
    """
    import uuid

    client = _kfp_client()
    pkg = _pipeline_package(pipeline_name)
    run_name = run_name or f"{pipeline_name}-{uuid.uuid4().hex[:8]}"
    log.info("submitting %s run %s (wait=%s)", pipeline_name, run_name, wait)
    try:
        run = client.create_run_from_pipeline_package(
            pipeline_file=pkg,
            run_name=run_name,
            namespace=namespace,
            arguments=params,
        )
    except Exception as e:  # noqa: BLE001
        log.error("submit %s failed: %s", pipeline_name, e)
        raise RuntimeError(f"pipeline submission failed: {e}")
    run_id = run.run_id
    log.info("%s run submitted: %s", pipeline_name, run_id)
    if not wait:
        return run_id

    start = time.time()
    while time.time() - start < timeout:
        try:
            detail = client.get_run(run_id)
            state = detail.state
        except Exception as e:  # noqa: BLE001
            log.warning("get_run %s failed: %s", run_id, e)
            time.sleep(poll_interval)
            continue
        if state == "SUCCEEDED":
            log.info("%s run %s succeeded", pipeline_name, run_id)
            return run_id
        if state in ("FAILED", "CANCELED", "CANCELLED", "ERROR"):
            msg = (detail.error or "")[:500]
            log.error("%s run %s %s: %s", pipeline_name, run_id, state, msg)
            raise RuntimeError(f"pipeline run {state}: {msg or 'unknown error'}")
        time.sleep(poll_interval)
    raise RuntimeError(f"pipeline run {run_id} timed out after {timeout}s")


def get_run_state(run_id: str) -> tuple[str, str]:
    """Return (state, error_message) for one KFP run; ("", "") when unreadable.

    Best-effort poll used by the worker's undeploy verification to fail fast
    when the deletion pipeline itself failed, instead of waiting the full
    absence-verification timeout.
    """
    try:
        detail = _kfp_client().get_run(run_id)
        return str(detail.state or ""), str(detail.error or "")[:500]
    except Exception as e:  # noqa: BLE001 - reader must never raise
        log.warning("get_run %s failed: %s", run_id, e)
        return "", ""
