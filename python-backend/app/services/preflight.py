# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Deployment preflight checks.

Runs cheap, read-only probes against the external dependencies of the
train -> deploy data plane (kubernetes API / serving namespace, S3 bucket,
KFP, runtime images, ML backends table, inference domain). The goal is to
surface configuration drift on an admin page / at startup instead of
discovering it as a cryptic deploy-time failure (see pit 170/171).

Every probe is best effort and must never raise: a failed probe becomes
{"status": "error", ...}. "error" blocks the data plane; "warn" degrades
it (e.g. no external inference domain, empty ML backends table).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import requests
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.clients.k8s import serving_reader
from app.core.config import config

log = logging.getLogger("app.preflight")

OK, WARN, ERROR = "ok", "warn", "error"


def _check(name: str, status: str, message: str, **extra: Any) -> dict[str, Any]:
    out = {"name": name, "status": status, "message": message}
    out.update(extra)
    return out


def _check_kubernetes() -> dict[str, Any]:
    if not serving_reader.available:
        return _check(
            "kubernetes", ERROR,
            "kubernetes API unreachable: serving verification, deployments "
            "list enrichment and namespace probes are disabled",
        )
    ns = config.serving.namespace or "nereidforge"
    exists = serving_reader.namespace_exists(ns)
    if exists is True:
        return _check("kubernetes", OK, f"API reachable; serving namespace '{ns}' exists", namespace=ns)
    if exists is False:
        return _check(
            "kubernetes", ERROR,
            f"serving namespace '{ns}' does not exist; ISVC creation will fail",
            namespace=ns,
        )
    return _check("kubernetes", WARN, "API reachable but namespace probe failed", namespace=ns)


def _check_s3() -> dict[str, Any]:
    from app.services import _s3

    bucket = _s3.bucket
    try:
        _s3.client.head_bucket(Bucket=bucket)
        return _check("s3", OK, f"bucket '{bucket}' reachable at {_s3.endpoint}", bucket=bucket)
    except Exception as e:  # noqa: BLE001 - probe must never raise
        return _check(
            "s3", ERROR,
            f"bucket '{bucket}' not reachable at {_s3.endpoint}: {str(e)[:200]}",
            bucket=bucket,
        )


def _check_kfp() -> dict[str, Any]:
    host = (config.kfp.host or "").rstrip("/")
    if not host:
        return _check("kfp", ERROR, "kfp.host is not configured")
    # KFP v2 health endpoint; 200/404 both prove the proxy is listening.
    url = f"{host}/apis/v1beta1/healthz"
    try:
        r = requests.get(
            url,
            headers={"kubeflow-userid": "admin@nereidai.local"},
            timeout=5,
        )
        if r.status_code < 500:
            return _check("kfp", OK, f"KFP API reachable at {host} ({r.status_code})")
        return _check("kfp", ERROR, f"KFP API answered {r.status_code} at {host}")
    except Exception as e:  # noqa: BLE001
        return _check("kfp", ERROR, f"KFP API unreachable at {host}: {str(e)[:200]}")


def _check_runtime_images() -> dict[str, Any]:
    images = config.serving.runtime_images or {}
    missing = [k for k in ("openvino", "triton") if not str(images.get(k, ""))]
    if missing:
        return _check(
            "runtime_images", WARN,
            f"no serving image configured for runtimes: {', '.join(missing)}",
        )
    return _check(
        "runtime_images", OK,
        ", ".join(f"{k}={v}" for k, v in sorted(images.items())),
    )


def _check_inference_domain() -> dict[str, Any]:
    suffix = config.serving.domain_suffix or "inference"
    if config.serving.platform_domain:
        return _check(
            "inference_domain", OK,
            f"external domain pattern *.{suffix}.{config.serving.platform_domain}",
        )
    # The admin system_config may supply the domain instead; the caller
    # re-checks that and upgrades this result when present.
    return _check(
        "inference_domain", WARN,
        "serving.platform_domain not set in config; predictors are "
        "cluster-internal unless the admin settings define a platform domain",
    )


def _check_ml_backends(db: Optional[Session]) -> dict[str, Any]:
    if db is None:
        return _check("ml_backends", WARN, "skipped (no DB session)")
    try:
        from app.models.orm import MLBackend

        n = db.scalar(select(func.count()).select_from(MLBackend)) or 0
        if n == 0:
            return _check(
                "ml_backends", WARN,
                "no ML backend registered; pre-annotation/active-learning "
                "features stay disabled (register one on the ML Backends page)",
            )
        unreachable = db.scalar(
            select(func.count()).select_from(MLBackend).where(MLBackend.status == "unreachable")
        ) or 0
        if unreachable:
            return _check(
                "ml_backends", WARN,
                f"{n} registered, {unreachable} currently unreachable",
                total=n, unreachable=unreachable,
            )
        return _check("ml_backends", OK, f"{n} ML backend(s) registered", total=n)
    except Exception as e:  # noqa: BLE001
        return _check("ml_backends", WARN, f"could not read ml_backends: {str(e)[:160]}")


def _check_admin_domain(db: Optional[Session]) -> Optional[str]:
    """Platform domain from admin system_config, if any."""
    if db is None:
        return None
    try:
        from app.services.admin import AdminService

        cfg = AdminService(db).get_system_config()
        return str(cfg.get("platform_domain", "") or "") or None
    except Exception:  # noqa: BLE001
        return None


def run_preflight(db: Optional[Session] = None) -> dict[str, Any]:
    """Run all probes and return {summary, checks:[...]}."""
    checks = [
        _check_kubernetes(),
        _check_s3(),
        _check_kfp(),
        _check_runtime_images(),
        _check_ml_backends(db),
    ]
    domain_check = _check_inference_domain()
    admin_domain = _check_admin_domain(db)
    if admin_domain and domain_check["status"] != OK:
        domain_check = _check(
            "inference_domain", OK,
            f"external domain from admin settings: *."
            f"{config.serving.domain_suffix or 'inference'}.{admin_domain}",
        )
    checks.append(domain_check)

    errors = [c for c in checks if c["status"] == ERROR]
    warns = [c for c in checks if c["status"] == WARN]
    summary = ERROR if errors else (WARN if warns else OK)
    return {
        "summary": summary,
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warns),
        "checks": checks,
    }


def log_preflight_on_startup(db: Session) -> None:
    """Best-effort startup probe; logs warnings/errors, never blocks startup."""
    try:
        report = run_preflight(db)
        for c in report["checks"]:
            if c["status"] == ERROR:
                log.error("preflight [%s] %s", c["name"], c["message"])
            elif c["status"] == WARN:
                log.warning("preflight [%s] %s", c["name"], c["message"])
        log.info(
            "preflight complete: summary=%s errors=%d warnings=%d",
            report["summary"], report["error_count"], report["warning_count"],
        )
    except Exception:  # noqa: BLE001 - startup must never fail on diagnostics
        log.exception("preflight probe crashed")
