# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application: wires routers, auth middleware, and error handling."""
from __future__ import annotations

import logging
import os
import sys

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app.api.helpers import error, get_svc
from app.api.routers import auth, datasets, integration, jobs, models, notebooks, projects, system
from app.api.routers import base_models as base_models_router
from app.api.routers import sso as sso_router
from app.core.auth import require_auth
from app.core.config import config
from app.services import Services


def _setup_logging() -> None:
    """Configure logging from config + env overrides.

    Priority: NEREIDFORGE_LOG_LEVEL env > config.logging.level > INFO.
    NEREIDFORGE_LOG_FORMAT=plain switches from JSON to human-readable.
    """
    from app.core.config import config as cfg

    level_name = os.environ.get("NEREIDFORGE_LOG_LEVEL", cfg.logging.level or "info").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.environ.get("NEREIDFORGE_LOG_FORMAT", cfg.logging.format or "json")

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "plain":
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
    else:
        handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    root.setLevel(level)
    # Replace any existing handlers (uvicorn adds its own at startup).
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy libraries unless DEBUG.
    for lib in ("httpx", "httpcore", "urllib3", "boto3", "botocore"):
        logging.getLogger(lib).setLevel(logging.WARNING if level > logging.DEBUG else level)

    logging.getLogger("app").setLevel(level)
    log = logging.getLogger("app.startup")
    log.info("logging configured: level=%s format=%s", level_name, fmt)


_setup_logging()

app = FastAPI(title="NereidForge API", version="0.1.0", docs_url="/docs", openapi_url="/openapi.json")

log = logging.getLogger("app.startup")


@app.on_event("startup")
async def _startup_check():
    from app.clients.hub import hub_client
    from app.core.config import config as cfg
    from app.db.session import Base, engine

    log.info("hub.host=%r enabled=%s", cfg.hub.host, hub_client.enabled())
    if hub_client._init_error:
        log.warning("hub init error: %s", hub_client._init_error)

    # Create tables if they don't exist (dev/first deploy convenience).
    Base.metadata.create_all(engine)
    # create_all never alters existing tables; lightweight additive migrations
    # for already-deployed DBs are applied here (idempotent).
    _ensure_columns(engine)
    _relax_job_timestamp_constraints(engine)
    _relax_dataset_version_constraints(engine)

    # Backfill the base-model pool rows from S3 (and migrate legacy
    # root-level dirs into llm/); best-effort, never blocks startup.
    try:
        from app.clients.s3 import SeaweedFSClient
        from app.core.config import config as cfg2
        from app.db.session import SessionLocal
        from app.services.base_models import BaseModelService

        def _reconcile():
            db = SessionLocal()
            try:
                svc = BaseModelService(
                    db, SeaweedFSClient(),
                    bucket=cfg2.storage.bucket or "nereidforge",
                )
                n = svc.reconcile_with_s3("default")
                db.commit()
                if n:
                    log.info("base-model pool reconcile: %d rows ensured", n)
            finally:
                db.close()

        import asyncio

        asyncio.get_event_loop().run_in_executor(None, _reconcile)
    except Exception as exc:  # noqa: BLE001
        log.warning("base-model pool reconcile failed: %s", exc)

    # Data-plane dependency preflight (k8s/S3/KFP/images/ML backends). Runs in
    # a worker thread so slow probes never block API readiness; results land
    # in the log and the same data is available live at GET /system/preflight.
    try:
        from app.db.session import SessionLocal
        from app.services.preflight import log_preflight_on_startup

        def _preflight():
            db = SessionLocal()
            try:
                log_preflight_on_startup(db)
            finally:
                db.close()

        import asyncio

        asyncio.get_event_loop().run_in_executor(None, _preflight)
    except Exception as exc:  # noqa: BLE001
        log.warning("startup preflight failed: %s", exc)


def _relax_job_timestamp_constraints(engine) -> None:
    """Drop NOT NULL on jobs.started_at/completed_at for existing databases.

    Pending jobs legitimately have no start/completion time; older schemas
    created these columns as NOT NULL, which made creating a job fail with a
    NotNullViolation. Idempotent: only alters columns that are still NO.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "jobs" not in set(inspector.get_table_names()):
        return
    nullable = {c["name"]: c["nullable"] for c in inspector.get_columns("jobs")}
    with engine.begin() as conn:
        for col in ("started_at", "completed_at"):
            if nullable.get(col) is False:
                conn.execute(text(f"ALTER TABLE jobs ALTER COLUMN {col} DROP NOT NULL"))
                log.info("applied lightweight migration: jobs.%s made nullable", col)


def _relax_dataset_version_constraints(engine) -> None:
    """Drop NOT NULL on dataset_versions.source_project_id for old databases.

    A plain dataset upload creates a draft version with no provenance link to
    an annotation project (None); only annotated exports carry a project id.
    Schemas created before the column became Optional reject the insert with a
    NotNullViolation. Idempotent: only alters the column when it is still NO.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "dataset_versions" not in set(inspector.get_table_names()):
        return
    nullable = {c["name"]: c["nullable"] for c in inspector.get_columns("dataset_versions")}
    if nullable.get("source_project_id") is False:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE dataset_versions ALTER COLUMN source_project_id DROP NOT NULL"))
        log.info("applied lightweight migration: dataset_versions.source_project_id made nullable")


def _ensure_columns(engine) -> None:
    """Add columns introduced after first release to existing tables.

    Idempotent: checks current columns via SQLAlchemy reflection and issues
    ADD COLUMN only for the missing ones. Postgres DDL is transactional.
    """
    from sqlalchemy import inspect, text

    wanted = {
        "annotation_projects": {
            "preannotate_model": "JSONB",
        },
        # Base-model metadata columns added after the first pool release.
        # Constant defaults make ADD COLUMN instant and give existing rows
        # sensible values; task/status/source are refined afterwards.
        "base_models": {
            "framework": ("VARCHAR", "'pytorch'"),
            "task": ("VARCHAR", "'detection'"),
            "status": ("VARCHAR", "'develop'"),
            "description": ("VARCHAR", "''"),
            "source": ("VARCHAR", "'manual'"),
        },
        # Suite SSO JIT linkage (identity sub/issuer, empty for local users).
        "users": {
            "external_id": ("VARCHAR", "''"),
            "external_issuer": ("VARCHAR", "''"),
        },
        # Capability announcement of a registered trainer image (ADR-015).
        # Existing rows stay empty: an image that announced nothing is still
        # usable, it simply contributes no capability facts until refreshed.
        "container_images": {
            "profile": ("JSONB", "'{}'"),
            "profile_at": "TIMESTAMPTZ",
            "profile_error": ("VARCHAR", "''"),
        },
    }
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, cols in wanted.items():
            if table not in existing_tables:
                continue  # create_all will build it with the new columns
            present = {c["name"] for c in inspector.get_columns(table)}
            for name, spec in cols.items():
                if name in present:
                    continue
                ddl_type, default = spec if isinstance(spec, tuple) else (spec, None)
                default_sql = f" DEFAULT {default}" if default else ""
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl_type}{default_sql}"))
                log.info("applied lightweight migration: %s.%s added", table, name)
        # Backfill source from the legacy uploaded_by marker, then infer the
        # CV task from the stock-model filename (ppyoloe-seg.pdparams, ...-cls.pdparams).
        if "base_models" in existing_tables:
            conn.execute(text(
                "UPDATE base_models SET source = 's3-reconcile' "
                "WHERE uploaded_by = 's3-reconcile' AND source = 'manual'"
            ))
            conn.execute(text(
                "UPDATE base_models SET task = 'segmentation' "
                "WHERE kind = 'cv' AND name ILIKE '%-seg.%' AND task = 'detection'"
            ))
            conn.execute(text(
                "UPDATE base_models SET task = 'classification' "
                "WHERE kind = 'cv' AND name ILIKE '%-cls.%' AND task = 'detection'"
            ))
            # Stock upstream weights backfilled by reconcile are "original".
            conn.execute(text(
                "UPDATE base_models SET status = 'original' "
                "WHERE source = 's3-reconcile' AND status = 'develop'"
            ))


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    """Return a Go-compatible error envelope for unhandled exceptions."""
    return JSONResponse(status_code=500, content={"code": "Internal Server Error", "message": str(exc)})


@app.get("/healthz", tags=["system"])
async def healthz() -> dict:
    """Liveness probe."""
    return {"status": "ok", "service": "nereidforge-api"}


@app.get("/ready", tags=["system"])
async def ready() -> dict:
    """Readiness probe."""
    return {"status": "ready"}


# Authentication routes (no auth required).
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
# Suite SSO (NereidIdentity standard v2): status/login/callback are public;
# the mode-ns ticket callback is a product route at config.sso.ns.callback_path.
app.include_router(sso_router.router, prefix="/api/v1/auth/sso", tags=["sso"])
app.include_router(
    sso_router.ns_router,
    prefix=config.sso.ns.callback_path or "/auth/ns-callback",
    tags=["sso"],
)

# NS integration (suite console tenant picker): secret-authenticated, so it is
# mounted WITHOUT _protected like the NS callback — NS carries no NF bearer
# token. Its path is /v1/integration/ns/... (not under /api) and reaches the
# API through the matching ingress path.
app.include_router(integration.router, tags=["integration"])

# Authenticated API routes: all business routers carry the auth dependency.
_protected = [Depends(require_auth)]
app.include_router(datasets.router, prefix="/api/v1", tags=["datasets"], dependencies=_protected)
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"], dependencies=_protected)
app.include_router(models.router, prefix="/api/v1", tags=["models"], dependencies=_protected)
app.include_router(projects.router, prefix="/api/v1", tags=["projects"], dependencies=_protected)
# Notebook servers: CRUD is protected; the jupyter proxy path stays open
# (browser <img>/ws cannot send auth headers; the token lives server-side).
app.include_router(notebooks.router, prefix="/api/v1", tags=["notebooks"])
# Bare-prefix mount: the browser talks to /notebook-servers/{name}/... which
# EQUALS the jupyter base_url, so no URL rewriting is ever needed (assets,
# relative routes, redirects all stay consistent end-to-end).
app.include_router(notebooks.router, tags=["notebooks-bare"])
app.include_router(system.router, prefix="/api/v1", tags=["system"], dependencies=_protected)
app.include_router(
    base_models_router.router, prefix="/api/v1", tags=["base-models"],
    dependencies=_protected,
)

# Image proxies are auth-exempt (browser <img> cannot send headers).
# Registered directly on the app with the /api/v1 prefix (a router include
# previously failed to mount these paths).

def _content_type(name: str) -> str:
    """Guess the content type from a file extension."""
    ext = os.path.splitext(name)[1].lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".flv": "video/x-flv",
        ".wmv": "video/x-ms-wmv",
        ".m4v": "video/x-m4v",
    }.get(ext, "application/octet-stream")


_PROXY_HEADERS = {
    "Cache-Control": "public, max-age=86400",
    "Access-Control-Allow-Origin": "*",
}


@app.get("/api/v1/s3-object")
async def s3_object(uri: str, svc: Services = Depends(get_svc)):
    """Proxy an S3 object as HTTP bytes (browser-reachable, no cluster IP exposure)."""
    if not uri or not uri.startswith("s3://"):
        raise error(400, "uri is required (s3://...)")
    filer = svc.s3.s3_uri_to_filer_path(uri)
    if not filer:
        raise error(400, "invalid s3 uri")
    try:
        data = svc.s3.read_object(filer)
    except Exception:
        raise error(404, "object not found")
    return Response(content=data, media_type=_content_type(uri), headers=_PROXY_HEADERS)


@app.get("/api/v1/datasets/{dataset_id}/image")
async def dataset_image(
    dataset_id: str, path: str, tenant: str = "default", version: int = 0,
    svc: Services = Depends(get_svc),
):
    """Proxy an image from a dataset version (browser-reachable, auth-exempt).

    version=0 (default) resolves to the dataset's latest version.
    """
    ds = svc.dataset.get_dataset(tenant, dataset_id)
    if not ds:
        raise error(404, "dataset not found")
    ver_no = version if version > 0 else ds["version"]
    ver = svc.dataset.get_dataset_version(tenant, dataset_id, ver_no)
    if not ver or not ver.uri:
        raise error(404, "dataset version not found")
    filer_base = svc.s3.s3_uri_to_filer_path(ver.uri)
    if not filer_base:
        raise error(404, "invalid version uri")
    # Path traversal protection.
    safe_path = os.path.normpath("/" + path).lstrip("/")
    filer = f"{filer_base}/{safe_path}"
    try:
        data = svc.s3.read_object(filer)
    except Exception:
        raise error(404, "image not found")
    return Response(content=data, media_type=_content_type(path), headers=_PROXY_HEADERS)
