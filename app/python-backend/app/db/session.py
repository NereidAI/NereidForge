# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Database engine and session factory using synchronous SQLAlchemy and pg8000."""
from __future__ import annotations

import ssl
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import sessionmaker

from app.core.config import config
from app.db.base import Base as Base

# Import all ORM models so they register with Base.metadata
from app.models.orm import (  # noqa: F401
    AnnotationProject,
    AsyncTask,
    ContainerImage,
    Dataset,
    DatasetVersion,
    Job,
    KvObject,
    MLBackend,
    Model,
    ModelVersion,
    Role,
    SystemConfig,
    Tenant,
    User,
)


def _engine_parameters(dsn: str) -> tuple[URL, dict[str, Any]]:
    """Translate the configured libpq-style DSN into pg8000 engine arguments.

    pg8000 does not accept the libpq ``sslmode`` query parameter. Its native
    values preserve disable/prefer/require semantics, while an SSLContext
    controls certificate and hostname verification. The libpq ``allow`` mode
    needs a plaintext-first reconnect that pg8000 cannot express, so rejecting
    it is safer than silently weakening or strengthening the requested policy.
    """
    url = make_url(dsn)
    if url.get_backend_name() != "postgresql":
        raise ValueError("NereidForge requires a PostgreSQL database DSN")

    query = dict(url.query)
    ssl_mode = str(query.pop("sslmode", "prefer")).lower()
    url = url.set(drivername="postgresql+pg8000", query=query)

    ssl_context: ssl.SSLContext | bool | None
    if ssl_mode == "disable":
        ssl_context = False
    elif ssl_mode == "prefer":
        ssl_context = None
    elif ssl_mode == "require":
        ssl_context = True
    elif ssl_mode in {"verify-ca", "verify-full"}:
        context = ssl.create_default_context()
        if ssl_mode == "verify-ca":
            context.check_hostname = False
        ssl_context = context
    else:
        supported = "disable, prefer, require, verify-ca, verify-full"
        raise ValueError(
            f"Unsupported database sslmode {ssl_mode!r} for pg8000; "
            f"supported modes: {supported}"
        )

    return url, {"ssl_context": ssl_context}


_engine_url, _engine_connect_args = _engine_parameters(config.database.dsn)
engine = create_engine(
    _engine_url,
    connect_args=_engine_connect_args,
    pool_size=10,
    max_overflow=5,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
