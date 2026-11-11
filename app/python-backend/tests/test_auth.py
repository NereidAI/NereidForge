# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Authentication route regressions for token refresh and user identity."""

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from app.api.routers.auth import RefreshRequest, refresh
from app.core.auth import LocalSigner
from app.core.config import config
from app.db.base import Base
from app.services.admin import AdminService


@compiles(JSONB, "sqlite")
def _sqlite_jsonb(element, compiler, **kw):
    """Compile Postgres JSONB as SQLite JSON for the test session."""
    return "JSON"


@pytest.fixture
def db() -> Session:
    """Provide an isolated database containing the real user schema."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _register_now(dbapi_connection, _record):
        dbapi_connection.create_function("now", 0, lambda: datetime.now().isoformat())

    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()


def test_refresh_resolves_the_user_id_stored_in_the_token(db, monkeypatch):
    """A refresh token carrying User.id must not be queried as a username."""
    monkeypatch.setattr(config.auth, "jwt_secret", "test-secret-with-at-least-32-bytes")
    admin = AdminService(db)
    user = admin.create_user(
        username="alice",
        email="alice@example.com",
        full_name="Alice",
        password="secret",
        tenant_id="default",
        roles=["user"],
    )
    assert user["id"] != user["username"]
    refresh_token = LocalSigner().sign_refresh(user["id"], user["tenant_id"])

    result = asyncio.run(refresh(
        RefreshRequest(refresh_token=refresh_token),
        SimpleNamespace(admin=admin),
    ))

    claims = LocalSigner().parse(result["access_token"])
    assert claims.subject == user["id"]
    assert claims.tenant_id == "default"
    assert claims.roles == ["user"]
    assert set(result) == {"access_token", "token_type", "expires_in"}
