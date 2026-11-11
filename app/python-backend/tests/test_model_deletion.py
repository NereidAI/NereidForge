# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Model deletion enforces platform identity, serving safety, and Hub consistency."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.helpers import get_svc
from app.api.routers.models import router
from app.clients.hub import HubClient
from app.core.auth import Claims, require_auth
from app.db.base import Base
from app.models.orm import Model as ModelORM
from app.models.orm import ModelVersion as ModelVersionORM
from app.models.orm import User
from app.services.admin import AdminService
from app.services.models import ModelService


@compiles(JSONB, "sqlite")
def _sqlite_jsonb(element, compiler, **kw):
    """Compile Postgres JSONB as SQLite JSON for the isolated deletion database."""
    return "JSON"


@pytest.fixture
def db() -> Session:
    """Provide a fresh ORM database for every authorization and lifecycle case."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _register_now(dbapi_connection, _record):
        dbapi_connection.create_function("now", 0, lambda: datetime.now().isoformat())

    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()


class _Hub:
    """Expose an empty active Hub while recording destructive lifecycle calls."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.fail = fail

    def enabled(self) -> bool:
        """Enable the archival path without changing model lookup's PG fallback."""
        return True

    def list_models_detailed(self) -> list[dict]:
        """Force route lookup to use the local test model."""
        return []

    def archive_model(self, name: str) -> None:
        """Record model archival or emulate an unavailable Hub."""
        self.calls.append(("model", name))
        if self.fail:
            raise RuntimeError("hub unavailable")

    def archive_model_version(self, model_name: str, version: str) -> None:
        """Record version archival or emulate an unavailable Hub."""
        self.calls.append(("version", model_name, version))
        if self.fail:
            raise RuntimeError("hub unavailable")


def _user(
    db: Session,
    *,
    username: str = "admin",
    tenant_id: str = "default",
    roles: list[str] | None = None,
    active: bool = True,
    external: bool = False,
) -> User:
    """Create a current user row independent of the claims presented by a request."""
    user = User(
        id=f"usr-{username}-{tenant_id}",
        username=username,
        tenant_id=tenant_id,
        roles=roles if roles is not None else ["admin", "user"],
        is_active=active,
        external_id="identity-subject" if external else "",
        external_issuer="https://identity.example" if external else "",
    )
    db.add(user)
    db.commit()
    return user


def _model(
    db: Session,
    *,
    tenant_id: str = "default",
    status: str = "ready",
    metadata: dict | None = None,
) -> tuple[ModelORM, ModelVersionORM, ModelVersionORM]:
    """Create one model with an offline version and a configurable target version."""
    model = ModelORM(
        id="model-id", tenant_id=tenant_id, name="demo", description="",
        framework="onnx", task_type="detection", created_by="admin",
    )
    offline = ModelVersionORM(
        id="mv-offline", model_id=model.id, tenant_id=tenant_id, version="offline",
        description="", file_path="s3://nereidforge/models/demo/offline/model.onnx",
        size_bytes=1, metadata_={}, status="ready", artifacts=[], created_by="admin",
    )
    target = ModelVersionORM(
        id="mv-target", model_id=model.id, tenant_id=tenant_id, version="target",
        description="", file_path="s3://nereidforge/models/demo/target/model.onnx",
        size_bytes=1, metadata_=metadata or {}, status=status, artifacts=[], created_by="admin",
    )
    db.add_all([model, offline, target])
    db.commit()
    return model, offline, target


def _client(
    db: Session, user: User, hub: _Hub, *,
    token_roles: list[str] | None = None, raise_server_exceptions: bool = True,
) -> TestClient:
    """Mount the real delete routes with authenticated claims and scoped real services."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    claims = Claims(
        subject=user.id,
        issuer="nereidforge",
        tenant_id=user.tenant_id,
        roles=token_roles if token_roles is not None else list(user.roles or []),
    )
    services = SimpleNamespace(
        admin=AdminService(db),
        model=ModelService(db, SimpleNamespace(), hub=hub),
    )
    app.dependency_overrides[require_auth] = lambda: claims
    app.dependency_overrides[get_svc] = lambda: services
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


@pytest.mark.parametrize(
    "user_kwargs,token_roles",
    [
        ({"username": "tenant-admin", "tenant_id": "acme"}, None),
        ({"username": "operator"}, None),
        ({"username": "admin", "external": True}, None),
        ({"username": "admin", "active": False}, None),
        ({"username": "admin", "roles": ["user"]}, ["admin", "user"]),
    ],
)
def test_only_the_enabled_local_default_admin_can_delete(
    db, user_kwargs, token_roles,
) -> None:
    """Tenant, operator, federated, disabled, and revoked admins all receive 403."""
    user = _user(db, **user_kwargs)
    _model(db, tenant_id=user.tenant_id)
    hub = _Hub()
    client = _client(db, user, hub, token_roles=token_roles)
    headers = {"X-Tenant-Id": user.tenant_id}

    version_response = client.delete("/api/v1/models/demo/versions/target", headers=headers)
    model_response = client.delete("/api/v1/models/demo", headers=headers)

    assert version_response.status_code == 403
    assert model_response.status_code == 403
    assert version_response.json()["detail"] == "platform super administrator privileges required"
    assert db.get(ModelORM, "model-id") is not None
    assert db.get(ModelVersionORM, "mv-target") is not None
    assert hub.calls == []


def test_platform_admin_deletes_one_version_then_the_whole_model(db) -> None:
    """Successful deletion archives Hub records, removes PG rows, and never touches S3."""
    user = _user(db)
    _model(db)
    hub = _Hub()
    client = _client(db, user, hub)

    version_response = client.delete("/api/v1/models/demo/versions/target")
    assert version_response.status_code == 200, version_response.text
    assert version_response.json() == {"deleted": True}
    assert db.get(ModelVersionORM, "mv-target") is None
    assert db.get(ModelORM, "model-id") is not None

    model_response = client.delete("/api/v1/models/demo")
    assert model_response.status_code == 200, model_response.text
    assert model_response.json() == {"deleted": True}
    assert db.get(ModelORM, "model-id") is None
    assert db.scalars(select(ModelVersionORM)).all() == []
    assert hub.calls == [("version", "demo", "target"), ("model", "demo")]


@pytest.mark.parametrize(
    "status,metadata",
    [
        ("deployed", {"deployed": True, "deploy_status": "ready"}),
        ("deploying", {"deployed": False, "deploy_status": "deploying"}),
        ("deployed", {"deployed": True, "deploy_status": "terminating"}),
        ("deployed", {"deployed": True, "deploy_status": "undeploy_failed"}),
    ],
)
def test_live_version_blocks_both_delete_operations_without_partial_cleanup(
    db, status, metadata,
) -> None:
    """Deployed, deploying, terminating, and failed-undeploy versions return 409."""
    user = _user(db)
    _model(db, status=status, metadata=metadata)
    hub = _Hub()
    client = _client(db, user, hub)

    version_response = client.delete("/api/v1/models/demo/versions/target")
    model_response = client.delete("/api/v1/models/demo")

    assert version_response.status_code == 409
    assert model_response.status_code == 409
    assert "undeploy them before deletion" in version_response.json()["detail"]
    assert db.get(ModelVersionORM, "mv-target") is not None
    assert db.get(ModelVersionORM, "mv-offline") is not None
    assert db.get(ModelORM, "model-id") is not None
    assert hub.calls == []


def test_failed_initial_deploy_does_not_block_version_deletion(db) -> None:
    """A failed rollout has no live service and remains eligible for registry cleanup."""
    user = _user(db)
    _model(db, status="deploy_failed", metadata={"deployed": False, "deploy_status": "failed"})
    hub = _Hub()
    client = _client(db, user, hub)

    response = client.delete("/api/v1/models/demo/versions/target")

    assert response.status_code == 200, response.text
    assert db.get(ModelVersionORM, "mv-target") is None


def test_hub_failure_cannot_report_success_or_remove_local_records(db) -> None:
    """Archival errors propagate instead of leaving an active Hub record behind."""
    user = _user(db)
    _model(db)
    hub = _Hub(fail=True)
    client = _client(db, user, hub, raise_server_exceptions=False)

    response = client.delete("/api/v1/models/demo")

    assert response.status_code == 500
    assert db.get(ModelORM, "model-id") is not None
    assert db.get(ModelVersionORM, "mv-target") is not None
    assert db.get(ModelVersionORM, "mv-offline") is not None


def test_hub_listing_omits_archived_versions() -> None:
    """Soft-deleted Hub versions cannot reappear after their PG row is removed."""
    client = SimpleNamespace(
        get_registered_models=lambda: [SimpleNamespace(
            name="demo", description="", external_id="", state="LIVE",
            owner="admin", create_time_since_epoch=1,
        )],
        get_model_versions=lambda _name: [
            SimpleNamespace(name="active", state="ModelVersionState.LIVE"),
            SimpleNamespace(name="deleted", state="ModelVersionState.ARCHIVED"),
        ],
    )
    hub = HubClient()
    hub._client = client

    models = hub.list_models_detailed()

    assert [version["version"] for version in models[0]["versions"]] == ["active"]


def test_delete_routes_have_typed_generated_and_handwritten_openapi() -> None:
    """Both API contracts expose success, authorization, and live-service responses."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    generated = app.openapi()
    handwritten = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "docs/api/openapi.yaml").read_text()
    )

    for path in ("/models/{model_name}", "/models/{model_name}/versions/{version}"):
        generated_operation = generated["paths"][f"/api/v1{path}"]["delete"]
        manual_operation = handwritten["paths"][path]["delete"]
        for operation in (generated_operation, manual_operation):
            success = operation["responses"]["200"]["content"]["application/json"]["schema"]
            assert success["$ref"].endswith("/ModelDeleteResponse")
            for status in ("403", "409"):
                error_schema = operation["responses"][status]["content"]["application/json"]["schema"]
                assert error_schema["$ref"].endswith("/ModelDeleteErrorResponse")
