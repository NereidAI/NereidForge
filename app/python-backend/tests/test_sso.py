# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Suite SSO tests: role mapping, redirect validation, state store, ticket
exchange error mapping, OIDC callback guards and JIT provisioning.

Uses an in-memory SQLite session (JSONB compiled as JSON), same pattern as
tests/test_base_models.py, and monkeypatched httpx for the NS/NI calls.
"""

from datetime import datetime

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.api.routers import integration
from app.db.base import Base
from app.core import sso
from app.core.config import config
from app.models.orm import Tenant
from app.services.admin import AdminService


@compiles(JSONB, "sqlite")
def _sqlite_jsonb(element, compiler, **kw):
    """Compile Postgres JSONB as SQLite JSON for the test session."""
    return "JSON"


@pytest.fixture
def db() -> Session:
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _register_now(dbapi_connection, _record):
        # The models use Postgres server_default now(); SQLite has no now().
        dbapi_connection.create_function("now", 0, lambda: datetime.now().isoformat())

    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()


@pytest.fixture(autouse=True)
def _reset_sso_state(monkeypatch):
    """Isolate module-level caches and point kv_store at a fresh sqlite DB."""
    from sqlalchemy.orm import sessionmaker

    from app.services import kv_store

    kv_engine = create_engine("sqlite:///:memory:")

    @event.listens_for(kv_engine, "connect")
    def _register_now_kv(dbapi_connection, _record):
        dbapi_connection.create_function("now", 0, lambda: datetime.now().isoformat())

    Base.metadata.create_all(kv_engine)
    kv_factory = sessionmaker(bind=kv_engine)
    monkeypatch.setattr(kv_store, "SessionLocal", kv_factory)
    kv_store._mem.clear()
    sso._discovery_cache.clear()
    yield
    kv_store._mem.clear()
    sso._discovery_cache.clear()


# --- safe_redirect_path ---------------------------------------------------

@pytest.mark.parametrize("bad", [
    None, "", "https://evil.example/x", "//evil.example/x", "/\\evil",
    "javascript:alert(1)", "/x\r\nSet-Cookie: a=b", "dashboard",
])
def test_safe_redirect_path_rejects(bad):
    assert sso.safe_redirect_path(bad) == "/dashboard"


@pytest.mark.parametrize("good", ["/", "/dashboard", "/models/42?tab=x", "/auth/callback"])
def test_safe_redirect_path_accepts(good):
    assert sso.safe_redirect_path(good) == good


def test_safe_redirect_path_custom_default():
    assert sso.safe_redirect_path("http://evil", "/home") == "/home"


# --- role mapping ---------------------------------------------------------

def test_map_roles_admin_and_user():
    assert set(sso.map_roles(["admin", "user"])) == {"admin", "user"}


def test_map_roles_unknown_falls_back_to_default():
    assert sso.map_roles(["viewer"]) == config.sso.oidc.default_roles


def test_map_roles_unknown_rejects_when_configured(monkeypatch):
    monkeypatch.setattr(config.sso.oidc, "reject_unknown_roles", True)
    with pytest.raises(sso.SSOError) as exc:
        sso.map_roles([])
    assert exc.value.code == "oidc_no_role"


def test_map_roles_honours_configured_map(monkeypatch):
    monkeypatch.setattr(config.sso.oidc, "role_map", {"operator": ["user"]})
    assert sso.map_roles(["operator"]) == ["user"]


def test_suite_roles_drop_keycloak_builtins(monkeypatch):
    claims = {"realm_access": {"roles": [
        "offline_access", "uma_authorization", "default-roles-nereidai", "admin",
    ]}}
    monkeypatch.setattr(config.sso.oidc, "roles_claim", "realm_access.roles")
    assert sso._suite_roles_from_claims(claims) == ["admin"]


# --- status ---------------------------------------------------------------

def test_status_local_disabled(monkeypatch):
    monkeypatch.setattr(config.sso, "mode", "local")
    st = sso.status()
    assert st["mode"] == "local" and st["enabled"] is False


def test_status_oidc_requires_full_config(monkeypatch):
    monkeypatch.setattr(config.sso, "mode", "oidc")
    o = config.sso.oidc
    monkeypatch.setattr(o, "issuer", "https://sso.example/realms/nereidai")
    monkeypatch.setattr(o, "client_id", "nereidforge")
    monkeypatch.setattr(o, "client_secret", "")
    monkeypatch.setattr(o, "redirect_url", "https://nereidforge.example/auth/callback")
    assert sso.status()["enabled"] is False  # secret missing
    monkeypatch.setattr(o, "client_secret", "s3cr3t")
    assert sso.status()["enabled"] is True


def test_status_ns_requires_secret(monkeypatch):
    monkeypatch.setattr(config.sso, "mode", "ns")
    n = config.sso.ns
    monkeypatch.setattr(n, "base_url", "https://nereidsphere.example")
    monkeypatch.setattr(n, "client_secret", "")
    assert sso.status()["enabled"] is False
    monkeypatch.setattr(n, "client_secret", "shared")
    assert sso.status()["enabled"] is True


# --- state store ----------------------------------------------------------

def test_state_store_is_single_use():
    sso._state_put("abc", {"to": "/x"})
    assert sso._state_pop("abc") == {"to": "/x"}
    assert sso._state_pop("abc") is None


def test_state_store_rejects_expired():
    import json as _json

    from app.services import kv_store

    kv_store._mem[(kv_store.SSO_STATE, "old")] = (0.0, _json.dumps({"to": "/x"}))
    assert sso._state_pop("old") is None


# --- ns ticket exchange ---------------------------------------------------

def _resp(status: int, payload: dict) -> httpx.Response:
    return httpx.Response(status_code=status, json=payload, request=httpx.Request("POST", "http://x"))


def test_verify_ticket_success(monkeypatch):
    def fake_post(url, **kw):
        assert kw["headers"]["X-NS-Client-Id"] == config.sso.ns.product_id
        return _resp(200, {"identity": {
            "sub": "s1", "email": "a@b.c", "name": "A",
            "preferred_username": "alice", "roles": ["admin"],
        }, "to": "/devices/9"})
    monkeypatch.setattr(sso.httpx, "post", fake_post)
    out = sso.verify_ticket("ticket-1")
    assert out["identity"]["sub"] == "s1"
    assert out["to"] == "/devices/9"


def test_verify_ticket_maps_error_codes(monkeypatch):
    monkeypatch.setattr(sso.httpx, "post", lambda url, **kw: _resp(
        400, {"error": {"code": "TICKET_EXPIRED", "message": "expired"}}))
    with pytest.raises(sso.SSOError) as exc:
        sso.verify_ticket("t")
    assert exc.value.code == "ticket_expired"


def test_verify_ticket_unauthorized(monkeypatch):
    monkeypatch.setattr(sso.httpx, "post", lambda url, **kw: _resp(401, {}))
    with pytest.raises(sso.SSOError) as exc:
        sso.verify_ticket("t")
    assert exc.value.code == "ns_unauthorized"


def test_verify_ticket_unreachable(monkeypatch):
    def boom(url, **kw):
        raise httpx.ConnectError("no route")
    monkeypatch.setattr(sso.httpx, "post", boom)
    with pytest.raises(sso.SSOError) as exc:
        sso.verify_ticket("t")
    assert exc.value.code == "ns_unreachable"


# --- oidc callback guards -------------------------------------------------

def test_complete_oidc_invalid_state():
    with pytest.raises(sso.SSOError) as exc:
        sso.complete_oidc("code", "unknown-state")
    assert exc.value.code == "oidc_invalid_state"


# --- logout url -----------------------------------------------------------

def test_logout_url_oidc_includes_end_session(monkeypatch):
    monkeypatch.setattr(config.sso, "mode", "oidc")
    o = config.sso.oidc
    monkeypatch.setattr(o, "issuer", "https://sso.example/realms/nereidai")
    monkeypatch.setattr(o, "client_id", "nereidforge")
    monkeypatch.setattr(o, "redirect_url", "https://nereidforge.example/auth/callback")
    monkeypatch.setattr(o, "frontend_base_url", "https://nereidforge.example")
    monkeypatch.setattr(o, "frontchannel_logout", True)
    url = sso.logout_url()
    assert url.startswith("https://sso.example/realms/nereidai/protocol/openid-connect/logout?")
    assert "client_id=nereidforge" in url
    assert "post_logout_redirect_uri=https%3A%2F%2Fnereidforge.example%2Flogin" in url


def test_logout_url_ns(monkeypatch):
    monkeypatch.setattr(config.sso, "mode", "ns")
    monkeypatch.setattr(config.sso.ns, "base_url", "https://nereidsphere.example")
    assert sso.logout_url() == "https://nereidsphere.example/auth/logout"


# --- JIT provisioning -----------------------------------------------------

def test_provision_creates_user_keyed_by_sub(db):
    admin = AdminService(db)
    u = admin.provision_sso_user(
        sub="sub-1", issuer="iss-1", username="alice", email="alice@x.io",
        full_name="Alice", roles=["admin", "user"], tenant_id="default",
    )
    assert u["username"] == "alice"
    assert u["roles"] == ["admin", "user"]

    again = admin.provision_sso_user(
        sub="sub-1", issuer="iss-1", username="alice", email="alice@x.io",
        full_name="Alice B", roles=["user"], tenant_id="default",
    )
    assert again["id"] == u["id"]          # same identity -> same local user
    assert again["roles"] == ["user"]      # roles refreshed on every login
    assert again["full_name"] == "Alice B"


def test_provision_separates_issuers(db):
    admin = AdminService(db)
    a = admin.provision_sso_user(
        sub="s", issuer="iss-a", username="bob", email="b@x.io",
        full_name="", roles=["user"], tenant_id="default")
    b = admin.provision_sso_user(
        sub="s", issuer="iss-b", username="bob", email="b@x.io",
        full_name="", roles=["user"], tenant_id="default")
    assert a["id"] != b["id"]
    assert b["username"] == "bob-2"  # never hijacks the existing username


def test_provision_username_fallback(db):
    admin = AdminService(db)
    u = admin.provision_sso_user(
        sub="abcdef1234567890", issuer="iss", username="",
        email="+weird@x.io", full_name="", roles=["user"], tenant_id="default")
    assert u["username"]


def test_provision_disabled_user_stays_disabled(db):
    admin = AdminService(db)
    u = admin.provision_sso_user(
        sub="s2", issuer="iss", username="carol", email="c@x.io",
        full_name="", roles=["user"], tenant_id="default")
    admin.update_user(u["id"], {"is_active": False})
    with pytest.raises(sso.SSOError) as exc:
        admin.provision_sso_user(
            sub="s2", issuer="iss", username="carol", email="c@x.io",
            full_name="", roles=["user"], tenant_id="default")
    assert exc.value.code == "account_disabled"


# --- session minting ------------------------------------------------------

def test_issue_session_matches_login_shape(monkeypatch):
    monkeypatch.setattr(config.auth, "jwt_secret", "test-secret")
    user = {"id": "usr_1", "username": "alice", "email": "a@x.io",
            "full_name": "Alice", "roles": ["admin"], "tenant_id": "default"}
    out = sso.issue_session(user)
    assert set(out) == {"access_token", "refresh_token", "token_type", "expires_in", "user"}
    from app.core.auth import LocalSigner
    claims = LocalSigner().parse(out["access_token"])
    assert claims.subject == "usr_1"
    assert claims.roles == ["admin"]


def test_login_url_ns_builds_enter_url(monkeypatch):
    monkeypatch.setattr(config.sso, "mode", "ns")
    monkeypatch.setattr(config.sso.ns, "base_url", "https://nereidsphere.example")
    monkeypatch.setattr(config.sso.ns, "product_id", "forge")
    assert sso.login_url("/dashboard") == (
        "https://nereidsphere.example/auth/enter/forge?to=%2Fdashboard")


# --- forge group parsing --------------------------------------------------

def test_parse_forge_groups_only_forge_entries():
    groups = [
        "/nereidforge/tnt-3f9a-1750000000000/admin",
        "/nereidforge/tnt-77bd-1750000000001/user",
        "/nereidlink/6f1e2b7c-9a4d-4c3e-8b21-0d5f7a9c1e42/admins",
        "/nereidhorizon/h/admins",
    ]
    assert sso.parse_forge_groups(groups) == [
        ("tnt-3f9a-1750000000000", "admin"),
        ("tnt-77bd-1750000000001", "user"),
    ]


def test_parse_forge_groups_drops_malformed_and_empty_segments():
    assert sso.parse_forge_groups([
        "/nereidforge/",              # empty tenant+role
        "/nereidforge/tnt-a",         # missing role
        "/nereidforge/tnt-a/user/x",  # too many segments
        "/user",                      # unrelated path
        "",                           # empty entry
    ]) == []


def test_parse_forge_groups_accepts_non_uuid_tenant():
    # Forge tenant ids are tnt-... strings, never validated as UUIDs.
    assert sso.parse_forge_groups(["/nereidforge/tnt-abc-1/admin"]) == [
        ("tnt-abc-1", "admin")]


# --- groups in the ticket exchange ----------------------------------------

def test_verify_ticket_surfaces_groups(monkeypatch):
    monkeypatch.setattr(sso.httpx, "post", lambda url, **kw: _resp(200, {"identity": {
        "sub": "s1", "preferred_username": "alice", "roles": ["user"],
        "groups": ["/nereidforge/tnt-a/admin", "/nereidlink/u/admins"],
    }}))
    assert sso.verify_ticket("t")["identity"]["groups"] == [
        "/nereidforge/tnt-a/admin", "/nereidlink/u/admins"]


def test_verify_ticket_without_groups_is_empty(monkeypatch):
    # NS uses omitempty: no groups key means "no assignments", never null.
    monkeypatch.setattr(sso.httpx, "post", lambda url, **kw: _resp(200, {"identity": {
        "sub": "s1", "preferred_username": "alice", "roles": ["user"]}}))
    assert sso.verify_ticket("t")["identity"]["groups"] == []


# --- provision_identity assignment semantics ------------------------------

def _tenant(db, tid: str, name: str, active: bool = True) -> None:
    db.add(Tenant(id=tid, name=name, display_name=name.capitalize(), is_active=active))
    db.commit()


def _identity(**over):
    base = {
        "sub": "s-1", "preferred_username": "alice", "email": "a@b.c",
        "name": "Alice", "roles": ["user"], "groups": [],
    }
    base.update(over)
    return base


def test_provision_identity_admin_group_binds_tenant_and_roles(db):
    _tenant(db, "tnt-acme", "acme")
    u = sso.provision_identity(
        AdminService(db),
        _identity(groups=["/nereidforge/tnt-acme/admin", "/nereidlink/x/admins"]),
        "ns:test",
    )
    assert u["tenant_id"] == "tnt-acme"
    assert set(u["roles"]) == {"admin", "user"}


@pytest.mark.parametrize("role,expected", [
    ("admin", {"admin", "user"}),
    ("operator", {"admin", "user"}),
    ("user", {"user"}),
])
def test_provision_identity_role_keys(db, role, expected):
    _tenant(db, "tnt-acme", "acme")
    u = sso.provision_identity(
        AdminService(db), _identity(groups=[f"/nereidforge/tnt-acme/{role}"]), "ns:test")
    assert set(u["roles"]) == expected


def test_provision_identity_multiple_assignments_first_wins(db, caplog):
    _tenant(db, "tnt-a", "aa")
    _tenant(db, "tnt-b", "bb")
    with caplog.at_level("WARNING"):
        u = sso.provision_identity(
            AdminService(db),
            _identity(groups=["/nereidforge/tnt-a/admin", "/nereidforge/tnt-b/user"]),
            "ns:test",
        )
    assert u["tenant_id"] == "tnt-a"
    assert "multiple forge assignments" in caplog.text


def test_provision_identity_unknown_role_key_errors(db):
    _tenant(db, "tnt-a", "aa")
    with pytest.raises(sso.SSOError) as exc:
        sso.provision_identity(
            AdminService(db), _identity(groups=["/nereidforge/tnt-a/wizard"]), "ns:test")
    assert exc.value.code == "sso_no_role"


def test_provision_identity_no_groups_uses_realm_role_fallback(db):
    u = sso.provision_identity(
        AdminService(db), _identity(roles=["admin"]), "ns:test")
    assert u["tenant_id"] == config.sso.jit_tenant
    assert set(u["roles"]) == {"admin", "user"}


def test_provision_identity_other_product_groups_fall_back(db):
    u = sso.provision_identity(
        AdminService(db),
        _identity(roles=["user"], groups=["/nereidlink/u/admins", "/nereidhorizon/h/x"]),
        "ns:test",
    )
    assert u["tenant_id"] == config.sso.jit_tenant
    assert u["roles"] == ["user"]


def test_provision_identity_unknown_tenant_falls_back(db):
    # Stale assignment (tenant deleted in NF) must not break login.
    u = sso.provision_identity(
        AdminService(db), _identity(groups=["/nereidforge/tnt-gone/user"]), "ns:test")
    assert u["tenant_id"] == config.sso.jit_tenant
    assert u["roles"] == ["user"]


def test_relogin_changed_assignment_refreshes_tenant_and_roles(db):
    _tenant(db, "tnt-a", "aa")
    _tenant(db, "tnt-b", "bb")
    admin = AdminService(db)
    first = sso.provision_identity(
        admin, _identity(groups=["/nereidforge/tnt-a/admin"]), "ns:test")
    again = sso.provision_identity(
        admin, _identity(groups=["/nereidforge/tnt-b/user"]), "ns:test")
    assert again["id"] == first["id"]
    assert again["tenant_id"] == "tnt-b"
    assert again["roles"] == ["user"]


def test_relogin_frozen_when_reconcile_disabled(db, monkeypatch):
    monkeypatch.setattr(config.sso.ns, "reconcile_memberships", False)
    _tenant(db, "tnt-a", "aa")
    _tenant(db, "tnt-b", "bb")
    admin = AdminService(db)
    first = sso.provision_identity(
        admin, _identity(groups=["/nereidforge/tnt-a/admin"]), "ns:test")
    again = sso.provision_identity(
        admin, _identity(groups=["/nereidforge/tnt-b/user"]), "ns:test")
    assert again["id"] == first["id"]
    assert again["tenant_id"] == "tnt-a"   # frozen
    assert set(again["roles"]) == {"admin", "user"}


def test_provision_sso_user_refreshes_username(db):
    _tenant(db, "tnt-a", "aa")
    admin = AdminService(db)
    first = sso.provision_identity(
        admin, _identity(preferred_username="alice",
                         groups=["/nereidforge/tnt-a/user"]), "ns:test")
    again = sso.provision_identity(
        admin, _identity(preferred_username="alicia",
                         groups=["/nereidforge/tnt-a/user"]), "ns:test")
    assert again["id"] == first["id"]
    assert again["username"] == "alicia"


# --- NS integration tenant-list endpoint ----------------------------------

def _integration_request(headers: dict) -> Request:
    """Build a minimal Starlette request carrying the given headers."""
    scope = {
        "type": "http", "method": "GET",
        "path": "/v1/integration/ns/tenants",
        "headers": [(k.lower().encode(), str(v).encode()) for k, v in headers.items()],
    }
    return Request(scope)


def test_ns_tenants_list_for_admin_and_operator(db, monkeypatch):
    monkeypatch.setattr(config.sso.ns, "client_secret", "shared")
    _tenant(db, "tnt-acme", "acme")
    _tenant(db, "tnt-old", "old", active=False)  # inactive tenants are hidden
    for roles in ("admin,user", "operator"):
        out = integration.ns_tenants(
            _integration_request({"X-NS-Integration-Secret": "shared",
                                  "X-User-Roles": roles}), db)
        assert out == [{"tenant_id": "tnt-acme", "tenant_name": "Acme"}]


@pytest.mark.parametrize("headers", [
    {},                                              # no header at all
    {"X-NS-Integration-Secret": "wrong"},           # wrong secret
    {"X-User-Roles": "admin,user"},                 # spoofed roles, no secret
    {"X-NS-Integration-Secret": "shared", "X-User-Roles": "user"},  # weak role
])
def test_ns_tenants_rejected(db, monkeypatch, headers):
    monkeypatch.setattr(config.sso.ns, "client_secret", "shared")
    _tenant(db, "tnt-acme", "acme")
    with pytest.raises(HTTPException) as exc:
        integration.ns_tenants(_integration_request(headers), db)
    # Missing/wrong/spoofed secret -> 401; valid secret but weak role -> 403.
    assert exc.value.status_code == (403 if headers.get("X-NS-Integration-Secret") == "shared"
                                     and "X-User-Roles" in headers else 401)


def test_ns_tenants_unconfigured_secret_fails_closed(db, monkeypatch):
    monkeypatch.setattr(config.sso.ns, "client_secret", "")
    with pytest.raises(HTTPException) as exc:
        integration.ns_tenants(
            _integration_request({"X-NS-Integration-Secret": "", "X-User-Roles": "admin"}), db)
    assert exc.value.status_code == 401
