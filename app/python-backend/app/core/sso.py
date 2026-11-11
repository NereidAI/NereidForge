# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Suite SSO for NereidForge (NereidIdentity sso-integration-standard v2).

Two front doors, one onboarding path:

- mode ``ns`` (standard section 4): the browser is sent to NereidSphere, which
  mints a 30-second single-use ticket; the product exchanges it server to
  server (``POST {ns}/api/v1/auth/ticket/verify``) and mints its normal local
  session. No OIDC config is used.
- mode ``oidc`` (standard section 6): the product is a confidential OIDC
  relying party against NereidIdentity (authorization-code + PKCE S256, JWKS
  ID-token verification, state/nonce, RP-initiated logout).

Both modes converge on ``provision_identity`` -> ``issue_session`` so REST
auth, refresh, WebSocket and audit keep working unchanged. Identity only ever
enters through these server-side calls: browser-supplied ``X-User-*`` headers
or identity claims are never trusted, and raw ``X-Tenant-Id`` is ignored while
SSO is active (see ``app.core.auth``).

State/nonce/PKCE records live in the Postgres kv_store table (the API runs
multiple uvicorn workers/replicas) and fall back to a bounded in-process dict
when the DB is briefly unreachable, exactly like the captcha store.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time
from typing import Any, Optional
from urllib.parse import quote, urlencode

import httpx
import jwt

from app.core.auth import LocalSigner
from app.core.config import config
from app.services import kv_store

log = logging.getLogger(__name__)

# Keycloak emits these on every token; they are not suite roles.
_BUILTIN_REALM_ROLES = {"offline_access", "uma_authorization"}
_BUILTIN_REALM_ROLE_PREFIXES = ("default-roles-",)


class SSOError(Exception):
    """SSO failure carrying a stable, browser-facing error code."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code


# ---------------------------------------------------------------------------
# State store (state/nonce/PKCE between /sso/login and /sso/callback)
# ---------------------------------------------------------------------------

_last_error: str = ""


def _state_put(state: str, payload: dict) -> None:
    ttl = config.sso.oidc.state_ttl or 600
    kv_store.kv_put(kv_store.SSO_STATE, state, payload, ttl)


def _state_pop(state: str) -> Optional[dict]:
    """Consume a state record (single use)."""
    return kv_store.kv_pop(kv_store.SSO_STATE, state)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def safe_redirect_path(path: Optional[str], default: str = "/dashboard") -> str:
    """Validate a post-login deep link: same-origin relative path only.

    Rejects absolute URLs, protocol-relative ``//host``, backslashes and
    control characters so a crafted ``to`` cannot become an open redirect.
    """
    if not path:
        return default
    p = path.strip()
    if not p.startswith("/") or p.startswith("//") or "\\" in p:
        return default
    if any(c in p for c in ("\r", "\n", "\t")):
        return default
    if ":" in p.split("?")[0]:
        return default
    return p


def _claim_path(claims: dict, dotted: str) -> Any:
    """Read a dotted claim path (``realm_access.roles``) from a token body."""
    cur: Any = claims
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def map_roles(suite_roles: list[str]) -> list[str]:
    """Map suite roles (user/operator/admin) onto NF roles via config.role_map.

    Unknown suite roles are ignored; when nothing maps, fall back to
    ``default_roles`` or reject with ``oidc_no_role`` if configured to.
    """
    known = [r for r in suite_roles if r in config.sso.oidc.role_map]
    if not known:
        if config.sso.oidc.reject_unknown_roles:
            raise SSOError("oidc_no_role", "no suite role mapped to a product role")
        return list(config.sso.oidc.default_roles)
    mapped: list[str] = []
    for r in known:
        for nf_role in config.sso.oidc.role_map[r]:
            if nf_role not in mapped:
                mapped.append(nf_role)
    return mapped


# Suite product assignment groups: /nereid{product}/{tenant}/{role}. NF only
# consumes its own product token (forge); the tenant segment is NF's own
# ``tnt-…`` id and is deliberately NOT validated as a UUID.
_FORGE_PREFIX = "/nereidforge/"


def parse_forge_groups(groups: list[str]) -> list[tuple[str, str]]:
    """Return ``(tenant_id, role_key)`` for each ``/nereidforge/...`` entry.

    Entries for other products (``/nereidlink``, ``/nereidhorizon``, …),
    malformed paths and empty segments are ignored. The tenant is NF's own
    tenant id (``tnt-…``), NOT a UUID.
    """
    out: list[tuple[str, str]] = []
    for g in groups:
        if not g.startswith(_FORGE_PREFIX):
            continue
        parts = g[len(_FORGE_PREFIX):].split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            continue
        out.append((parts[0], parts[1]))
    return out


def _suite_roles_from_claims(claims: dict) -> list[str]:
    """Extract suite roles from the ID token, dropping Keycloak built-ins."""
    raw = _claim_path(claims, config.sso.oidc.roles_claim)
    roles = [str(r) for r in raw] if isinstance(raw, list) else []
    return [
        r for r in roles
        if r not in _BUILTIN_REALM_ROLES
        and not r.startswith(_BUILTIN_REALM_ROLE_PREFIXES)
    ]


def issue_session(user: dict) -> dict:
    """Mint the normal local session pair for a provisioned SSO user.

    Mirrors ``POST /api/v1/auth/login`` exactly (LocalSigner.sign), so every
    downstream mechanism (REST auth, refresh, WebSocket ``?token=``, audit)
    is unchanged.
    """
    signer = LocalSigner()
    roles = list(user.get("roles") or [])
    return {
        "access_token": signer.sign(user["id"], user["tenant_id"], roles),
        "refresh_token": signer.sign_refresh(user["id"], user["tenant_id"]),
        "token_type": "Bearer",
        "expires_in": signer.ttl,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user.get("email", ""),
            "full_name": user.get("full_name", ""),
            "roles": roles,
            "tenant_id": user["tenant_id"],
        },
    }


# ---------------------------------------------------------------------------
# Status / login entry / logout URLs
# ---------------------------------------------------------------------------

def status() -> dict:
    """SSO status consumed by the login page (hide the entry unless enabled)."""
    mode = config.sso.mode
    enabled = False
    if mode == "oidc":
        o = config.sso.oidc
        enabled = bool(o.issuer and o.client_id and o.client_secret and o.redirect_url)
    elif mode == "ns":
        n = config.sso.ns
        enabled = bool(n.base_url and n.client_secret)
    return {
        "mode": mode,
        "enabled": enabled,
        "login_path": "/api/v1/auth/sso/login",
        "logout_url": logout_url() if enabled else "",
        "last_error": _last_error,
    }


def logout_url() -> str:
    """Browser-followable logout URL chaining the IdP end-session (section 4.4.7)."""
    if config.sso.mode == "ns":
        return f"{config.sso.ns.base_url.rstrip('/')}/auth/logout"
    if config.sso.mode == "oidc" and config.sso.oidc.frontchannel_logout:
        front = (config.sso.oidc.frontend_base_url or config.sso.oidc.redirect_url).rstrip("/")
        base = front.split("/auth/")[0]
        params = {"client_id": config.sso.oidc.client_id}
        if front:
            params["post_logout_redirect_uri"] = f"{base}/login"
        return (
            f"{config.sso.oidc.issuer.rstrip('/')}/protocol/openid-connect/logout"
            f"?{urlencode(params)}"
        )
    return ""


# ---------------------------------------------------------------------------
# Mode oidc: authorization-code + PKCE against NereidIdentity
# ---------------------------------------------------------------------------

_discovery_cache: dict[str, tuple[float, dict]] = {}
_DISCOVERY_TTL = 300


def _discovery() -> dict:
    """Fetch and cache the IdP OIDC discovery document (best effort)."""
    issuer = config.sso.oidc.issuer.rstrip("/")
    if not issuer:
        raise SSOError("not_configured", "sso oidc issuer not configured")
    cached = _discovery_cache.get(issuer)
    if cached and cached[0] > time.time():
        return cached[1]
    url = f"{issuer}/.well-known/openid-configuration"
    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        doc = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise SSOError("oidc_discovery_failed", f"discovery failed: {exc}") from exc
    _discovery_cache[issuer] = (time.time() + _DISCOVERY_TTL, doc)
    return doc


def start_oidc_login(to: str) -> str:
    """Build the IdP authorization URL and remember state/nonce/PKCE."""
    o = config.sso.oidc
    doc = _discovery()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    _state_put(state, {
        "mode": "oidc",
        "nonce": nonce,
        "code_verifier": verifier,
        "to": safe_redirect_path(to),
        "issuer": doc.get("issuer", ""),
    })
    params = {
        "response_type": "code",
        "client_id": o.client_id,
        "redirect_uri": o.redirect_url,
        "scope": " ".join(o.scopes),
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    endpoint = doc.get("authorization_endpoint") or f"{o.issuer.rstrip('/')}/protocol/openid-connect/auth"
    return f"{endpoint}?{urlencode(params)}"


def complete_oidc(code: str, state: str) -> dict:
    """Exchange the code, verify the ID token, return ``{identity, to}``."""
    o = config.sso.oidc
    record = _state_pop(state) if state else None
    if not record or record.get("mode") != "oidc":
        raise SSOError("oidc_invalid_state", "unknown or expired login state")
    doc = _discovery()
    token_endpoint = doc.get("token_endpoint") or f"{o.issuer.rstrip('/')}/protocol/openid-connect/token"
    try:
        resp = httpx.post(
            token_endpoint,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": o.redirect_url,
                "code_verifier": record["code_verifier"],
            },
            auth=(o.client_id, o.client_secret),
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        raise SSOError("oidc_failed", f"token endpoint unreachable: {exc}") from exc
    if resp.status_code != 200:
        raise SSOError("oidc_failed", f"token exchange failed: HTTP {resp.status_code}")
    tokens = resp.json()
    id_token = tokens.get("id_token")
    if not id_token:
        raise SSOError("oidc_failed", "no id_token in token response")

    claims = _verify_id_token(id_token, doc)
    if claims.get("nonce") and claims["nonce"] != record["nonce"]:
        raise SSOError("oidc_failed", "nonce mismatch")

    identity = _identity_from_claims(claims)
    return {"identity": identity, "to": record["to"]}


def _verify_id_token(id_token: str, doc: dict) -> dict:
    """Verify the ID token signature (JWKS) and standard claims via PyJWT."""
    o = config.sso.oidc
    jwks_uri = doc.get("jwks_uri") or f"{o.issuer.rstrip('/')}/protocol/openid-connect/certs"
    try:
        signing_key = jwt.PyJWKClient(jwks_uri, cache_jwk_set=True, lifespan=3600).get_signing_key_from_jwt(id_token)
        return jwt.decode(
            id_token,
            signing_key.key,
            algorithms=[signing_key.algorithm_name or "RS256"],
            audience=o.client_id,
            issuer=doc.get("issuer") or o.issuer,
            leeway=30,
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise SSOError("oidc_failed", f"id_token verification failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise SSOError("oidc_failed", f"jwks fetch failed: {exc}") from exc


def _identity_from_claims(claims: dict) -> dict:
    """Normalize verified ID-token claims into the standard identity shape."""
    email = str(claims.get("email") or "")
    if not email:
        raise SSOError("oidc_no_email", "identity has no email claim")
    name = str(claims.get("name") or "") or str(claims.get("preferred_username") or "")
    return {
        "sub": str(claims.get("sub") or ""),
        "email": email,
        "name": name,
        "preferred_username": str(claims.get("preferred_username") or ""),
        "roles": _suite_roles_from_claims(claims),
    }


# ---------------------------------------------------------------------------
# Mode ns: ticket exchange against NereidSphere
# ---------------------------------------------------------------------------

_NS_ERROR_CODES = {
    "TICKET_INVALID": "ticket_invalid",
    "TICKET_EXPIRED": "ticket_expired",
    "TICKET_AUDIENCE": "ticket_audience",
    "BAD_REQUEST": "ticket_invalid",
}


def verify_ticket(ticket: str) -> dict:
    """Exchange an NS ticket server to server; return ``{identity, to}``."""
    n = config.sso.ns
    url = f"{n.base_url.rstrip('/')}/api/v1/auth/ticket/verify"
    try:
        resp = httpx.post(
            url,
            json={"ticket": ticket},
            headers={
                "X-NS-Client-Id": n.product_id,
                "X-NS-Client-Secret": n.client_secret,
            },
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        raise SSOError("ns_unreachable", f"nereidsphere unreachable: {exc}") from exc
    if resp.status_code == 401:
        raise SSOError("ns_unauthorized", "nereidsphere rejected the product credentials")
    if resp.status_code != 200:
        code = ""
        try:
            code = str((resp.json().get("error") or {}).get("code") or "")
        except Exception:  # noqa: BLE001
            code = ""
        raise SSOError(_NS_ERROR_CODES.get(code, "ticket_invalid"), f"ticket verify failed (HTTP {resp.status_code})")
    body = resp.json()
    identity = body.get("identity") or {}
    if not identity.get("sub"):
        raise SSOError("ticket_invalid", "ticket response carried no identity")
    return {
        "identity": {
            "sub": str(identity.get("sub") or ""),
            "email": str(identity.get("email") or ""),
            "name": str(identity.get("name") or ""),
            "preferred_username": str(identity.get("preferred_username") or ""),
            "roles": [str(r) for r in (identity.get("roles") or [])],
            # Suite product assignments (/nereidforge/{tenant}/{role}); NS omits
            # the key entirely when the user has none, so treat absent == empty.
            "groups": [str(g) for g in (identity.get("groups") or [])],
        },
        "to": safe_redirect_path(body.get("to")),
    }


# ---------------------------------------------------------------------------
# Entry point used by both front doors
# ---------------------------------------------------------------------------

def login_url(to: Optional[str]) -> str:
    """Backend 302 target for the SSO entry (both modes).

    mode ns  -> NereidSphere enter URL (ticket is minted there and posted back
                to the public callback).
    mode oidc-> IdP authorization endpoint; the IdP returns to the registered
                ``/auth/callback`` SPA route which POSTs code+state back.
    """
    target = safe_redirect_path(to)
    if config.sso.mode == "ns":
        n = config.sso.ns
        return f"{n.base_url.rstrip('/')}/auth/enter/{n.product_id}?to={quote(target, safe='')}"
    if config.sso.mode == "oidc":
        return start_oidc_login(target)
    raise SSOError("not_configured", "sso is not enabled")


def provision_identity(admin: Any, identity: dict, issuer: str) -> dict:
    """JIT-provision or refresh the local user for a verified identity.

    Suite SSO (NS ticket) carries the product assignments as
    ``/nereidforge/{tenant}/{role}`` groups. When at least one is present it
    defines the local tenant binding and roles (single-tenant: first wins, extra
    assignments are logged and ignored). Without any Forge group the legacy
    realm-role mapping into ``sso.jit_tenant`` is used unchanged, so revoking a
    group in NS reduces access on the next login.
    """
    assignments = parse_forge_groups(identity.get("groups") or [])
    if assignments:
        tenant_id, role_key = assignments[0]
        if len(assignments) > 1:
            log.warning(
                "sso: multiple forge assignments for %s; NF is single-tenant, "
                "applying %s and ignoring %s",
                identity.get("preferred_username"), assignments[0], assignments[1:],
            )
        if role_key not in config.sso.oidc.role_map:
            raise SSOError("sso_no_role", f"unknown forge role key {role_key!r}")
        return admin.provision_sso_user(
            sub=identity["sub"],
            issuer=issuer,
            username=identity.get("preferred_username") or "",
            email=identity.get("email") or "",
            full_name=identity.get("name") or "",
            roles=map_roles([role_key]),
            tenant_id=tenant_id,
        )
    # Fallback: legacy realm-role mapping into sso.jit_tenant (unchanged).
    roles = map_roles(identity.get("roles") or [])
    return admin.provision_sso_user(
        sub=identity["sub"],
        issuer=issuer,
        username=identity.get("preferred_username") or "",
        email=identity.get("email") or "",
        full_name=identity.get("name") or "",
        roles=roles,
        tenant_id=config.sso.jit_tenant or "default",
    )


def record_error(code: str) -> None:
    """Remember the last SSO failure for the status endpoint (diagnostics)."""
    global _last_error
    _last_error = code
    if code:
        log.warning("sso error: %s", code)
