# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Authentication layer: local JWT signer, Dev middleware, OIDC, and tenant context.

Mirrors the Go auth package (internal/auth/):
- LocalSigner: HS256 JWT signing/parsing with access + refresh tokens
- DevMiddleware: bypass auth in dev (injects default tenant + admin roles)
- tenant resolution: claims tenant -> X-Tenant-Id header -> "default"
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import config

bearer_scheme = HTTPBearer(auto_error=False)

REFRESH_ISSUER_SUFFIX = "-refresh"


@dataclass
class Claims:
    """Parsed JWT claims."""
    subject: str
    issuer: str
    tenant_id: str
    roles: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


class LocalSigner:
    """HS256 JWT signer for access and refresh tokens."""

    def __init__(self) -> None:
        self.secret = config.auth.jwt_secret
        self.issuer = config.auth.issuer or "nereidforge"
        self.audience = "nereidforge-api"
        self.ttl = config.auth.access_ttl or 12 * 3600
        self.refresh_ttl = config.auth.refresh_ttl or 7 * 24 * 3600

    def sign(self, user_id: str, tenant_id: str, roles: list[str], ttl: Optional[int] = None) -> str:
        """Sign an access token (sub=user_id, tenant_id + roles claims)."""
        now = int(time.time())
        payload: dict[str, Any] = {
            "sub": user_id,
            "iss": self.issuer,
            "aud": self.audience,
            "iat": now,
            "exp": now + (ttl or self.ttl),
            "tenant_id": tenant_id,
            "roles": roles,
        }
        return jwt.encode(payload, self.secret, algorithm="HS256")

    def sign_refresh(self, user_id: str, tenant_id: str) -> str:
        """Sign a refresh token (issuer suffixed with -refresh, no roles)."""
        now = int(time.time())
        payload: dict[str, Any] = {
            "sub": user_id,
            "iss": self.issuer + REFRESH_ISSUER_SUFFIX,
            "aud": self.audience,
            "iat": now,
            "exp": now + self.refresh_ttl,
            "tenant_id": tenant_id,
            "roles": [],
        }
        return jwt.encode(payload, self.secret, algorithm="HS256")

    def parse(self, token: str) -> Claims:
        """Parse and validate a token, returning Claims."""
        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=["HS256"],
                audience=self.audience,
                leeway=30,
            )
        except jwt.PyJWTError as e:
            raise ValueError(f"invalid token: {e}")
        return Claims(
            subject=str(payload.get("sub", "")),
            issuer=str(payload.get("iss", "")),
            tenant_id=str(payload.get("tenant_id", "default")),
            roles=list(payload.get("roles", []) or []),
            groups=list(payload.get("groups", []) or []),
        )


def resolve_tenant_id(request: Request) -> str:
    """Resolve the effective tenant id from headers/claims (fallback 'default').

    Suite SSO identities never accept a raw X-Tenant-Id header: the tenant
    comes only from the provisioned identity (standard section 4.4.5), so the
    header is ignored while mode ns/oidc is active.
    """
    if config.sso.active:
        return "default"
    if tid := request.headers.get("X-Tenant-Id"):
        return tid
    return "default"


def auth_required() -> bool:
    """True when requests must carry a local JWT.

    Local password auth is on when auth.enabled is set; suite SSO (mode ns or
    oidc) also forces authentication - the auth.enabled=false dev bypass must
    never open the API while SSO identity is the gate (standard section 6.4).
    """
    return config.auth.enabled or config.sso.active


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> Claims:
    """FastAPI dependency that authenticates the request via Bearer JWT.

    Returns Claims. Raises 401 on missing/invalid token or refresh-token misuse.
    """
    if not auth_required():
        # dev mode: bypass auth, inject default tenant + admin roles
        return Claims(subject="admin", issuer="dev", tenant_id="default", roles=["admin", "user"])
    if credentials is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    signer = LocalSigner()
    try:
        claims = signer.parse(credentials.credentials)
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid token")
    if claims.issuer.endswith(REFRESH_ISSUER_SUFFIX):
        raise HTTPException(status_code=401, detail="refresh token not allowed here")
    return claims


def get_tenant_id(claims: Claims = Depends(require_auth)) -> str:
    """Dependency returning the tenant id from the authenticated claims."""
    return claims.tenant_id or "default"


def get_username(claims: Claims = Depends(require_auth)) -> str:
    """Dependency returning the subject (username) from claims."""
    return claims.subject or "anonymous"
