# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Auth routes: login, refresh, logout, me."""
from __future__ import annotations

import random
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.auth import LocalSigner, check_password, require_auth, Claims
from app.services import Services
from app.services import kv_store
from app.api.helpers import get_svc, error

router = APIRouter()

# Captcha answers are single-use TTL records in Postgres (kv_store), so every
# uvicorn worker / api replica shares the same set.
_CAPTCHA_TTL = 600


def _new_captcha() -> dict:
    """Generate an arithmetic captcha (addition/subtraction/multiplication
    within 10) and store it, returning {captcha_id, expression}."""
    op = random.choice(["+", "-", "x"])
    if op == "+":
        a, b = random.randint(0, 10), random.randint(0, 10)
        answer = a + b
    elif op == "-":
        a, b = random.randint(0, 10), random.randint(0, 10)
        a, b = max(a, b), min(a, b)
        answer = a - b
    else:
        a, b = random.randint(1, 5), random.randint(1, 5)
        answer = a * b
    cid = uuid.uuid4().hex
    kv_store.kv_put(kv_store.CAPTCHA, cid,
                   {"expression": f"{a} {op} {b} = ?", "answer": answer,
                    "expires_at": time.time() + _CAPTCHA_TTL},
                   _CAPTCHA_TTL)
    return {"captcha_id": cid, "expression": f"{a} {op} {b} = ?"}


def _check_captcha(captcha_id: Optional[str], answer: Optional[int]) -> bool:
    """Validate a captcha answer; consumes it on success or expiry."""
    if not captcha_id or answer is None:
        return False
    entry = kv_store.kv_pop(kv_store.CAPTCHA, captcha_id)
    if not entry:
        return False
    if entry["expires_at"] < time.time():
        return False
    try:
        return int(answer) == entry["answer"]
    except (TypeError, ValueError):
        return False


class LoginRequest(BaseModel):
    """Login request body."""
    username: str
    password: str
    # Optional tenant scope: when set, the username must belong to this
    # tenant (namespace/profile); when blank, super-admin login is assumed.
    namespace: Optional[str] = ""
    # Arithmetic captcha protection.
    captcha_id: Optional[str] = None
    captcha_answer: Optional[int] = None


class RefreshRequest(BaseModel):
    """Refresh request body."""
    refresh_token: str


class RefreshResponse(BaseModel):
    """New access-token response; the original refresh token remains unchanged."""

    access_token: str
    token_type: str
    expires_in: int


class AuthErrorResponse(BaseModel):
    """FastAPI error envelope returned when refresh authentication fails."""

    detail: str


@router.get("/captcha")
async def captcha():
    """Generate an arithmetic captcha (addition/subtraction/multiplication within 10) and return it."""
    return _new_captcha()


@router.post("/login")
async def login(req: LoginRequest, svc: Services = Depends(get_svc)):
    """Authenticate a user and return access + refresh tokens.

    - captcha_id + captcha_answer are required and validated first.
    - namespace (KF Profile/tenant): when provided the user must belong to
      that tenant; when blank the super-admin path is assumed.
    """
    if not _check_captcha(req.captcha_id, req.captcha_answer):
        raise error(401, "captcha invalid or expired")
    user = svc.admin.get_user_by_username(req.username)
    if not user or not check_password(req.password, user.password_hash):
        raise error(401, "invalid credentials")
    namespace = (req.namespace or "").strip()
    if namespace:
        # Tenant login: the user must be bound to the requested namespace.
        if user.tenant_id != namespace:
            raise error(401, "user does not belong to the given namespace")
    elif user.tenant_id not in ("default", ""):
        # Blank namespace implies super-admin; tenant users must scope in.
        raise error(401, "a namespace is required for tenant users")
    signer = LocalSigner()
    roles = list(user.roles or [])
    access = signer.sign(user.id, user.tenant_id, roles)
    refresh = signer.sign_refresh(user.id, user.tenant_id)
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer",
        "expires_in": signer.ttl,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "roles": roles,
            "tenant_id": user.tenant_id,
        },
    }


@router.post(
    "/refresh",
    response_model=RefreshResponse,
    summary="Refresh an access token",
    responses={401: {"model": AuthErrorResponse, "description": "Invalid or expired refresh token"}},
)
async def refresh(req: RefreshRequest, svc: Services = Depends(get_svc)):
    """Exchange a refresh token for a new access token."""
    signer = LocalSigner()
    try:
        claims = signer.parse(req.refresh_token)
    except ValueError:
        raise error(401, "invalid refresh token")
    if not claims.issuer.endswith("-refresh"):
        raise error(401, "invalid refresh token")
    user = svc.admin.get_user_by_id(claims.subject)
    if not user:
        raise error(401, "invalid refresh token")
    access = signer.sign(user.id, user.tenant_id, list(user.roles or []))
    return {"access_token": access, "token_type": "Bearer", "expires_in": signer.ttl}


@router.post("/logout")
async def logout():
    """Logout is stateless; returns ok."""
    return {"ok": True}


@router.get("/me")
async def me(claims: Claims = Depends(require_auth)):
    """Return the current authenticated user's profile (from JWT claims)."""
    return {
        "id": claims.subject,
        "username": claims.subject,
        "tenant_id": claims.tenant_id or "default",
        "roles": claims.roles,
    }
