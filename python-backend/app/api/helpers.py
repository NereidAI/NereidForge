# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""API helpers: response envelopes, auth dependencies, service container access."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.auth import require_auth, Claims
from app.db.session import SessionLocal
from app.services import Services


def get_db():
    """Yield a scoped DB session for the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_svc(request: Request, db: Session = Depends(get_db)) -> Services:
    """Build the Services container for the request (scoped to the DB session)."""
    return Services(db)


def tenant_id(request: Request) -> str:
    """Resolve the tenant id from the request headers/claims."""
    if tid := request.headers.get("X-Tenant-Id"):
        return tid
    return "default"


def username(request: Request) -> str:
    """Resolve the acting username from claims."""
    return request.headers.get("X-Username", "anonymous")


def require_platform_super_admin(
    claims: Claims = Depends(require_auth),
    svc: Services = Depends(get_svc),
) -> Claims:
    """Restrict an operation to the enabled local admin in the default tenant.

    The development bypass already authenticates as that identity. Real tokens
    carry a user ID, so authorization resolves the current database row rather
    than trusting tenant headers or comparing the subject with a username.
    """
    is_dev_admin = (
        claims.issuer == "dev"
        and claims.subject == "admin"
        and claims.tenant_id == "default"
        and "admin" in claims.roles
    )
    if not is_dev_admin and not svc.admin.is_platform_super_admin(
        claims.subject, claims.tenant_id or "default", claims.roles,
    ):
        raise error(403, "platform super administrator privileges required")
    return claims


def error(status: int, message: str) -> HTTPException:
    """Build a Go-compatible error envelope."""
    return HTTPException(status_code=status, detail=message)


def not_found() -> HTTPException:
    """Return the standard 404 envelope."""
    return error(404, "resource not found")
