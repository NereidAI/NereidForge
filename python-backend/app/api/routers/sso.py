# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Suite SSO routes (NereidIdentity sso-integration-standard v2).

- ``GET  /api/v1/auth/sso/status``   login-page status (hide the entry unless enabled)
- ``GET  /api/v1/auth/sso/login``    browser entry, 302 to the NS enter URL or the IdP
- ``POST /api/v1/auth/sso/callback`` SPA callback: exchanges ``code``+``state``
- ``GET  <callback_path>``           mode-ns public callback (ticket -> local session)

Identity only enters through these server-side calls; the browser never
supplies identity claims (standard section 4.4).
"""
from __future__ import annotations

import json
from typing import Optional
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.api.helpers import get_svc
from app.core import sso
from app.core.config import config
from app.services import Services

router = APIRouter()
ns_router = APIRouter()


def _login_error_redirect(code: str) -> RedirectResponse:
    """Send the browser back to the SPA login page with a stable error code."""
    return RedirectResponse(f"/login?oidc_error={code}", status_code=302)


def _ns_enter_url(to: Optional[str]) -> str:
    """NS enter URL for a fresh ticket (validated relative deep link)."""
    n = config.sso.ns
    return (
        f"{n.base_url.rstrip('/')}/auth/enter/{n.product_id}"
        f"?to={quote(sso.safe_redirect_path(to), safe='')}"
    )


@router.get("/status")
async def sso_status() -> dict:
    """SSO mode/status for the login page (no secrets exposed)."""
    return sso.status()


@router.get("/login")
async def sso_login(to: Optional[str] = None) -> RedirectResponse:
    """Browser entry point: redirect to NereidSphere (ns) or the IdP (oidc)."""
    if not sso.status()["enabled"]:
        sso.record_error("not_configured")
        return _login_error_redirect("not_configured")
    try:
        url = await run_in_threadpool(sso.login_url, to)
    except sso.SSOError as exc:
        sso.record_error(exc.code)
        return _login_error_redirect(exc.code)
    return RedirectResponse(url, status_code=302)


class SSOCallbackRequest(BaseModel):
    """SPA callback payload (oidc mode)."""
    code: str = ""
    state: str = ""


@router.post("/callback")
async def sso_callback(req: SSOCallbackRequest, svc: Services = Depends(get_svc)) -> dict:
    """Exchange the authorization code and mint the normal local session.

    Returns the same shape as ``POST /api/v1/auth/login`` plus the validated
    ``to`` deep link; ``{"error": code}`` on any SSO failure.
    """
    try:
        result = await run_in_threadpool(sso.complete_oidc, req.code, req.state)
        user = await run_in_threadpool(
            sso.provision_identity, svc.admin, result["identity"], config.sso.oidc.issuer
        )
    except sso.SSOError as exc:
        sso.record_error(exc.code)
        return {"error": exc.code}
    payload = sso.issue_session(user)
    payload["to"] = result["to"]
    return payload


@ns_router.get("")
async def ns_callback(
    ticket: str = "", to: str = "", svc: Services = Depends(get_svc)
) -> RedirectResponse:
    """Public mode-ns callback: exchange the NS ticket for a local session.

    Success redirects into the SPA ``/auth/callback#session=...`` (the same
    delivery path as oidc mode). Burned/expired tickets bounce back to the NS
    enter URL for a fresh one; credential/config errors do not loop.
    """
    if not ticket:
        return RedirectResponse(_ns_enter_url(to), status_code=302)
    try:
        result = await run_in_threadpool(sso.verify_ticket, ticket)
        user = await run_in_threadpool(
            sso.provision_identity, svc.admin, result["identity"],
            f"ns:{config.sso.ns.base_url.rstrip('/')}",
        )
    except sso.SSOError as exc:
        sso.record_error(exc.code)
        if exc.code in ("ticket_invalid", "ticket_expired", "ticket_audience"):
            return RedirectResponse(_ns_enter_url(to), status_code=302)
        return _login_error_redirect(exc.code)
    payload = sso.issue_session(user)
    payload["to"] = result["to"]
    return RedirectResponse(
        "/auth/callback#" + urlencode({"session": json.dumps(payload)}), status_code=302
    )
