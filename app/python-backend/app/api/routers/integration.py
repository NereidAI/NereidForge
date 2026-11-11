# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NS integration endpoints (NereidSphere admin console -> NereidForge).

NereidSphere's suite console renders a product tenant picker, so it needs
NereidForge's own tenant list. The call arrives through the NS reverse proxy at
``GET /v1/integration/ns/tenants`` and NS is the only party that can set the
``X-NS-Integration-Secret`` header (the gateway stamps it on exactly the
configured ``products[forge].tenants_path`` and strips any inbound copy).

This matters because the product ingress is directly reachable: without the
shared secret a caller could forge ``X-User-*`` headers and impersonate an
administrator. NF therefore requires the secret (constant-time compared against
``sso.ns.client_secret``, rejecting when either side is empty) *and* an
``admin``/``operator`` role, mirroring NS's own product-console role gate.

``groups`` is deliberately not carried here — it rides inside the ticket
identity only; this path carries the coarse ``X-User-Roles`` header.
Read-only: no tenant is ever mutated.
"""
from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.helpers import get_db
from app.core.config import config
from app.models.orm import Tenant

router = APIRouter()


def _require_ns_integration(request: Request) -> None:
    """Authenticate an NS-originated admin call: shared secret + admin role."""
    secret = request.headers.get("X-NS-Integration-Secret", "")
    expected = config.sso.ns.client_secret or ""
    # Constant-time compare; reject when either side is empty so an unconfigured
    # deployment can never be reached with a blank secret.
    if not secret or not expected or not hmac.compare_digest(secret, expected):
        raise HTTPException(status_code=401, detail="invalid integration secret")
    roles = {r.strip() for r in request.headers.get("X-User-Roles", "").split(",") if r.strip()}
    # admin OR operator: NS's own product-console gates allow both, so requiring
    # admin alone would break the tenant picker for operator admins.
    if not ({"admin", "operator"} & roles):
        raise HTTPException(status_code=403, detail="admin role required")


@router.get("/v1/integration/ns/tenants")
def ns_tenants(request: Request, db: Session = Depends(get_db)) -> list[dict]:
    """Return the active tenants in the canonical ``tenant_id``/``tenant_name`` shape."""
    _require_ns_integration(request)
    rows = db.scalars(
        select(Tenant).where(Tenant.is_active.is_(True)).order_by(Tenant.name)
    ).all()
    return [{"tenant_id": t.id, "tenant_name": t.display_name or t.name} for t in rows]
