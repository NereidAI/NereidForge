# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Entitlements: which announced capabilities this deployment may execute.

The platform *registers* everything a component announces, including the
capabilities it does not ship (ADR-015). Execution is what the entitlement
controls:

* `core` — always allowed (the default product);
* `extended` — allowed only when the deployment grants the entitlement key the
  component asked for (`requires`), so complex capabilities can be delivered as
  paid extensions without leaving the Apache-2.0 tree.

This is deliberately a *platform* concern, not a licence check: it never looks at
what a component is licensed under (user-supplied trainers are the user's
responsibility, `LICENSE-POLICY.md` §3) and it never blocks a job for a
third-party trainer that announces nothing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.services import trainer_profiles

logger = logging.getLogger(__name__)

#: system_config key holding `{"entitlements": ["extended.optimization", ...]}`.
CONFIG_KEY = "platform"
CONFIG_FIELD = "entitlements"


class CapabilityLocked(RuntimeError):
    """Raised when a job asks for an extended capability without entitlement."""

    def __init__(self, missing: list[str], *, task: str = "", arch: str = "",
                 capability: str = "", image: str = "") -> None:
        self.missing = missing
        self.task = task
        self.arch = arch
        self.capability = capability
        self.image = image
        what = capability or arch or task or "this capability"
        super().__init__(
            f"{what} is an extended capability and needs an entitlement "
            f"({', '.join(missing) or 'unknown'}); it is not enabled in this deployment"
            + (f" [image {image}]" if image else "")
        )


def granted(db: Any) -> list[str]:
    """Entitlement keys granted by the deployment (empty when unconfigured)."""
    try:
        from sqlalchemy import select

        from app.models.orm import SystemConfig

        row = db.execute(
            select(SystemConfig).where(SystemConfig.key == CONFIG_KEY)
        ).scalar()
        cfg = (row.value if row else {}) or {}
        raw = cfg.get(CONFIG_FIELD) or []
        if isinstance(raw, str):
            raw = [item.strip() for item in raw.split(",")]
        return sorted({str(item).strip() for item in raw if str(item).strip()})
    except Exception as exc:  # noqa: BLE001 - config problems must not break jobs
        logger.warning("could not read entitlements: %s", exc)
        return []


def check(
    profile: Optional[dict[str, Any]],
    entitlements: list[str],
    *,
    task: str = "",
    arch: str = "",
    capability: str = "",
    image: str = "",
) -> None:
    """Raise `CapabilityLocked` when the announcement requires a missing key.

    Unknown/absent announcements pass: without a profile there is nothing the
    platform can (or should) gate, which keeps third-party trainers working.
    """
    if not profile:
        return
    missing = trainer_profiles.missing_entitlements(
        profile, entitlements, task=task, arch=arch, capability=capability
    )
    if missing:
        raise CapabilityLocked(missing, task=task, arch=arch, capability=capability, image=image)


def describe(profile: Optional[dict[str, Any]], entitlements: list[str]) -> dict[str, Any]:
    """Lock state of every extended capability a component announces.

    Used by the API/UI so extended capabilities can be *shown* (and sold)
    instead of hidden: each entry carries `locked` and the missing keys.
    """
    if not profile:
        return {"entitlements": entitlements, "locked": [], "available": []}
    granted_set = set(entitlements)
    locked: list[dict[str, Any]] = []
    available: list[dict[str, Any]] = []
    for entry in trainer_profiles.extended_capabilities(profile):
        missing = [key for key in (entry.get("requires") or []) if key not in granted_set]
        record = {**entry, "locked": bool(missing), "missing": missing}
        (locked if missing else available).append(record)
    return {"entitlements": entitlements, "locked": locked, "available": available}
