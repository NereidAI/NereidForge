# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Admin service: tenants, users, roles, and system_config management.

Mirrors the Go AdminService (internal/service/admin.go). All system runtime
config (kfp_host, pipeline ids, trainer image, etc.) lives in the single
system_config row keyed 'platform' as a JSON object.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.core.auth import hash_password
from app.core.config import config
from app.core.util import iso, new_id, page_bounds
from app.models.orm import Role, SystemConfig, Tenant, User


def _tenant_dict(t: Tenant) -> dict[str, Any]:
    """Serialize a Tenant to the API response shape."""
    return {
        "id": t.id,
        "name": t.name,
        "display_name": t.display_name,
        "description": t.description,
        "is_active": t.is_active,
        "admin_username": t.admin_username,
        # tenant layer namespace: nereidforge-<name> (the KF Profile ns);
        # the platform default tenant shares the super-admin's own profile.
        "namespace": "nereidforge" if t.name == "default" else f"nereidforge-{t.name}",
        "kfp_userid": t.kfp_userid,
        "kfp_experiment_id": t.kfp_experiment_id,
        "created_at": iso(t.created_at),
        "updated_at": iso(t.updated_at),
    }


def _user_dict(u: User) -> dict[str, Any]:
    """Serialize a User to the API response shape (no password hash)."""
    return {
        "id": u.id,
        "username": u.username,
        "email": u.email,
        "full_name": u.full_name,
        "roles": u.roles or [],
        "tenant_id": u.tenant_id,
        "is_active": u.is_active,
        "created_at": iso(u.created_at),
        "updated_at": iso(u.updated_at),
    }


class AdminService:
    """Handles platform administration: tenants, users, roles, system config."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # -- tenants ------------------------------------------------------------

    def list_tenants(self, page: int, page_size: int, search: str = "") -> tuple[list[dict], int]:
        """List tenants with pagination, optionally filtered by name."""
        q = select(Tenant)
        if search:
            q = q.where(Tenant.name.ilike(f"%{search.strip()}%"))
        limit, offset = page_bounds(page, page_size)
        total = self.db.scalar(select(func.count()).select_from(q.subquery())) or 0
        rows = self.db.scalars(
            q.order_by(Tenant.created_at.desc()).limit(limit).offset(offset)
        ).all()
        return [_tenant_dict(t) for t in rows], total

    def get_tenant(self, id: str) -> Optional[dict]:
        """Get a tenant by id."""
        t = self.db.get(Tenant, id)
        return _tenant_dict(t) if t else None

    def create_tenant(self, name: str, display_name: str, description: str,
                      admin_username: str = "") -> dict:
        """Create a tenant and provision its namespace layer.

        The tenant layer is a KF Profile named nereidforge-<name> (KFP
        experiments, KServe visibility, notebooks all live there). The
        platform super-admin ('admin' in the default tenant) is separate
        from tenant admins stored in admin_username.
        """
        t = Tenant(
            id=new_id("tnt"), name=name, display_name=display_name or name,
            description=description or "", admin_username=admin_username or "",
        )
        self.db.add(t)
        self.db.commit()
        self.db.refresh(t)
        try:
            provision_tenant_profile(name, admin_username)
        except Exception as e:  # noqa: BLE001
            # Tenant row survives; the namespace can be provisioned later.
            logging.getLogger(__name__).warning(
                "tenant %s profile provisioning failed: %s", name, e
            )
        return _tenant_dict(t)

    def update_tenant(self, id: str, patch: dict[str, Any]) -> Optional[dict]:
        """Update tenant fields from a patch dict."""
        t = self.db.get(Tenant, id)
        if not t:
            return None
        for k, v in patch.items():
            if hasattr(t, k):
                setattr(t, k, v)
        self.db.commit()
        self.db.refresh(t)
        return _tenant_dict(t)

    def delete_tenant(self, id: str, keep_namespace: bool = True) -> bool:
        """Delete a tenant by id (namespace kept by default for safety)."""
        t = self.db.get(Tenant, id)
        if not t:
            return False
        if t.name == "default":
            raise ValueError("the platform default tenant cannot be deleted")
        if not keep_namespace and t.name:
            try:
                deprovision_tenant_profile(t.name)
            except Exception:  # noqa: BLE001
                pass
        self.db.delete(t)
        self.db.commit()
        return True

    # -- users --------------------------------------------------------------

    def get_user_by_id(self, id: str) -> Optional[User]:
        """Fetch a User row by its stable primary key for token subject resolution."""
        return self.db.get(User, id)

    def is_platform_super_admin(
        self, user_id: str, tenant_id: str, roles: list[str],
    ) -> bool:
        """Authorize destructive platform operations against the current local admin row.

        Access-token subjects are stable user IDs, not usernames. Resolve that ID
        instead of trusting a header or display name, and re-check current role and
        account state so a revoked or disabled administrator cannot delete registry
        records with an unexpired token. Federated accounts are excluded even when
        their Identity username happens to be ``admin``.
        """
        if tenant_id != "default" or "admin" not in roles:
            return False
        user = self.get_user_by_id(user_id)
        return bool(
            user
            and user.is_active
            and user.username == "admin"
            and user.tenant_id == "default"
            and "admin" in (user.roles or [])
            and not user.external_id
            and not user.external_issuer
        )

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Fetch a User row by username (used by login)."""
        return self.db.scalar(select(User).where(User.username == username))

    def list_users(self, page: int, page_size: int, search: str = "") -> tuple[list[dict], int]:
        """List users with pagination, optionally filtered by name/email."""
        q = select(User)
        if search:
            needle = f"%{search.strip()}%"
            q = q.where(or_(User.username.ilike(needle), User.full_name.ilike(needle), User.email.ilike(needle)))
        limit, offset = page_bounds(page, page_size)
        total = self.db.scalar(select(func.count()).select_from(q.subquery())) or 0
        rows = self.db.scalars(
            q.order_by(User.created_at.desc()).limit(limit).offset(offset)
        ).all()
        return [_user_dict(u) for u in rows], total

    def create_user(
        self, username: str, email: str, full_name: str, password: str,
        tenant_id: str, roles: list[str],
    ) -> dict:
        """Create a user with a bcrypt-hashed password."""
        u = User(
            id=new_id("usr"), username=username, email=email or "",
            full_name=full_name or "", password_hash=hash_password(password) if password else "",
            roles=roles or [], tenant_id=tenant_id or "default",
        )
        self.db.add(u)
        self.db.commit()
        self.db.refresh(u)
        return _user_dict(u)

    def provision_sso_user(
        self, *, sub: str, issuer: str, username: str, email: str, full_name: str,
        roles: list[str], tenant_id: str,
    ) -> dict:
        """JIT-provision or refresh the local user bound to an SSO identity.

        Keyed by (external_issuer, external_id=sub) per the SSO standard: the
        same realm identity always resolves to the same local user, however
        often it logs in.

        Suite-managed membership (roles + tenant binding) follows NS on every
        login while ``sso.ns.reconcile_memberships`` is true, so revoking an
        assignment in NS reduces access on the next login; display fields
        (email, full name, username) always refresh. Only federated
        (external_id-bound) rows are touched here, so a local/break-glass admin
        row is never demoted by an NS login — such an admin simply gets a
        separate federated account.

        A stale assignment (the tenant was deleted in NF) falls back to
        ``sso.jit_tenant`` with a warning rather than pointing the user at a
        non-existent tenant. A locally disabled user (is_active=False) stays
        disabled.
        """
        from app.core.sso import SSOError

        # Validate the tenant binding before writing it: NF's own tenants table
        # is authoritative, and the group may outlive the tenant row it names.
        if tenant_id and self.db.get(Tenant, tenant_id) is None:
            logging.getLogger(__name__).warning(
                "sso: unknown tenant %s for %s, using %s",
                tenant_id, username, config.sso.jit_tenant,
            )
            tenant_id = config.sso.jit_tenant or "default"

        u = self.db.scalar(select(User).where(
            User.external_id == sub, User.external_issuer == issuer
        ))
        if u is not None:
            if not u.is_active:
                raise SSOError("account_disabled", "local account is disabled")
            u.email = email or u.email
            u.full_name = full_name or u.full_name
            if (username or "").strip():
                u.username = self._unique_username(username, exclude_id=u.id)
            if config.sso.ns.reconcile_memberships:
                u.roles = roles or u.roles
                u.tenant_id = tenant_id or u.tenant_id
            self.db.commit()
            self.db.refresh(u)
            return _user_dict(u)

        name = self._unique_username(username or email.split("@")[0] or f"sso-{sub[:12]}")
        u = User(
            id=new_id("usr"), username=name, email=email or "",
            full_name=full_name or "", password_hash="",
            roles=roles or [], tenant_id=tenant_id or "default",
            external_id=sub, external_issuer=issuer,
        )
        self.db.add(u)
        self.db.commit()
        self.db.refresh(u)
        return _user_dict(u)

    def _unique_username(self, preferred: str, exclude_id: str = "") -> str:
        """Derive a unique local username from a preferred SSO name.

        ``exclude_id`` excludes the current user row so a re-login keeps its
        own name instead of suffixed-renaming itself.
        """
        base = re.sub(r"[^A-Za-z0-9._-]+", "-", (preferred or "").strip()).strip("-") or "sso-user"
        base = base[:48]
        candidate = base
        suffix = 2
        while True:
            existing = self.db.scalar(select(User).where(User.username == candidate))
            if existing is None or existing.id == exclude_id:
                return candidate
            candidate = f"{base}-{suffix}"
            suffix += 1

    def update_user(self, id: str, patch: dict[str, Any]) -> Optional[dict]:
        """Update user fields; supports password reset via 'password' key."""
        u = self.db.get(User, id)
        if not u:
            return None
        for k, v in patch.items():
            if k == "password" and v:
                u.password_hash = hash_password(v)
            elif hasattr(u, k):
                setattr(u, k, v)
        self.db.commit()
        self.db.refresh(u)
        return _user_dict(u)

    def delete_user(self, id: str) -> bool:
        """Delete a user by id."""
        u = self.db.get(User, id)
        if not u:
            return False
        self.db.delete(u)
        self.db.commit()
        return True

    # -- roles --------------------------------------------------------------

    def list_roles(self) -> list[dict]:
        """List all roles."""
        rows = self.db.scalars(select(Role).order_by(Role.name)).all()
        return [
            {"name": r.name, "description": r.description, "permissions": r.permissions or []}
            for r in rows
        ]

    def create_role(self, name: str, description: str, permissions: list[str]) -> dict:
        """Create a role."""
        r = Role(name=name, description=description or "", permissions=permissions or [])
        self.db.add(r)
        self.db.commit()
        return {"name": r.name, "description": r.description, "permissions": r.permissions or []}

    def update_role(self, name: str, patch: dict[str, Any]) -> Optional[dict]:
        """Update role fields."""
        r = self.db.get(Role, name)
        if not r:
            return None
        for k, v in patch.items():
            if hasattr(r, k):
                setattr(r, k, v)
        self.db.commit()
        return {"name": r.name, "description": r.description, "permissions": r.permissions or []}

    def delete_role(self, name: str) -> bool:
        """Delete a role by name."""
        r = self.db.get(Role, name)
        if not r:
            return False
        self.db.delete(r)
        self.db.commit()
        return True

    # -- system config ------------------------------------------------------

    def get_system_config(self) -> dict[str, Any]:
        """Read the platform system_config JSON (key='platform')."""
        row = self.db.get(SystemConfig, "platform")
        if not row:
            return {}
        return row.value or {}

    def update_system_config(self, data: dict[str, Any]) -> dict[str, Any]:
        """Merge data into the platform system_config and persist."""
        existing = self.get_system_config()
        existing.update(data or {})
        row = self.db.get(SystemConfig, "platform")
        if row:
            row.value = existing
        else:
            self.db.add(SystemConfig(key="platform", value=existing))
        self.db.commit()
        return existing

def provision_tenant_profile(name: str, admin_username: str) -> str:
    """Return the tenant namespace name (idempotent).

    The KF Profile + namespace (with its kubeflow-admin/editor/viewer role
    bindings and kserve service account) must be provisioned by cluster ops —
    the NereidForge container has no cluster access and never mutates it.
    NF only records the tenant in its own DB.
    """
    import logging

    logging.getLogger("nereidforge.admin").warning(
        "tenant %s: KF Profile/namespace provisioning is an ops task; "
        "no cluster resources were changed by NF", name,
    )
    return f"nereidforge-{name}"


def deprovision_tenant_profile(name: str) -> None:
    """No-op: deprovisioning the KF Profile is an ops task.

    Deleting a Profile cascades the whole namespace (Pods, PVCs, data) so the
    NereidForge worker must never do it. Ops runs `kubectl delete profile`.
    """
    import logging

    logging.getLogger("nereidforge.admin").warning(
        "tenant %s: KF Profile deprovisioning is an ops task; "
        "no cluster resources were changed by NF", name,
    )
