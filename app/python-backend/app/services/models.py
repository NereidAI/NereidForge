# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Model service: model/version CRUD, artifact streaming, and hub registration.

Mirrors the Go ModelService (internal/service/models.go). PostgreSQL stores
operational model metadata (for the UI); the Kubeflow Hub (Model Registry) is
the lifecycle source of truth, written via the official Python SDK.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.clients.s3 import SeaweedFSClient
from app.core.util import iso, new_id, page_bounds
from app.models.orm import Model as ModelORM
from app.models.orm import ModelVersion as ModelVersionORM
from app.services.model_format import (
    artifact_capability_permitted,
    normalize_artifact_relative_path,
)


def _artifact_anchor(file_path: str, metadata: dict[str, Any]) -> str:
    """_artifact_anchor resolves one consistent root for listing and object authorization.

    Only the backend-issued capability list marks bundle_uri as trustworthy;
    legacy or metadata-only input cannot redirect listing to another S3 prefix.
    """
    if isinstance(metadata.get("artifact_capabilities"), list) and metadata.get("bundle_uri"):
        return str(metadata["bundle_uri"])
    return str(metadata.get("onnx_uri") or file_path or "")


def _model_dict(m: ModelORM, versions: Optional[list[dict]] = None) -> dict[str, Any]:
    """Serialize a Model to the API response shape."""
    return {
        "id": m.id,
        "tenant_id": m.tenant_id,
        "name": m.name,
        "description": m.description,
        "framework": m.framework,
        "task_type": m.task_type,
        "created_by": m.created_by,
        "created_at": iso(m.created_at),
        "updated_at": iso(m.updated_at),
        "versions": versions or [],
    }


def _version_dict(v: ModelVersionORM) -> dict[str, Any]:
    """Serialize a ModelVersion to the API response shape."""
    meta = v.metadata_ or {}
    onnx_uri = str(meta.get("onnx_uri", "") or "")
    # Prefer ONNX as primary file_path for backward compatibility
    file_path = onnx_uri or str(v.file_path or "")
    return {
        "id": v.id,
        "model_id": v.model_id,
        "version": v.version,
        "description": v.description,
        "file_path": file_path,
        "size_bytes": v.size_bytes,
        "metadata": v.metadata_ or {},
        "status": v.status,
        "artifacts": v.artifacts or [],
        # Real artifact count (files in the version's S3 prefix) is filled by
        # _attach_artifact_counts on list endpoints; 0 keeps the shape stable.
        "artifacts_count": 0,
        "created_by": v.created_by,
        "created_at": iso(v.created_at),
    }


class ModelTrialError(ValueError):
    """ModelTrialError preserves an ML Backend failure's HTTP status, detail, and code."""

    def __init__(self, status_code: int, detail: str, code: Optional[str] = None) -> None:
        """__init__ carries structured downstream error information to the API boundary."""
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.code = code


class ModelDeleteConflict(ValueError):
    """ModelDeleteConflict identifies live versions that must be undeployed first."""

    def __init__(self, versions: list[str]) -> None:
        """ModelDeleteConflict retains the blocking version names for API diagnostics."""
        self.versions = sorted(versions)
        joined = ", ".join(self.versions)
        super().__init__(
            f"model versions are still deployed or transitioning: {joined}; "
            "undeploy them before deletion"
        )


def _trial_request_hints(meta: dict[str, Any], model_path: str) -> dict[str, Any]:
    """_trial_request_hints builds the common ML request from non-authoritative metadata.

    Source, architecture, and input shape help the ML Backend inspect or run an
    artifact; NF deliberately makes no decoder-selection decision from them.
    """
    body: dict[str, Any] = {
        "model_path": model_path,
        "source": str(meta.get("source", "") or "auto").strip().lower(),
    }
    if meta.get("architecture"):
        body["architecture"] = str(meta["architecture"]).strip()
    if meta.get("input_shape") is not None:
        body["input_shape"] = meta["input_shape"]
    return body


class ModelService:
    """Manages models/versions with optional Kubeflow Hub registration."""

    def __init__(self, db: Session, s3: SeaweedFSClient, hub: Any = None) -> None:
        self.db = db
        self.s3 = s3
        self.hub = hub

    # -- models -------------------------------------------------------------

    def create_model(
        self, tenant_id: str, name: str, description: str, framework: str,
        task_type: str, created_by: str,
    ) -> dict:
        """Create a model (PG row); the caller may also register it to hub.

        Idempotent by (tenant_id, name): if the model already exists it is
        returned as-is, so job completion/register can be retried safely.
        """
        if not name:
            raise ValueError("name is required")
        existing = self.db.scalar(
            select(ModelORM).where(
                and_(ModelORM.tenant_id == tenant_id, ModelORM.name == name)
            )
        )
        if existing:
            return _model_dict(existing)
        m = ModelORM(
            id=new_id("m"),
            tenant_id=tenant_id,
            name=name,
            description=description or "",
            framework=framework or "pytorch",
            task_type=task_type or "classification",
            created_by=created_by or "anonymous",
        )
        self.db.add(m)
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            existing = self.db.scalar(
                select(ModelORM).where(
                    and_(ModelORM.tenant_id == tenant_id, ModelORM.name == name)
                )
            )
            if existing:
                return _model_dict(existing)
            raise
        self.db.refresh(m)
        return _model_dict(m)

    def list_models(
        self, tenant_id: str, page: int, page_size: int, framework: str = "",
        task_type: str = "", search: str = "",
    ) -> tuple[list[dict], int]:
        """List models. The Kubeflow Hub is the source of truth for lifecycle
        metadata; PostgreSQL is only an operational snapshot. When the hub is
        reachable we read from it (all models, no tenant scoping), otherwise we
        fall back to PG rows."""
        if self.hub and self.hub.enabled():
            try:
                hub_models = self.hub.list_models_detailed()
                if hub_models:
                    # Optional filters applied over hub data.
                    filtered = hub_models
                    if search:
                        needle = search.strip().lower()
                        filtered = [m for m in filtered if needle in m["name"].lower()]
                    if framework:
                        filtered = [m for m in filtered if self._pg_framework(m["name"]) == framework]
                    if task_type:
                        filtered = [m for m in filtered if self._pg_task(m["name"]) == task_type]
                    # Merge in PG-only models that were never registered to the
                    # hub (e.g. a model whose first version upload failed). Hub
                    # rows have no "versions" for these, so append them as
                    # version-less shells built from the PG row; otherwise the
                    # model silently disappears from the list the moment the
                    # hub becomes reachable (user sees "my model is gone").
                    hub_names = {m["name"] for m in filtered}
                    pg_only = self._pg_models_missing_from_hub(tenant_id, hub_names)
                    for m in pg_only:
                        if search and search.strip().lower() not in m["name"].lower():
                            continue
                        if framework and (m.get("framework") or "") != framework:
                            continue
                        if task_type and (m.get("task_type") or "") != task_type:
                            continue
                        filtered.append({
                            "name": m["name"],
                            "id": m["id"],
                            "tenant_id": tenant_id,
                            "description": m.get("description", ""),
                            "framework": m.get("framework", ""),
                            "task_type": m.get("task_type", ""),
                            "created_by": m.get("created_by", ""),
                            "created_at": m.get("created_at"),
                            "updated_at": m.get("updated_at"),
                            "versions": m.get("versions", []),
                        })
                    total = len(filtered)
                    start = (page - 1) * page_size
                    items = filtered[start:start + page_size]
                    # Merge PG framework/task_type/created_at into hub rows for display.
                    merged = []
                    latest_versions: list[dict[str, Any]] = []
                    for m in items:
                        pg = self._pg_model_by_name(tenant_id, m["name"])
                        # Hub versions carry no artifact URI; back-fill
                        # file_path from PG so artifact counts resolve.
                        pg_paths = self._pg_version_paths(pg["id"]) if pg else {}
                        for hv in m["versions"]:
                            if not hv.get("file_path") and hv.get("version") in pg_paths:
                                hv["file_path"] = pg_paths[hv["version"]]
                        merged.append({
                            **m,
                            "id": pg["id"] if pg else m["name"],
                            "tenant_id": tenant_id,
                            "framework": pg["framework"] if pg else "",
                            "task_type": pg["task_type"] if pg else "",
                            "created_by": pg["created_by"] if pg else "",
                            "created_at": pg["created_at"] if pg else None,
                            "updated_at": pg["updated_at"] if pg else None,
                            "versions": m["versions"],
                        })
                        if m["versions"]:
                            latest_versions.append(m["versions"][0])
                    # The list page only shows the latest version's count.
                    self._attach_artifact_counts(latest_versions)
                    return merged, total
            except Exception:
                pass
        # Fallback to PG rows.
        from sqlalchemy import func

        query = select(ModelORM).where(ModelORM.tenant_id == tenant_id)
        count_q = select(func.count(ModelORM.id)).where(ModelORM.tenant_id == tenant_id)
        if search:
            query = query.where(ModelORM.name.ilike(f"%{search.strip()}%"))
            count_q = count_q.where(ModelORM.name.ilike(f"%{search.strip()}%"))
        if framework:
            query = query.where(ModelORM.framework == framework)
            count_q = count_q.where(ModelORM.framework == framework)
        if task_type:
            query = query.where(ModelORM.task_type == task_type)
            count_q = count_q.where(ModelORM.task_type == task_type)
        total = self.db.scalar(count_q) or 0
        limit, offset = page_bounds(page, page_size)
        rows = self.db.scalars(query.order_by(ModelORM.created_at.desc()).limit(limit).offset(offset)).all()
        items = []
        latest_versions: list[dict[str, Any]] = []
        for m in rows:
            versions = self._versions_for(m.id, tenant_id)
            if versions:
                latest_versions.append(versions[0])
            items.append(_model_dict(m, versions))
        # The list page only renders the artifact count of the newest version.
        self._attach_artifact_counts(latest_versions)
        return items, total

    def _pg_model_by_name(self, tenant_id: str, name: str) -> Optional[dict]:
        """Look up a PG model row by name (for framework/task display)."""
        m = self.db.scalar(
            select(ModelORM).where(
                and_(ModelORM.tenant_id == tenant_id, ModelORM.name == name)
            )
        )
        if not m:
            return None
        return {
            "id": m.id,
            "framework": m.framework,
            "task_type": m.task_type,
            "created_by": m.created_by,
            "created_at": iso(m.created_at),
            "updated_at": iso(m.updated_at),
        }

    def _pg_framework(self, name: str) -> str:
        """Fetch the framework for a model name from PG ('' when absent)."""
        m = self.db.scalar(select(ModelORM).where(ModelORM.name == name))
        return m.framework if m else ""

    def _pg_task(self, name: str) -> str:
        """Fetch the task type for a model name from PG ('' when absent)."""
        m = self.db.scalar(select(ModelORM).where(ModelORM.name == name))
        return m.task_type if m else ""

    def _pg_models_missing_from_hub(self, tenant_id: str, hub_names: set[str]) -> list[dict[str, Any]]:
        """PG rows for this tenant whose names are absent from the hub.

        Used by list_models' hub branch so that models that never reached the
        hub (version upload failure, manual DB insert, etc.) still appear in
        the UI instead of silently vanishing. Version-less shells have
        versions=[]; the UI's empty-state/textual hint in the model row makes
        that state visible without registering anything.
        """
        rows = self.db.scalars(
            select(ModelORM).where(ModelORM.tenant_id == tenant_id)
        ).all()
        out: list[dict[str, Any]] = []
        for m in rows:
            if m.name in hub_names:
                continue
            out.append(_model_dict(m, self._versions_for(m.id, tenant_id)))
        return out

    def _pg_version_paths(self, model_id: str) -> dict[str, str]:
        """Map PG version id -> primary file_path for hub-version enrichment."""
        rows = self.db.scalars(
            select(ModelVersionORM).where(ModelVersionORM.model_id == model_id)
        ).all()
        out: dict[str, str] = {}
        for v in rows:
            meta = v.metadata_ or {}
            uri = str(meta.get("onnx_uri", "") or v.file_path or "")
            if uri:
                out[v.version] = uri
        return out

    def _attach_artifact_counts(self, versions: list[dict[str, Any]]) -> None:
        """Fill artifacts_count by counting files in each version's S3 prefix.

        One bounded recursive LIST per version runs in a small thread pool
        since list pages are server-side paginated (<= a few dozen rows).
        Best effort: S3 errors degrade to 0 instead of failing the registry page.
        """
        from concurrent.futures import ThreadPoolExecutor

        targets: list[dict[str, Any]] = []
        for v in versions:
            v.setdefault("artifacts_count", 0)
            anchor = _artifact_anchor(str(v.get("file_path") or ""), v.get("metadata") or {})
            if anchor.startswith("s3://"):
                targets.append(v)

        def count_one(ver: dict[str, Any]) -> tuple[dict[str, Any], int]:
            try:
                anchor = _artifact_anchor(str(ver.get("file_path") or ""), ver.get("metadata") or {})
                return ver, int(self.s3.count_files_in_dir(anchor))
            except Exception:  # noqa: BLE001 - display-only stat, never fatal
                return ver, 0

        if targets:
            with ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:
                for ver, n in pool.map(count_one, targets):
                    ver["artifacts_count"] = n

    def get_model(self, tenant_id: str, ref: str) -> Optional[dict]:
        """Get a model by id or name. Prefers the hub (source of truth); falls
        back to PG when the hub is unreachable."""
        if self.hub and self.hub.enabled():
            try:
                hub_models = self.hub.list_models_detailed()
                for m in hub_models:
                    if m["name"] == ref:
                        pg = self._pg_model_by_name(tenant_id, m["name"])
                        return {
                            **m,
                            "id": pg["id"] if pg else m["name"],
                            "tenant_id": tenant_id,
                            "framework": pg["framework"] if pg else "",
                            "task_type": pg["task_type"] if pg else "",
                            "created_by": pg["created_by"] if pg else "",
                            "created_at": pg["created_at"] if pg else None,
                            "updated_at": pg["updated_at"] if pg else None,
                            "versions": m["versions"],
                        }
            except Exception:
                pass
        m = self.db.scalar(
            select(ModelORM).where(
                and_(
                    ModelORM.tenant_id == tenant_id,
                    (ModelORM.id == ref) | (ModelORM.name == ref),
                )
            )
        )
        if not m:
            return None
        versions = self._versions_for(m.id, tenant_id)
        return _model_dict(m, versions)

    def update_model(self, tenant_id: str, id: str, patch: dict[str, Any]) -> Optional[dict]:
        """Update model fields from a patch dict."""
        m = self.db.scalar(
            select(ModelORM).where(and_(ModelORM.id == id, ModelORM.tenant_id == tenant_id))
        )
        if not m:
            return None
        for k, v in patch.items():
            if hasattr(m, k) and k not in ("id", "created_at"):
                setattr(m, k, v)
        self.db.commit()
        self.db.refresh(m)
        return _model_dict(m, self._versions_for(m.id, tenant_id))

    @staticmethod
    def _ensure_versions_not_live(versions: list[ModelVersionORM]) -> None:
        """Reject deletion before any registry write while a version may still serve.

        ``undeploy_failed`` is also blocking because the last verified cluster
        state says the inference service remains present. A failed initial deploy
        is not live and therefore does not prevent registry cleanup.
        """
        live_states = {"ready", "deploying", "terminating", "undeploy_failed"}
        blocking = []
        for version in versions:
            metadata = version.metadata_ or {}
            if (
                version.status in {"deployed", "deploying"}
                or metadata.get("deployed") is True
                or metadata.get("deploy_status") in live_states
            ):
                blocking.append(version.version)
        if blocking:
            raise ModelDeleteConflict(blocking)

    def delete_model(self, tenant_id: str, ref: str) -> bool:
        """Archive a model and delete its local records without deleting S3 bytes.

        The all-version liveness check finishes before either registry is changed.
        Hub archival precedes the local commit so a hub failure cannot report
        success while leaving an active model that immediately reappears in the UI.
        """
        m = self.db.scalar(
            select(ModelORM).where(
                and_(
                    ModelORM.tenant_id == tenant_id,
                    (ModelORM.id == ref) | (ModelORM.name == ref),
                )
            )
        )
        versions = list(self.db.scalars(
            select(ModelVersionORM).where(
                and_(
                    ModelVersionORM.model_id == m.id,
                    ModelVersionORM.tenant_id == tenant_id,
                )
            )
        ).all()) if m else []
        self._ensure_versions_not_live(versions)
        # A model may exist only in the hub (no PG row). Resolve the name either
        # from PG or by treating the ref itself as the name.
        model_name = m.name if m else ref
        if self.hub and self.hub.enabled():
            self.hub.archive_model(model_name)
        if m:
            for version in versions:
                self.db.delete(version)
            self.db.delete(m)
            self.db.commit()
        return True

    # -- versions -----------------------------------------------------------

    # -- trainable versions (training form base-model selector) -------------

    # File suffixes accepted as a base checkpoint for continued training.
    # Ultralytics/timm continue training from PyTorch weights only; an exported
    # ONNX is an inference artifact and can never be a training base.
    _TRAINABLE_SUFFIXES = (".pt", ".pth")

    @staticmethod
    def _anchor_dir(uri: str) -> str:
        """Return the containing directory (s3://.../) of an object URI."""
        if "/" not in uri[len("s3://"):]:
            return uri.rstrip("/") + "/"
        return uri.rsplit("/", 1)[0] + "/"

    @classmethod
    def _pick_trainable(cls, candidates: list[str]) -> str:
        """Choose the best trainable checkpoint from candidate object URIs.

        ``best.pt`` is preferred (the run's fitness-best checkpoint); the
        remaining files are otherwise ordered by name for determinism.
        """
        usable = [
            u for u in dict.fromkeys(candidates)
            if u.lower().endswith(cls._TRAINABLE_SUFFIXES)
        ]
        if not usable:
            return ""
        for u in usable:
            if u.rsplit("/", 1)[-1].lower().startswith("best"):
                return u
        return sorted(usable)[0]

    def list_trainable_versions(self, tenant_id: str, task_type: str = "") -> list[dict[str, Any]]:
        """List model versions that can serve as a continued-training base.

        Walks the tenant's PG models (optionally filtered by task type) and
        every ready version, resolving a trainable checkpoint from:
        ``metadata.weights_uri`` -> ``file_path`` -> sibling files under the
        version's S3 prefix. Versions holding only an ``.onnx`` artifact are
        returned with ``trainable=false`` (shown disabled in the selector).
        """
        model_query = select(ModelORM).where(ModelORM.tenant_id == tenant_id)
        if task_type:
            model_query = model_query.where(ModelORM.task_type == task_type)
        models = self.db.scalars(model_query).all()
        out: list[dict[str, Any]] = []
        for m in models:
            version_rows = self.db.scalars(
                select(ModelVersionORM).where(
                    and_(
                        ModelVersionORM.model_id == m.id,
                        ModelVersionORM.tenant_id == tenant_id,
                    )
                )
            ).all()
            for v in version_rows:
                meta = v.metadata_ or {}
                candidates = [str(meta.get("weights_uri") or "")]
                candidates.append(str(v.file_path or ""))
                anchors: list[str] = []
                for u in list(candidates):
                    if u:
                        anchors.append(self._anchor_dir(u))
                scanned: set[str] = set()
                for anchor in dict.fromkeys(anchors):
                    if anchor in scanned:
                        continue
                    scanned.add(anchor)
                    try:
                        for f in self.s3.list_files_in_dir(anchor + "probe"):
                            candidates.append(f["uri"])
                    except Exception:
                        continue
                checkpoint = self._pick_trainable(candidates)
                arch = str(meta.get("arch") or "")
                # A timm state dict cannot be reloaded without knowing the
                # architecture id; classification versions that predate arch
                # recording are therefore not selectable as training bases.
                if m.task_type == "classification" and not arch:
                    checkpoint = ""
                out.append({
                    "model_id": m.id,
                    "model_name": m.name,
                    "version": v.version,
                    "task_type": m.task_type,
                    "trainable": bool(checkpoint),
                    "checkpoint_uri": checkpoint,
                    "arch": arch,
                    "size_bytes": int(v.size_bytes or 0),
                    "created_at": iso(v.created_at),
                })
        # Version labels are now job ids (not chronologically sortable), so
        # order newest first via the recorded creation timestamp; stable
        # re-sort groups by model name while keeping newest-first inside.
        out.sort(key=lambda x: x["created_at"] or "", reverse=True)
        out.sort(key=lambda x: x["model_name"])
        return out

    def create_version(
        self, tenant_id: str, model_id: str, version: str, description: str,
        file_path: str, size_bytes: int, metadata: dict, created_by: str,
    ) -> dict:
        """Create a model version, or replace it in place if it already exists.

        Re-uploading the same version id overwrites the artifact/file_path and
        metadata instead of failing the (model_id, version) unique constraint;
        the row id, created_at and original creator are preserved.
        """
        existing = self.db.scalar(
            select(ModelVersionORM).where(
                and_(
                    ModelVersionORM.model_id == model_id,
                    ModelVersionORM.tenant_id == tenant_id,
                    ModelVersionORM.version == version,
                )
            )
        )
        if existing is not None:
            existing.description = description or existing.description
            if file_path:
                existing.file_path = file_path
            if size_bytes:
                existing.size_bytes = size_bytes
            merged = dict(existing.metadata_ or {})
            merged.update(metadata or {})
            existing.metadata_ = merged
            existing.status = "ready"
            self.db.commit()
            self.db.refresh(existing)
            return _version_dict(existing)

        v = ModelVersionORM(
            id=new_id("mv"),
            model_id=model_id,
            tenant_id=tenant_id,
            version=version,
            description=description or "",
            file_path=file_path or "",
            size_bytes=size_bytes or 0,
            metadata_=metadata or {},
            created_by=created_by or "anonymous",
        )
        self.db.add(v)
        self.db.commit()
        self.db.refresh(v)
        return _version_dict(v)

    def update_version_metadata(
        self, tenant_id: str, model_id: str, version: str, patch: dict
    ) -> Optional[dict]:
        """Merge extra keys into a version's metadata (upsert semantics)."""
        v = self.db.scalar(
            select(ModelVersionORM).where(
                and_(
                    ModelVersionORM.model_id == model_id,
                    ModelVersionORM.tenant_id == tenant_id,
                    ModelVersionORM.version == version,
                )
            )
        )
        if v is None:
            return None
        merged = dict(v.metadata_ or {})
        merged.update(patch or {})
        v.metadata_ = merged
        self.db.commit()
        self.db.refresh(v)
        return _version_dict(v)

    def list_versions(self, tenant_id: str, model_id: str) -> list[dict]:
        """List versions of a model (newest first), without S3 artifact counts."""
        items, _ = self.list_versions_paged(
            tenant_id, model_id, 1, 10_000, with_artifact_counts=False
        )
        return items

    def list_versions_paged(
        self, tenant_id: str, model_id: str, page: int = 1, page_size: int = 20,
        with_artifact_counts: bool = True,
    ) -> tuple[list[dict], int]:
        """List versions of a model with server-side pagination (newest first).

        Artifact counts require one S3 LIST per version, so the version drawer
        computes them per page; bulk callers (list_models loading every
        version) pass with_artifact_counts=False and enrich only what the UI
        shows.
        """
        from app.core.util import page_bounds

        base = select(ModelVersionORM).where(
            and_(
                ModelVersionORM.model_id == model_id,
                ModelVersionORM.tenant_id == tenant_id,
            )
        )
        total = len(self.db.scalars(base).all())
        limit, offset = page_bounds(page, page_size)
        rows = self.db.scalars(
            base.order_by(ModelVersionORM.created_at.desc()).limit(limit).offset(offset)
        ).all()
        items = [_version_dict(v) for v in rows]
        if with_artifact_counts:
            self._attach_artifact_counts(items)
        return items, total

    def list_deployed(self, tenant_id: str) -> list[dict]:
        """List deployed/deploying/failed inference services.

        Base rows come from the local DB; live readiness, scheduler node,
        predictor image and rollout errors are enriched from the cluster via
        the read-only serving client. Cluster failures degrade gracefully to
        the last known DB state instead of failing the whole list.
        """
        from app.clients.k8s import serving_reader
        from app.core.config import config

        rows = self.db.scalars(
            select(ModelVersionORM)
            .where(
                and_(
                    ModelVersionORM.tenant_id == tenant_id,
                    ModelVersionORM.status.in_(("deployed", "deploying", "deploy_failed")),
                )
            )
            .order_by(ModelVersionORM.created_at.desc())
        ).all()
        ns = config.serving.namespace or "nereidforge"
        if tenant_id and tenant_id != "default":
            ns = f"nereidforge-{tenant_id}"
        services = []
        for v in rows:
            meta = v.metadata_ or {}
            service_name = str(meta.get("service", ""))
            if not service_name:
                continue
            model = self.db.get(ModelORM, v.model_id)
            if not model:
                continue
            is_onnx = str(meta.get("onnx_uri", "") or v.file_path or "").lower().endswith(".onnx")
            domain = str(meta.get("domain", "") or "")
            deploy_state = str(
                meta.get("deploy_status") or ("ready" if meta.get("deployed") else "deploying")
            )
            # Best-effort live view; when the cluster is unreachable trust the
            # DB state instead of marking every service not-ready.
            live = (
                serving_reader.describe(service_name, ns)
                if service_name and serving_reader.available
                else {}
            )
            db_error = str(meta.get("deploy_error", "") or "")
            error = db_error or str(live.get("error", "") or "")
            ready = deploy_state == "ready" and (not live or live.get("ready"))
            services.append({
                "name": service_name,
                "namespace": ns,
                "state": deploy_state,
                "ready": ready,
                "url": f"http://{domain}".rstrip("/") if domain else "",
                "domain": domain,
                "runtime": str(meta.get("runtime", "") or ""),
                "image": str(live.get("image", "") or ""),
                "model_uri": str(meta.get("onnx_uri", "") or v.file_path or ""),
                "node": str(live.get("node", "") or ""),
                "age": str(
                    meta.get("deployed_at", "")
                    or live.get("created_at", "")
                    or ""
                ),
                "error": error,
                "created_at": str(
                    meta.get("deployed_at", "")
                    or live.get("created_at", "")
                    or iso(v.created_at)
                ),
                "model_id": v.model_id,
                "model_name": model.name,
                "model_version": v.version,
                "model_type": "ONNX" if is_onnx else "PT",
                "model_framework": model.framework,
                "model_task_type": model.task_type,
            })
        return services

    def get_version(self, tenant_id: str, model_id: str, version: str) -> Optional[dict]:
        """Get a version by model+version."""
        v = self.db.scalar(
            select(ModelVersionORM).where(
                and_(
                    ModelVersionORM.model_id == model_id,
                    ModelVersionORM.tenant_id == tenant_id,
                    ModelVersionORM.version == version,
                )
            )
        )
        return _version_dict(v) if v else None

    def _resolve_trial_artifact(
        self, v: dict[str, Any], artifact_path: Optional[str] = None, cam: bool = False
    ) -> tuple[str, bool]:
        """_resolve_trial_artifact resolves and validates the inspect/trial artifact.

        Default resolution retains the ONNX-first behavior. An explicit relative
        path is accepted only after a live recursive bundle listing confirms
        that exact file, preventing traversal and arbitrary-object selection.
        The boolean result records whether the selected URI is a PyTorch checkpoint.
        """
        meta = v.get("metadata") or {}
        if not artifact_capability_permitted(meta, "trial"):
            artifact_kind = str(meta.get("artifact_format") or "this artifact")
            raise ValueError(f"{artifact_kind} does not support image trials in this release")
        # Trained versions store the .pt checkpoint under weights_uri; manually
        # uploaded versions carry it directly in file_path.
        onnx_uri = str(meta.get("onnx_uri") or "").strip()
        pt_uri = str(meta.get("weights_uri") or "").strip()
        if not pt_uri:
            fp = str(v.get("file_path") or "")
            if fp.lower().endswith(".pt") or fp.lower().endswith(".onnx"):
                pt_uri = fp

        requested = (artifact_path or "").strip()
        if requested:
            try:
                wanted = normalize_artifact_relative_path(requested)
            except ValueError as exc:
                raise ValueError("invalid artifact path") from exc
            anchor = _artifact_anchor(str(v.get("file_path") or ""), meta)
            siblings = self.s3.list_files_in_dir(anchor) if anchor.startswith("s3://") else []
            hit = next((f for f in siblings if f["name"] == wanted), None)
            if not hit:
                raise ValueError("artifact file not found in version prefix")
            is_pt = hit["name"].lower().endswith(".pt")
            if cam and not is_pt:
                raise ValueError("Grad-CAM requires a .pt (PyTorch) model")
            return str(hit["uri"]), is_pt
        if cam:
            if pt_uri.lower().endswith(".pt"):
                return pt_uri, True
            raise ValueError("Grad-CAM requires a .pt (PyTorch) model")
        # Prefer the converted ONNX when deploy recorded it: legacy checkpoints
        # may not be loadable by the trial backend while their ONNX export is.
        if onnx_uri.lower().endswith(".onnx"):
            return onnx_uri, False
        if pt_uri:
            return pt_uri, pt_uri.lower().endswith(".pt")
        raise ValueError(
            "this version has no usable weights; trial needs a .pt PyTorch "
            "checkpoint or a .onnx model"
        )

    @staticmethod
    def _saved_trial_decoder(meta: dict[str, Any], artifact_uri: str) -> Optional[dict[str, Any]]:
        """_saved_trial_decoder returns an isolated selection for one concrete artifact."""
        saved_by_uri = meta.get("trial_decoders")
        if not isinstance(saved_by_uri, dict):
            return None
        saved = saved_by_uri.get(artifact_uri)
        return deepcopy(saved) if isinstance(saved, dict) else None

    @staticmethod
    def _post_trial_request(ml_url: str, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        """_post_trial_request calls ML without discarding structured error details."""
        import httpx

        resp = httpx.post(
            f"{ml_url}{endpoint}", json=body, timeout=httpx.Timeout(180.0)
        )
        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001 - a downstream proxy may return plain text
            payload = None
        if resp.status_code != 200:
            data = payload if isinstance(payload, dict) else {}
            detail = data.get("error") or data.get("detail") or resp.text[:200]
            if not isinstance(detail, str):
                detail = str(detail)
            code = data.get("code")
            raise ModelTrialError(
                resp.status_code,
                detail or "model trial failed",
                str(code) if code else None,
            )
        if not isinstance(payload, dict):
            raise ValueError("ml backend returned an invalid trial response")
        return payload

    def inspect_trial(
        self,
        tenant_id: str,
        model_id: str,
        version: str,
        artifact_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """inspect_trial inspects the exact artifact that a model trial would execute.

        The ML Backend owns graph and decoder discovery. NF only adds the saved
        selection keyed by the resolved S3 URI and the upload's input-shape hint.
        """
        from app.core.config import config

        v = self.get_version(tenant_id, model_id, version)
        if not v:
            raise ValueError("model version not found")
        meta = v.get("metadata") or {}
        artifact_uri, _ = self._resolve_trial_artifact(v, artifact_path=artifact_path)
        ml_url = (config.labelstudio.ml_backend_url or "").rstrip("/")
        if not ml_url:
            raise ValueError("ml backend url not configured")
        inspection = self._post_trial_request(
            ml_url, "/try/inspect", _trial_request_hints(meta, artifact_uri)
        )
        return {
            **inspection,
            "artifact": artifact_uri,
            "saved_decoder": self._saved_trial_decoder(meta, artifact_uri),
            "input_shape": meta.get("input_shape"),
        }

    def save_trial_decoder(
        self,
        tenant_id: str,
        model_id: str,
        version: str,
        artifact_uri: str,
        decoder_config: dict[str, Any],
    ) -> None:
        """save_trial_decoder persists one artifact's successful selection atomically.

        Re-reading with a row lock where the database supports it prevents a
        trial from overwriting concurrent metadata updates; populate_existing
        also refreshes the ORM row loaded before the long inference call. Deep
        copies and reassignment are required for JSONB change tracking and
        preserve every sibling artifact entry and unrelated nested value.
        """
        v = self.db.scalar(
            select(ModelVersionORM).where(
                and_(
                    ModelVersionORM.model_id == model_id,
                    ModelVersionORM.tenant_id == tenant_id,
                    ModelVersionORM.version == version,
                )
            ).with_for_update().execution_options(populate_existing=True)
        )
        if v is None:
            return
        metadata = deepcopy(v.metadata_ or {})
        existing = metadata.get("trial_decoders")
        decoders = deepcopy(existing) if isinstance(existing, dict) else {}
        decoders[artifact_uri] = deepcopy(decoder_config)
        metadata["trial_decoders"] = decoders
        v.metadata_ = metadata
        self.db.commit()

    def try_predict(
        self,
        tenant_id: str,
        model_id: str,
        version: str,
        image_base64: str,
        conf: float = 0.25,
        iou: float = 0.7,
        cam: bool = False,
        class_index: Optional[int] = None,
        artifact_path: Optional[str] = None,
        decoder_config: Optional[dict[str, Any]] = None,
        top_k: int = 5,
    ) -> dict:
        """try_predict runs one uploaded image through a registry model version.

        Thin proxy to the shared ML backend's /try/predict route. No image is
        persisted here — bytes are relayed inline as base64. An explicit decoder
        wins; otherwise the selection saved for the resolved artifact is reused,
        falling back to automatic decoder discovery.
        """
        from app.core.config import config

        v = self.get_version(tenant_id, model_id, version)
        if not v:
            raise ValueError("model version not found")
        meta = v.get("metadata") or {}
        artifact_uri, _ = self._resolve_trial_artifact(v, artifact_path=artifact_path, cam=cam)
        ml_url = (config.labelstudio.ml_backend_url or "").rstrip("/")
        if not ml_url:
            raise ValueError("ml backend url not configured")

        if decoder_config is not None:
            effective_decoder = deepcopy(decoder_config)
        else:
            effective_decoder = self._saved_trial_decoder(meta, artifact_uri) or {"mode": "auto"}
        body = {
            **_trial_request_hints(meta, artifact_uri),
            "image_base64": image_base64,
            "conf": conf,
            "iou": iou,
            "cam": bool(cam),
            "decoder": effective_decoder,
            "top_k": top_k,
        }
        if class_index is not None:
            body["class_index"] = class_index
        result = self._post_trial_request(ml_url, "/try/predict", body)
        # A selection becomes reusable only after the ML Backend proves that it
        # can execute the artifact successfully; validation/inference failures
        # must leave the previously saved configuration untouched.
        self.save_trial_decoder(
            tenant_id, model_id, version, artifact_uri, effective_decoder
        )
        return result

    def delete_version(self, tenant_id: str, model_id: str, version: str) -> bool:
        """Archive one version and delete its local record, retaining S3 artifacts."""
        v = self.db.scalar(
            select(ModelVersionORM).where(
                and_(
                    ModelVersionORM.model_id == model_id,
                    ModelVersionORM.tenant_id == tenant_id,
                    ModelVersionORM.version == version,
                )
            )
        )
        if not v:
            return False
        self._ensure_versions_not_live([v])
        # Resolve the model name for hub archiving. Archive first: a failed hub
        # update must not leave an active hub-only version after local deletion.
        model = self.db.get(ModelORM, model_id)
        model_name = model.name if model else ""
        if model_name and self.hub and self.hub.enabled():
            self.hub.archive_model_version(model_name, version)
        self.db.delete(v)
        self.db.commit()
        return True

    # -- deployment lifecycle ----------------------------------------------
    # A deploy runs asynchronously (model_deploy AsyncTask, executed by the
    # worker) because ONNX conversion alone can take 5-10 minutes. The version
    # status flows: ready -> deploying -> deployed, or -> deploy_failed. The
    # metadata keys below are the API contract read by /serving and by the
    # Deployed Model Services list.

    def _get_version_row(
        self, tenant_id: str, model_id: str, version: str
    ) -> Optional[ModelVersionORM]:
        """Fetch a model version row (model_id + tenant + version)."""
        return self.db.scalar(
            select(ModelVersionORM).where(
                and_(
                    ModelVersionORM.model_id == model_id,
                    ModelVersionORM.tenant_id == tenant_id,
                    ModelVersionORM.version == version,
                )
            )
        )

    def mark_deploying(
        self, tenant_id: str, model_id: str, version: str,
        service: str, runtime: str, domain: str, task_id: str,
    ) -> None:
        """Record that an async deploy task was enqueued for a version."""
        v = self._get_version_row(tenant_id, model_id, version)
        if not v:
            return
        md = dict(v.metadata_ or {})
        md.update({
            "deployed": False,
            "deploy_status": "deploying",
            "deploy_error": "",
            "deploy_task_id": task_id,
            "service": service,
            "runtime": runtime,
            "domain": domain,
        })
        md.pop("deployed_at", None)
        v.metadata_ = md
        v.status = "deploying"
        self.db.commit()

    def mark_deployed(
        self, tenant_id: str, model_id: str, version: str,
        service: str, runtime: str, domain: str,
    ) -> None:
        """Mark a model version as deployed (store service/runtime/domain in metadata)."""
        from datetime import datetime, timezone

        v = self._get_version_row(tenant_id, model_id, version)
        if not v:
            return
        md = dict(v.metadata_ or {})
        md.update({
            "deployed": True,
            "deploy_status": "ready",
            "deploy_error": "",
            "service": service,
            "runtime": runtime,
            "domain": domain,
            "deployed_at": md.get("deployed_at") or datetime.now(timezone.utc).isoformat(),
        })
        v.metadata_ = md
        v.status = "deployed"
        self.db.commit()

    def mark_deploy_failed(
        self, tenant_id: str, model_id: str, version: str, error: str
    ) -> None:
        """Record an async deploy failure with its human-readable reason."""
        v = self._get_version_row(tenant_id, model_id, version)
        if not v:
            return
        md = dict(v.metadata_ or {})
        md["deployed"] = False
        md["deploy_status"] = "failed"
        md["deploy_error"] = (error or "")[:2000]
        v.metadata_ = md
        v.status = "deploy_failed"
        self.db.commit()

    def mark_undeployed(self, tenant_id: str, model_id: str, version: str) -> None:
        """Clear a version's deployed status."""
        v = self._get_version_row(tenant_id, model_id, version)
        if not v:
            return
        md = dict(v.metadata_ or {})
        for key in (
            "deployed", "service", "domain", "deploy_status", "deploy_error",
            "deploy_task_id", "undeploy_task_id", "deployed_at",
        ):
            md.pop(key, None)
        v.metadata_ = md
        v.status = "ready"
        self.db.commit()

    def mark_terminating(
        self, tenant_id: str, model_id: str, version: str, undeploy_task_id: str
    ) -> None:
        """Record that an undeploy was submitted and is being verified.

        The row stays visible in Deployments (status stays "deployed") with
        deploy_status="terminating" until the worker confirms the live object
        is gone; this prevents DB/cluster divergence if the deletion fails.
        """
        v = self._get_version_row(tenant_id, model_id, version)
        if not v:
            return
        md = dict(v.metadata_ or {})
        md["deploy_status"] = "terminating"
        md["deploy_error"] = ""
        md["undeploy_task_id"] = undeploy_task_id
        v.metadata_ = md
        self.db.commit()

    def mark_undeploy_failed(
        self, tenant_id: str, model_id: str, version: str, error: str
    ) -> None:
        """Record that an undeploy could not be verified (service still present)."""
        v = self._get_version_row(tenant_id, model_id, version)
        if not v:
            return
        md = dict(v.metadata_ or {})
        md["deployed"] = True
        md["deploy_status"] = "undeploy_failed"
        md["deploy_error"] = (error or "")[:2000]
        md.pop("undeploy_task_id", None)
        v.metadata_ = md
        v.status = "deployed"
        self.db.commit()

    def cancel_deploy_task(
        self, tenant_id: str, model_id: str, version: str
    ) -> bool:
        """Cancel a still-running model_deploy task for a version.

        The worker checks the task state at stage boundaries (before/after the
        long ONNX conversion) and skips the cluster deploy when it sees
        "cancelled". Returns True when a live task was cancelled.
        """
        from app.models.orm import AsyncTask

        v = self._get_version_row(tenant_id, model_id, version)
        if not v:
            return False
        task_id = str((v.metadata_ or {}).get("deploy_task_id", "") or "")
        if not task_id:
            return False
        t = self.db.get(AsyncTask, task_id)
        if t and t.status in ("pending", "processing", "paused"):
            t.status = "cancelled"
            self.db.commit()
            return True
        return False

    def _versions_for(self, model_id: str, tenant_id: str) -> list[dict]:
        """Load versions for a model as API dicts."""
        return self.list_versions(tenant_id, model_id)

    # -- artifact streaming -------------------------------------------------

    def stream_artifact(self, tenant_id: str, model_id: str, version: str, out: Any) -> int:
        """Stream a version's artifact bytes from S3, returning bytes written."""
        v = self.db.scalar(
            select(ModelVersionORM).where(
                and_(
                    ModelVersionORM.model_id == model_id,
                    ModelVersionORM.tenant_id == tenant_id,
                    ModelVersionORM.version == version,
                )
            )
        )
        if not v or not v.file_path:
            return 0
        filer = self.s3.s3_uri_to_filer_path(v.file_path)
        if not filer:
            return 0
        return self.s3.stream_file(filer, out)

    def _version_anchor_uri(self, v: ModelVersionORM) -> str:
        """S3 URI anchoring a version's artifact prefix.

        New uploads carry an explicit bundle root because their primary file
        may be nested. Legacy versions retain ONNX-first primary-file anchoring.
        """
        return _artifact_anchor(str(v.file_path or ""), v.metadata_ or {})

    def list_version_files(
        self, tenant_id: str, model_id: str, version: str,
    ) -> Optional[list[dict]]:
        """List all files in a model version's S3 prefix (the artifact bundle).

        Returns None when the version does not exist; an empty list when it
        exists but its prefix holds no list-able S3 objects. S3 errors are
        display-only and degrade to an empty list.
        """
        v = self.db.scalar(
            select(ModelVersionORM).where(
                and_(
                    ModelVersionORM.model_id == model_id,
                    ModelVersionORM.tenant_id == tenant_id,
                    ModelVersionORM.version == version,
                )
            )
        )
        if v is None:
            return None
        anchor = self._version_anchor_uri(v)
        if not anchor.startswith("s3://"):
            return []
        try:
            return self.s3.list_files_in_dir(anchor)
        except Exception:  # noqa: BLE001 - file browser must never 500 the page
            return []

    def stream_version_file(
        self, tenant_id: str, model_id: str, version: str, rel_path: str, out: Any,
    ) -> Optional[str]:
        """Stream one file from a version's artifact prefix; return its s3 URI.

        Nested paths are addressable only when the live bundle LIST contains
        that exact normalized relative name, so traversal and arbitrary-key
        access are impossible. None means version/path not found.
        """
        files = self.list_version_files(tenant_id, model_id, version)
        if files is None:
            return None
        try:
            wanted = normalize_artifact_relative_path(rel_path)
        except ValueError:
            return None
        match = next((f for f in files if f["name"] == wanted), None)
        if not match:
            return None
        filer = self.s3.s3_uri_to_filer_path(match["uri"])
        if not filer:
            return None
        self.s3.stream_file(filer, out)
        return str(match["uri"])

    # -- hub registration ---------------------------------------------------

    def register_to_hub(self, name: str, uri: str, version: str, model_format: str, metadata: dict) -> bool:
        """Register a model+version to Kubeflow Hub via the Python SDK.

        Returns True on success and False when the hub is not configured/enabled
        (best-effort; the local registry remains the source of truth).
        """
        if not self.hub:
            return False
        if not self.hub.enabled():
            return False
        fmt_name = metadata.get("framework", "pytorch")
        # The model-registry SDK's ModelVersion.custom_properties only accepts
        # scalar values (bool/int/float/str). Lists/dicts that are valid in the
        # PG metadata (e.g. input_shape: []) crash Pydantic validation and make
        # the whole register a no-op. Serialize anything non-scalar to JSON so
        # registration survives regardless of what the frontend/worker sent.
        from json import dumps as _json_dumps

        hub_meta: dict = {}
        for k, val in metadata.items():
            if val is None:
                continue
            if isinstance(val, (bool, int, float, str)):
                hub_meta[k] = val
            else:
                hub_meta[k] = _json_dumps(val)
        self.hub.register_model(
            name=name,
            uri=uri,
            model_format_name=fmt_name,
            model_format_version=model_format or "1",
            version=version,
            metadata=hub_meta,
        )
        return True
