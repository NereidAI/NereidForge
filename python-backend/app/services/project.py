# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Annotation project service: Label Studio provisioning, tasks, pre-annotation.

Mirrors the Go ProjectService (internal/service/project.go):
- project CRUD with Label Studio provisioning (project + S3 storage + ML backend)
- task list/save/export via LS
- semi-auto pre-annotation (NF-driven ML backend /predict -> LS predictions)
- batch accept (predictions -> annotations, empty -> confirmed-no-target)
- annotated export back to a dataset version
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.clients.labelstudio import (
    LabelStudioClient,
    build_data_yaml,
    build_label_config,
    resolve_ls_image,
)
from app.clients.s3 import SeaweedFSClient
from app.core.config import config
from app.core.util import iso, new_id, page_bounds
from app.models.orm import AnnotationProject, AsyncTask, Dataset, DatasetVersion, Job
from app.models.orm import Model as ModelRow
from app.models.orm import ModelVersion as ModelVersionRow
from app.models.orm import User

logger = logging.getLogger(__name__)


def _image_dims(data: bytes) -> tuple[int, int] | None:
    """Parse width/height out of an image byte stream.

    Supports PNG (IHDR chunk) and JPEG (SOF marker scan). CVAT video frames are
    returned as PNG even for ``quality=compressed``; returning None for them used
    to force a 640x480 fallback, which scaled the ML backend's percentage boxes
    against the wrong width and caused the boxes to drift from the objects.

    Returns None when the stream is neither a parseable PNG nor JPEG.
    """
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        # IHDR is the first chunk: 8-byte signature, 4-byte length, 4-byte "IHDR",
        # then 4-byte width and 4-byte height (all big-endian).
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        if width and height:
            return int(width), int(height)
        return None
    # JPEG: scan for a Start-Of-Frame marker.
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height = (data[i + 5] << 8) | data[i + 6]
            width = (data[i + 7] << 8) | data[i + 8]
            return int(width), int(height)
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = (data[i + 2] << 8) | data[i + 3]
        if seg_len < 2:
            return None
        i += 2 + seg_len
    return None


# Backwards-compatible alias (older call sites used the JPEG-only name).
_jpeg_dims = _image_dims


def _project_dict(p: AnnotationProject) -> dict[str, Any]:
    """Serialize an AnnotationProject to the API response shape."""
    return {
        "id": p.id,
        "tenant_id": p.tenant_id,
        "name": p.name,
        "description": p.description,
        "annotation_tool": p.annotation_tool,
        "task_type": p.task_type,
        "status": p.status,
        "schema": p.schema_ or {},
        "import_uri": p.import_uri,
        "dataset_id": p.dataset_id,
        "dataset_version": p.dataset_version,
        "progress": p.progress,
        "total_items": p.total_items,
        "annotated_items": p.annotated_items,
        "assignee_id": p.assignee_id,
        "assignee_name": "",
        "external_id": p.external_id,
        "external_url": p.external_url,
        "preannotate_model": p.preannotate_model or {},
        "created_by": p.created_by,
        "created_at": iso(p.created_at),
        "updated_at": iso(p.updated_at),
    }


def _attach_assignee_names(db: "Session", rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Populate assignee_name (full_name > username) for serialized projects.

    One batched query for all referenced assignee ids; missing users resolve
    to an empty display name rather than raising.
    """
    ids = {r.get("assignee_id") for r in rows if r.get("assignee_id")}
    if not ids:
        return rows
    users = db.scalars(select(User).where(User.id.in_(ids))).all()
    name_by_id = {
        u.id: (u.full_name or u.username or "")
        for u in users
    }
    for r in rows:
        r["assignee_name"] = name_by_id.get(r.get("assignee_id"), "")
    return rows


class ProjectService:
    """Manages annotation projects and their Label Studio integration."""

    def __init__(
        self, db: Session, s3: SeaweedFSClient, ls: LabelStudioClient,
        cvat: Any = None, dataset: Any = None, ml_backend: Any = None,
        s3_access_key: str = "", s3_secret_key: str = "",
    ) -> None:
        self.db = db
        self.s3 = s3
        self.ls = ls
        self.cvat = cvat
        self.dataset = dataset
        self.ml_backend = ml_backend
        self.s3_access_key = s3_access_key
        self.s3_secret_key = s3_secret_key

    # -- CRUD ---------------------------------------------------------------

    @staticmethod
    def _normalize_schema(task_type: str, schema: dict | None) -> dict:
        """Ensure a visual detection/tracking project carries an explicit label set.

        When the user created the project without labels (and no raw_config), the
        label tool provisioning previously fell back to the 8 default COCO labels
        in LS/CVAT while the NF schema stayed empty, so the NF annotation UI used
        a different hard-coded fallback. Persisting the canonical defaults on the
        project makes DB schema, LS/CVAT config, and NF UI agree.
        """
        from app.clients.labelstudio import DEFAULT_DETECTION_LABELS

        s = dict(schema or {})
        # Text/NLP tasks and OCR have their own label semantics; leave them alone.
        text_types = {"text-classification", "ner", "summarization", "text-generation", "chat-rlhf", "ocr"}
        if task_type in text_types:
            return s
        # Explicit raw config or an explicit label list wins.
        if isinstance(s.get("raw_config"), str) and s["raw_config"].strip():
            return s
        labels = s.get("labels") or []
        if not labels:
            s["labels"] = list(DEFAULT_DETECTION_LABELS)
        return s

    def _derive_schema_from_version(
        self, tenant_id: str, dataset_id: str, dataset_version: int
    ) -> dict:
        """Derive an LS schema (control type + labels) from a dataset version.

        Thin wrapper over ``ls_dataset.derive_schema``; returns {} when the
        version is missing, raw, or its metadata cannot be read.
        """
        try:
            ver = self.dataset.get_dataset_version(
                tenant_id, dataset_id, dataset_version)
            if not ver or not ver.uri:
                return {}
            from app.services.ls_dataset import derive_schema

            filer = self.s3.s3_uri_to_filer_path(ver.uri)
            rel_keys = [key[len(filer) + 1:] for key in self.s3.walk(filer)]
            return derive_schema(self.s3, filer, ver.format or "", rel_keys)
        except Exception:
            logger.warning("derive schema from %s v%s failed",
                           dataset_id, dataset_version, exc_info=True)
            return {}

    def create(
        self, tenant_id: str, host: str, name: str, description: str, annotation_tool: str,
        task_type: str, schema: dict, import_uri: str, dataset_id: str,
        dataset_version: int, created_by: str, assignee_id: str = "",
    ) -> dict:
        """Create an annotation project; provisions a CVAT or LS project based on task_type.

        Routing rule (ADR-010 D1): video-tracking -> CVAT, everything else ->
        Label Studio. The stored annotation_tool field is normalized to the
        tool that was actually provisioned so all later routing (tasks,
        preannotate, export, frontend) agrees with reality.
        """
        external_id, external_url = "", ""
        # When the project references a dataset version and the user supplied
        # no label set/raw config, derive the schema from the version content
        # (data.yaml names + rectangle/polygon, or class_names.json choices)
        # so the provisioned LS config matches the actual corpus labels.
        if dataset_id and not (schema or {}).get("raw_config") \
                and not (schema or {}).get("labels"):
            derived = self._derive_schema_from_version(
                tenant_id, dataset_id, dataset_version or 1)
            if derived:
                schema = {**(schema or {}), **derived}
        # Normalize schema: visual detection/tracking projects created without an
        # explicit label set get the canonical default labels persisted on the
        # project, so the stored schema, the provisioned LS/CVAT config, and the
        # NF annotation UI all show the same labels (single source of truth).
        schema = self._normalize_schema(task_type or "image-detection", schema)
        # Route by task_type: video-tracking -> CVAT, others -> Label Studio
        use_cvat = (task_type == "video-tracking")
        if use_cvat:
            if not (self.cvat and self.cvat.enabled()):
                raise ValueError("CVAT is not configured (video-tracking projects require CVAT)")
            annotation_tool = "cvat"
            cvat_id, cvat_url = self._provision_cvat_project(
                tenant_id, host, name, description, schema, import_uri, dataset_id, dataset_version
            )
            external_id = str(cvat_id) if cvat_id else ""
            external_url = cvat_url
        else:
            if not self.ls.enabled():
                raise ValueError("label studio not configured")
            annotation_tool = "label-studio"
            ls_id, ls_url = self._provision_ls_project(
                tenant_id, host, name, description, schema, import_uri,
                dataset_id, dataset_version, task_type,
            )
            external_id = str(ls_id) if ls_id else ""
            external_url = ls_url
        p = AnnotationProject(
            id=new_id("prj"),
            tenant_id=tenant_id,
            name=name,
            description=description or "",
            annotation_tool=annotation_tool,
            task_type=task_type or "image-detection",
            status="pending",
            schema_=schema or {},
            import_uri=import_uri or "",
            dataset_id=dataset_id or "",
            dataset_version=dataset_version or 1,
            external_id=external_id,
            external_url=external_url,
            assignee_id=assignee_id or "",
            created_by=created_by or "anonymous",
        )
        self.db.add(p)
        self.db.commit()
        self.db.refresh(p)
        return _project_dict(p)

    def list(
        self, tenant_id: str, page: int, page_size: int, dataset_id: str = "", status: str = "",
        assignee_id: str = "", search: str = "",
    ) -> tuple[list[dict], int]:
        """List annotation projects with filters and pagination.

        assignee_id filters by the assigned user id; the literal "none" matches
        unassigned projects, "any" matches projects with any assignee.
        """
        from sqlalchemy import func

        query = select(AnnotationProject).where(AnnotationProject.tenant_id == tenant_id)
        count_q = select(func.count(AnnotationProject.id)).where(AnnotationProject.tenant_id == tenant_id)
        if dataset_id:
            query = query.where(AnnotationProject.dataset_id == dataset_id)
            count_q = count_q.where(AnnotationProject.dataset_id == dataset_id)
        if status:
            query = query.where(AnnotationProject.status == status)
            count_q = count_q.where(AnnotationProject.status == status)
        if assignee_id == "none":
            query = query.where((AnnotationProject.assignee_id == "") | (AnnotationProject.assignee_id.is_(None)))
            count_q = count_q.where((AnnotationProject.assignee_id == "") | (AnnotationProject.assignee_id.is_(None)))
        elif assignee_id == "any":
            query = query.where(AnnotationProject.assignee_id != "")
            count_q = count_q.where(AnnotationProject.assignee_id != "")
        elif assignee_id:
            query = query.where(AnnotationProject.assignee_id == assignee_id)
            count_q = count_q.where(AnnotationProject.assignee_id == assignee_id)
        if search:
            query = query.where(AnnotationProject.name.ilike(f"%{search.strip()}%"))
            count_q = count_q.where(AnnotationProject.name.ilike(f"%{search.strip()}%"))
        total = self.db.scalar(count_q) or 0
        limit, offset = page_bounds(page, page_size)
        rows = self.db.scalars(query.order_by(AnnotationProject.created_at.desc()).limit(limit).offset(offset)).all()
        return _attach_assignee_names(self.db, [_project_dict(p) for p in rows]), total

    def get(self, tenant_id: str, id: str) -> Optional[dict]:
        """Get an annotation project by id."""
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p:
            return None
        return _attach_assignee_names(self.db, [_project_dict(p)])[0]

    def update(self, tenant_id: str, id: str, patch: dict[str, Any]) -> Optional[dict]:
        """Update project fields from a patch dict."""
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p:
            return None
        # ORM column is `schema_`; the API/frontend sends the JSON key `schema`.
        # Without this alias a generic PATCH silently dropped the whole schema
        # (add_labels() below writes schema_ directly and was unaffected).
        if "schema" in patch and "schema_" not in patch:
            patch = {**patch, "schema_": patch["schema"]}
        for k, v in patch.items():
            if hasattr(p, k) and k not in ("id", "created_at"):
                setattr(p, k, v)
        self.db.commit()
        self.db.refresh(p)
        return _project_dict(p)

    def delete(self, tenant_id: str, id: str) -> bool:
        """Delete an annotation project (and its LS project).

        Blocks deletion only while work is still in flight:
        - Active async tasks (pre-annotation, batch accept)
        - Active/pending training jobs using a dataset version exported here

        Dataset versions previously exported from this project are independent
        data and are kept; their provenance link is detached (source_project_id
        set to NULL) instead of blocking the deletion.
        """
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p:
            return False

        # Check for active async tasks (project id lives in the JSON params)
        async_ref = self.db.scalar(
            select(AsyncTask).where(
                and_(
                    AsyncTask.params["project_id"].astext == id,
                    AsyncTask.tenant_id == tenant_id,
                    AsyncTask.status.in_(["pending", "running"]),
                )
            )
        )
        if async_ref:
            raise ValueError(f"project {id} has active async task {async_ref.id} (cannot delete)")

        # Check for active/pending training jobs using a dataset version that
        # was exported from this project. Every such version is considered, so
        # one blocked job is not hidden behind an unrelated active job.
        active_jobs = self.db.scalars(
            select(Job).where(
                and_(
                    Job.type == "training",
                    Job.tenant_id == tenant_id,
                    Job.status.in_(["pending", "running"]),
                )
            )
        ).all()
        for job_ref in active_jobs:
            ds_id = job_ref.params.get("dataset_id", "")
            ver_str = job_ref.params.get("dataset_version", "1")
            try:
                ver = int(ver_str)
            except ValueError:
                ver = 1
            if not ds_id:
                continue
            ds_ver = self.db.scalar(
                select(DatasetVersion).where(
                    and_(
                        DatasetVersion.dataset_id == ds_id,
                        DatasetVersion.version == ver,
                        DatasetVersion.source_project_id == id,
                        DatasetVersion.tenant_id == tenant_id,
                    )
                )
            )
            if ds_ver:
                raise ValueError(f"project {id} is used by training job {job_ref.id} (cannot delete)")

        # Detach exported dataset versions: they remain usable datasets while
        # the project that produced them is removed.
        exported = self.db.scalars(
            select(DatasetVersion).where(
                and_(
                    DatasetVersion.source_project_id == id,
                    DatasetVersion.tenant_id == tenant_id,
                )
            )
        ).all()
        for dv in exported:
            dv.source_project_id = None

        if p.external_id:
            use_cvat = (p.task_type == "video-tracking")
            if use_cvat and self.cvat and self.cvat.enabled():
                try:
                    self.cvat.delete_project(int(p.external_id))
                except Exception:
                    pass
            elif self.ls.enabled():
                try:
                    self.ls._delete(f"/api/projects/{p.external_id}/")
                except Exception:
                    pass
        self.db.delete(p)
        self.db.commit()
        return True

    def start(self, tenant_id: str, id: str) -> Optional[dict]:
        """Mark a project as in_progress."""
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p:
            return None
        p.status = "in_progress"
        self.db.commit()
        self.db.refresh(p)
        return _project_dict(p)

    def complete(self, tenant_id: str, id: str) -> Optional[dict]:
        """Mark a project as completed (progress 100).

        For video (CVAT) projects this also accepts every underlying job
        (stage=acceptance, state=completed) so CVAT's own task status reaches
        'completed' and a later progress sync does not reopen the project.
        """
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p:
            return None
        if p.task_type == "video-tracking" and p.external_id and self.cvat and self.cvat.enabled():
            try:
                self._accept_all_cvat_jobs(int(p.external_id))
            except Exception:
                # NF completion must succeed even if the CVAT transition fails;
                # log the cause but still finalize the NF-side state.
                logger.exception("failed to accept CVAT jobs for project %s", p.external_id)
        p.status = "completed"
        p.progress = 100
        p.annotated_items = p.total_items
        self.db.commit()
        self.db.refresh(p)
        return _project_dict(p)

    def _accept_all_cvat_jobs(self, cvat_project_id: int) -> int:
        """Accept every job of every task under a CVAT project.

        Returns the number of jobs transitioned. Pages through the project's
        tasks (each video is one task with one segment/job here).
        """
        accepted = 0
        page = 1
        while True:
            data = self.cvat.list_tasks(cvat_project_id, page, 100)
            tasks = data.get("results", []) or []
            for t in tasks:
                for job in self.cvat.list_jobs(int(t.get("id", 0))):
                    if job.get("stage") == "acceptance" and job.get("state") == "completed":
                        accepted += 1
                        continue
                    self.cvat.update_job(int(job["id"]), stage="acceptance", state="completed")
                    accepted += 1
            if not data.get("next") or not tasks:
                break
            page += 1
        return accepted

    def cancel(self, tenant_id: str, id: str) -> Optional[dict]:
        """Mark a project as cancelled."""
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p:
            return None
        p.status = "cancelled"
        self.db.commit()
        self.db.refresh(p)
        return _project_dict(p)

    def add_labels(self, tenant_id: str, project_id: str, names: list[str]) -> Optional[dict]:
        """Append label names to a project while annotating.

        Persists the merged label set into schema.labels and syncs it to the
        underlying tool: CVAT project labels (PATCH /api/projects) or the LS
        project label_config. Duplicate names are ignored; whitespace trimmed.
        Raises ValueError on an empty/invalid request or tool failure.
        """
        clean = [n.strip() for n in (names or []) if n and n.strip()]
        if not clean:
            raise ValueError("no label names provided")
        # De-duplicate while preserving order (CVAT/LS reject duplicate names).
        seen_in: set[str] = set()
        clean = [n for n in clean if not (n in seen_in or seen_in.add(n))]
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == project_id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p:
            return None
        schema = dict(p.schema_ or {})
        current = [str(x) for x in (schema.get("labels") or [])]
        have = set(current)
        added = [n for n in clean if n not in have]
        if not added:
            return _project_dict(p)
        schema["labels"] = current + added
        p.schema_ = schema

        if p.task_type == "video-tracking":
            if not (self.cvat and self.cvat.enabled() and p.external_id):
                raise ValueError("CVAT is not configured for this project")
            self.cvat.add_labels(int(p.external_id), added)
        else:
            if not (self.ls and self.ls.enabled() and p.external_id):
                raise ValueError("label studio is not configured for this project")
            self.ls.update_label_config(int(p.external_id), build_label_config(schema))

        self.db.commit()
        self.db.refresh(p)
        return _project_dict(p)

    # -- LS provisioning ----------------------------------------------------

    def _provision_ls_project(
        self, tenant_id: str, host: str, name: str, description: str, schema: dict,
        import_uri: str, dataset_id: str, dataset_version: int,
        task_type: str = "",
    ) -> tuple[Optional[int], str]:
        """Create an LS project + S3 storage + ML backend + import tasks."""
        label_config = build_label_config(schema)
        if not label_config:
            return None, ""
        ls_proj = self.ls.create_project(name, description, label_config)
        ls_id = int(ls_proj.get("id", 0))

        # S3 source storage so LS can load images from SeaweedFS (presign=False).
        # LS runs in-cluster, so it must use the cluster S3 endpoint (the dev host's
        # localhost port-forward is not reachable from the LS pod).
        try:
            ls_s3_endpoint = config.storage.s3_cluster_endpoint or self.s3.s3_endpoint()
            self.ls.create_s3_storage(
                ls_id, self.s3.s3_bucket(), ls_s3_endpoint,
                self.s3_access_key, self.s3_secret_key,
            )
        except Exception:
            # Do NOT fail project creation, but log loudly: without this storage
            # LS cannot resolve bare s3:// task images ("issue loading URL from
            # $image value"). It can be attached later, but a silent pass made the
            # missing storage invisible and images 404 in the LS UI.
            logger.exception("LS project %s: create_s3_storage failed; images won't load in LS UI", ls_id)

        # ML backend registration uses the cluster-internal URL (LS pod reachable).
        if self.ls.ml_backend_cluster:
            try:
                self.ls.register_ml_backend(ls_id, self.ls.ml_backend_cluster, f"{name}-ppyoloe")
            except Exception:
                logger.warning("LS project %s: register_ml_backend failed", ls_id, exc_info=True)

        # Import image tasks from the dataset version (s3:// paths).
        # Large versions are imported by an async worker task so the
        # project-creation request returns immediately.
        from app.services.ls_dataset import ASYNC_THRESHOLD, discover_images

        ver = self.dataset.get_dataset_version(
            tenant_id, dataset_id, dataset_version) \
            if self.dataset and dataset_id else None
        image_count = 0
        if ver and ver.uri:
            try:
                vfiler = self.s3.s3_uri_to_filer_path(ver.uri)
                rel = [k[len(vfiler) + 1:] for k in self.s3.walk(vfiler)]
                image_count = len(discover_images(ver.format or "", rel))
            except Exception:
                image_count = 0
        if image_count >= ASYNC_THRESHOLD:
            atask = AsyncTask(
                id=new_id("async"), tenant_id=tenant_id, type="ls_import",
                status="pending", progress=0,
                params={
                    "ls_id": ls_id, "dataset_id": dataset_id,
                    "dataset_version": dataset_version, "schema": schema,
                    "task_type": task_type,
                },
            )
            self.db.add(atask)
            self.db.commit()
            logger.info("LS project %s: scheduled async ls_import for %s images",
                        ls_id, image_count)
        else:
            tasks = self.build_tasks_from_dataset(
                tenant_id, dataset_id, dataset_version, schema, task_type)
            if tasks:
                self.ls.import_tasks(ls_id, tasks)

        url = f"{self.ls.public_url}/projects/{ls_id}"
        return ls_id, url

    # -- CVAT provisioning --------------------------------------------------

    def _provision_cvat_project(
        self, tenant_id: str, host: str, name: str, description: str, schema: dict,
        import_uri: str, dataset_id: str, dataset_version: int,
    ) -> tuple[Optional[int], str]:
        """Create a CVAT project + one task per video, importing from SeaweedFS.

        Per ADR-010 D1/D4: CVAT is used for video-tracking only. Each video
        file in the dataset version becomes its own CVAT task (the natural
        unit for track annotation). Data is imported via the deployment-level
        cloud storage that mounts the SeaweedFS bucket. No ML backend
        registration (D2: NF ML Backend is called directly for semi-auto).
        """
        from app.clients.cvat import build_cvat_label_spec

        labels = build_cvat_label_spec(schema)
        cvat_proj = self.cvat.create_project(name, description, labels)
        cvat_id = int(cvat_proj.get("id", 0))
        if not cvat_id:
            return None, ""

        cloud_storage_id = self.cvat.find_cloud_storage()
        if not cloud_storage_id:
            logger.warning(
                "cvat project %s: no SeaweedFS cloud storage configured on CVAT; "
                "tasks will be created without data", cvat_id,
            )

        # One CVAT task per video file (falls back to a single image-set
        # task when the dataset version has no videos, so the project is
        # never empty).
        data_files = self._build_cvat_server_files(tenant_id, dataset_id, dataset_version)
        for display_name, server_paths in data_files:
            try:
                task = self.cvat.create_task(cvat_id, display_name)
                task_id = int(task.get("id", 0))
                if not task_id:
                    continue
                self.cvat.upload_data(task_id, server_paths, cloud_storage_id=cloud_storage_id)
                # Upload is async; wait briefly so the task is usable right
                # away, but do not fail provisioning on timeout.
                self.cvat.wait_task_ready(task_id, poll_interval=3, max_wait=120)
            except Exception as e:
                logger.warning("cvat task import for %s failed: %s", display_name, e)

        url = f"{self.cvat._public_url()}/projects/{cvat_id}"
        return cvat_id, url

    def _build_cvat_server_files(
        self, tenant_id: str, dataset_id: str, dataset_version: int,
    ) -> list[tuple[str, list[str]]]:
        """Build (task display name, [server file paths]) pairs for CVAT import.

        Videos come first (one task each, the video-tracking use case).
        When no videos exist, all images are returned as a single entry so
        the project still gets data. Paths are bucket-relative (cloud
        storage root), which is what CVAT server_files expects.
        """
        files: list[tuple[str, list[str]]] = []
        if dataset_id:
            for entry in self._list_version_files(tenant_id, dataset_id, dataset_version, "videos"):
                stem = entry.rsplit("/", 1)[-1]
                stem = stem.rsplit(".", 1)[0] if "." in stem else stem
                files.append((stem, [entry]))
        if not files:
            images = self._list_version_files(tenant_id, dataset_id, dataset_version, "images")
            if images:
                files.append(("image-set", images))
        return files

    def _cvat_server_paths(
        self, tenant_id: str, dataset_id: str, dataset_version: int, sub: str,
    ) -> list[str]:
        """Return bucket-relative paths of all files under a version's sub dir.

        CVAT cloud storage is mounted on the SeaweedFS bucket root, so
        server_files keys must NOT carry the filer-style 'buckets/<name>/'
        prefix.
        """
        if not dataset_id or not self.dataset:
            return []
        ver = self.dataset.get_dataset_version(tenant_id, dataset_id, dataset_version)
        if not ver or not ver.uri:
            return []
        filer = self.s3.s3_uri_to_filer_path(ver.uri)
        if not filer:
            return []
        bucket_prefix = f"/buckets/{self.s3.s3_bucket()}/"
        if filer.startswith(bucket_prefix):
            filer = filer[len(bucket_prefix):]
        try:
            entries = self.s3.list_entries(f"{self.s3.s3_uri_to_filer_path(ver.uri)}/{sub}")
        except Exception:
            return []
        paths = []
        for e in entries:
            if e.is_dir:
                continue
            paths.append(f"{filer}/{sub}/{e.name}".lstrip("/"))
        return paths

    def _list_version_files(
        self, tenant_id: str, dataset_id: str, dataset_version: int, sub: str,
    ) -> list[str]:
        """List bucket-relative file paths under a version's sub directory.

        videos: any video extension; images: jpg/jpeg/png only.
        """
        video_ext = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v")
        img_ext = (".jpg", ".jpeg", ".png")
        wanted = video_ext if sub == "videos" else img_ext
        out = []
        for p in self._cvat_server_paths(tenant_id, dataset_id, dataset_version, sub):
            name = p.rsplit("/", 1)[-1].lower()
            if name.endswith(wanted):
                out.append(p)
        return out

    def build_tasks_from_dataset(
        self, tenant_id: str, dataset_id: str, dataset_version: int,
        schema: dict | None = None, task_type: str = "",
    ) -> list[dict]:
        """Build LS task list from a dataset version.

        Visual versions (raw/yolo/coco/imagenet, flat or split layouts) are
        converted by the universal canonical->LS converter
        (``app/services/ls_dataset.py``): labels become saved RectangleLabels/
        PolygonLabels/Choices annotations and raw images become unlabeled
        tasks. Text corpora use the legacy JSONL/CSV/.txt builders; a
        ``text,label`` JSONL corpus for text-classification also gets saved
        Choices annotations.
        """
        if not dataset_id or not self.dataset:
            return []
        ver = self.dataset.get_dataset_version(tenant_id, dataset_id, dataset_version)
        if not ver or not ver.uri:
            return []

        from app.services.ls_dataset import build_ls_tasks, build_ls_text_tasks

        ver_dict = {"uri": ver.uri, "format": ver.format or ""}
        if ver.format in ("", "yolo", "coco", "imagenet"):
            result = build_ls_tasks(self.s3, ver_dict, schema)
            if result["tasks"]:
                if result["warning"]:
                    logger.warning("LS tasks from %s v%s: %s",
                                   dataset_id, dataset_version, result["warning"])
                return result["tasks"]
            if ver.format in ("yolo", "coco", "imagenet"):
                return []
        if ver.format == "llm-sft" and task_type == "text-classification":
            result = build_ls_text_tasks(self.s3, ver_dict)
            return result["tasks"]

        filer = self.s3.s3_uri_to_filer_path(ver.uri)
        if not filer:
            return []
        tasks: list[dict] = []
        # text datasets
        try:
            entries = self.s3.list_entries(f"{filer}/texts")
        except Exception:
            entries = []
        for e in entries:
            if e.is_dir:
                continue
            lower = e.name.lower()
            try:
                raw = self.s3.read_object(f"{filer}/texts/{e.name}")
            except Exception:
                continue
            if lower.endswith(".jsonl"):
                tasks.extend(self._tasks_from_jsonl(raw))
            elif lower.endswith(".json"):
                tasks.extend(self._tasks_from_jsonl(raw))
            elif lower.endswith(".txt"):
                # one file = one document task; split on form feed for multi-doc
                docs = raw.decode("utf-8", errors="ignore").split("\f")
                tasks.extend({"data": {"text": d.strip()}} for d in docs if d.strip())
            elif lower.endswith(".csv"):
                tasks.extend(self._tasks_from_csv(raw))
        return tasks

    @staticmethod
    def _chat_to_hypertext(messages: list) -> str:
        """Render sharegpt-style messages as simple chat hypertext."""
        import html

        parts = []
        for m in messages or []:
            role = str(m.get("from", "") or "").upper()
            content = html.escape(str(m.get("value", "") or ""))
            parts.append(f"<b>[{role}]</b> {content}")
        return "<br/>".join(parts)

    def _tasks_from_jsonl(self, raw: bytes) -> list[dict]:
        """Parse JSONL: accepts {'data': {...}}, {'text': ...} or sharegpt
        {'conversations': [...]}; each line = one task."""
        out = []
        for line in raw.decode("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # plain text line inside a .jsonl (sloppy data) → text task
                out.append({"data": {"text": line}})
                continue
            if isinstance(obj, dict):
                if isinstance(obj.get("data"), dict):
                    out.append({"data": obj["data"]})
                elif obj.get("conversations"):
                    out.append({"data": {
                        "chat": self._chat_to_hypertext(obj["conversations"]),
                        "text": "\n".join(
                            str(m.get("value", "")) for m in obj["conversations"]
                        ),
                    }})
                else:
                    out.append({"data": {"text": str(obj.get("text") or obj.get("content") or json.dumps(obj, ensure_ascii=False))}})
        return out

    @staticmethod
    def _tasks_from_csv(raw: bytes) -> list[dict]:
        """Parse CSV with a header; each row becomes a task (columns → data)."""
        import csv
        import io as _io

        try:
            rows = list(csv.DictReader(_io.StringIO(raw.decode("utf-8", errors="ignore"))))
        except Exception:
            return []
        return [{"data": {k: (v or "") for k, v in r.items()}} for r in rows]

    def reimport_tasks(self, tenant_id: str, host: str, project_id: str) -> int:
        """Re-import tasks into the project's LS project."""
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == project_id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p or not p.external_id:
            return 0
        ls_id = int(p.external_id)
        tasks = self.build_tasks_from_dataset(
            tenant_id, p.dataset_id, p.dataset_version,
            dict(p.schema_ or {}), p.task_type)
        if tasks:
            self.ls.delete_all_tasks(ls_id)
            self.ls.import_tasks(ls_id, tasks)
        return len(tasks)

    def sync_progress(self, tenant_id: str, project_id: str) -> None:
        """Sync project progress from the annotation tool (task/annotation counts)."""
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == project_id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p or not p.external_id:
            return

        # Route by task_type: video-tracking -> CVAT, others -> LS
        use_cvat = (p.task_type == "video-tracking")
        if use_cvat and self.cvat and self.cvat.enabled():
            try:
                cvat_id = int(p.external_id)
                # CVAT does not expose a numeric per-task progress while
                # annotating: a task stays status='annotation' until every job
                # is accepted (stage=acceptance/state=completed), then flips to
                # status='completed'. So completion is derived from task status,
                # not a (usually null) 'progress' field. total_items is the sum
                # of frames across tasks, matching the editor's frame counter.
                data = self.cvat.list_tasks(cvat_id, 1, 100)
                results = data.get("results", []) or []
                task_count = int(data.get("count", 0)) or len(results)
                if task_count > 0:
                    total_frames = sum(int(t.get("size", 0) or 0) for t in results)
                    done_tasks = sum(1 for t in results if t.get("status") == "completed")
                    all_done = done_tasks >= task_count
                    p.total_items = total_frames
                    p.annotated_items = total_frames if all_done else 0
                    p.progress = 100 if all_done else 0
                    p.status = "completed" if all_done else "in_progress"
                    self.db.commit()
            except Exception:
                pass
            return

        # Label Studio (default for non-video tasks).
        ls_id = int(p.external_id)
        try:
            data = self.ls.list_tasks(ls_id, 1, 1)
        except Exception:
            return
        total = int(data.get("total", 0))
        # total_annotations may not be present in list; use project fields if absent.
        annotated = int(data.get("total_annotations", 0) or 0)
        if total > 0:
            p.total_items = total
            p.annotated_items = annotated
            p.progress = min(100, annotated * 100 // total)
            # Mirror the CVAT branch: only completed once every task carries an
            # annotation (predictions accepted / confirmed-no-target); a project
            # at 100% must not stay in_progress forever.
            p.status = "completed" if annotated >= total else "in_progress"
            self.db.commit()

    # -- LS tasks -----------------------------------------------------------

    def ls_tasks(self, tenant_id: str, project_id: str, page: int, page_size: int) -> tuple[int, list[dict], int]:
        """ls_tasks returns studio-ready LS views with annotations and predictions.

        LS fields=all embeds the same result arrays as task detail. Using that list
        mode avoids a serial detail request per task, which otherwise exceeds the
        studio's request deadline on an ordinary 20-task page.
        """
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == project_id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p or not p.external_id:
            return 0, [], 0
        ls_id = int(p.external_id)
        data = self.ls.list_tasks(ls_id, page, page_size, include_results=True)
        tasks = data.get("tasks", []) or []
        total = int(data.get("total", 0))
        views = []
        for t in tasks:
            annotations = [
                {
                    "id": a.get("id"),
                    "result": a.get("result", []),
                    "created_at": a.get("created_at"),
                }
                for a in (t.get("annotations") or [])
            ]
            predictions = [
                {
                    "id": pr.get("id"),
                    "result": pr.get("result", []),
                    "score": pr.get("score"),
                    "model_version": pr.get("model_version"),
                }
                for pr in (t.get("predictions") or [])
            ]
            image = resolve_ls_image(t.get("data", {}).get("image"), ls_id)
            task_data = dict(t.get("data", {}) or {})
            if image:
                task_data["image"] = image
            views.append({
                "id": t.get("id"),
                "inner_id": t.get("inner_id"),
                "data": task_data,
                "image_url": image,
                "is_labeled": t.get("is_labeled", False),
                "annotations": annotations,
                "predictions": predictions,
                "created_at": t.get("created_at"),
                "updated_at": t.get("updated_at"),
            })
        return ls_id, views, total

    # -- CVAT tasks -----------------------------------------------------------

    def cvat_tasks(self, tenant_id: str, project_id: str, page: int, page_size: int) -> tuple[int, list[dict], int]:
        """List tasks from CVAT with annotations resolved, returning (cvat_id, views, total)."""
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == project_id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p or not p.external_id:
            return 0, [], 0
        cvat_id = int(p.external_id)
        data = self.cvat.list_tasks(cvat_id, page, page_size)
        tasks = data.get("results", []) or []
        total = int(data.get("count", 0))
        views = []
        for t in tasks:
            annotations = []
            # Fetch annotations for this task
            try:
                ann = self.cvat.get_annotations(int(t.get("id", 0)))
                for shape in ann.get("shapes") or []:
                    annotations.append({
                        "id": shape.get("id"),
                        "type": shape.get("type"),
                        "label_id": shape.get("label_id"),
                        "frame": shape.get("frame", 0),
                        "points": shape.get("points", []),
                    })
                for track in ann.get("tracks") or []:
                    for shape in track.get("shapes") or []:
                        if not shape.get("outside"):
                            annotations.append({
                                "id": shape.get("id"),
                                "type": shape.get("type"),
                                "label_id": track.get("label_id"),
                                "frame": shape.get("frame", 0),
                                "points": shape.get("points", []),
                            })
            except Exception:
                pass
            views.append({
                "id": t.get("id"),
                "name": t.get("name"),
                "size": t.get("size", 0),
                "mode": t.get("mode", ""),
                "status": t.get("status"),
                "progress": t.get("progress", 0),
                "start_frame": t.get("start_frame", 0),
                "stop_frame": t.get("stop_frame", 0),
                "width": t.get("width", 0),
                "height": t.get("height", 0),
                "updated_date": t.get("updated_date"),
                "data": t.get("data", {}),
                "annotations": annotations,
            })
        return int(p.external_id), views, total

    def cvat_task_detail(self, tenant_id: str, project_id: str, task_id: int) -> dict | None:
        """Get CVAT task details with annotations."""
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == project_id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p or not p.external_id:
            return None
        try:
            task = self.cvat.get_task(task_id)
            annotations = []
            try:
                ann = self.cvat.get_annotations(task_id)
                for shape in ann.get("shapes") or []:
                    annotations.append({
                        "id": shape.get("id"),
                        "type": shape.get("type"),
                        "label_id": shape.get("label_id"),
                        "frame": shape.get("frame", 0),
                        "points": shape.get("points", []),
                        "occluded": shape.get("occluded", False),
                        "outside": shape.get("outside", False),
                    })
                for track in ann.get("tracks") or []:
                    for shape in track.get("shapes") or []:
                        if not shape.get("outside"):
                            annotations.append({
                                "id": shape.get("id"),
                                "type": shape.get("type"),
                                "label_id": track.get("label_id"),
                                "frame": shape.get("frame", 0),
                                "points": shape.get("points", []),
                                "track_id": track.get("id"),
                            })
            except Exception:
                pass
            return {
                "id": task.get("id"),
                "name": task.get("name"),
                "size": task.get("size", 0),
                "width": task.get("width", 0),
                "height": task.get("height", 0),
                "status": task.get("status"),
                "data": task.get("data", {}),
                "annotations": annotations,
            }
        except Exception:
            return None

    def cvat_save_annotation(self, tenant_id: str, project_id: str, task_id: int, result: list[dict]) -> None:
        """Save an annotation back to CVAT."""
        self.cvat.update_annotations(task_id, {"shapes": result, "tracks": []}, import_mode="append", action="create")

    # -- CVAT native editor support ------------------------------------------

    def _cvat_project_row(self, tenant_id: str, project_id: str) -> AnnotationProject:
        """Look up an annotation project row or raise."""
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == project_id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p or not p.external_id:
            raise ValueError("project not linked to cvat")
        return p

    def cvat_task_annotation_detail(self, tenant_id: str, project_id: str, task_id: int) -> dict:
        """Full annotation state of a CVAT task for the NF-native editor.

        Returns task metadata, the project label list, and all tracks with
        their keyframes (frame, points, outside, occluded) so the frontend
        can interpolate exactly like CVAT does.
        """
        p = self._cvat_project_row(tenant_id, project_id)
        task = self.cvat.get_task(task_id)
        label_map = self.cvat.resolve_label_map(int(p.external_id))
        ann = self.cvat.get_annotations(task_id)

        tracks: list[dict[str, Any]] = []
        for tr in ann.get("tracks") or []:
            if not isinstance(tr, dict):
                continue
            keyframes = []
            for s in (tr.get("shapes") or []):
                if not isinstance(s, dict):
                    continue
                all_pts = [float(x) for x in (s.get("points") or [])]
                is_poly = str(s.get("type", "rectangle")) == "polygon"
                kf: dict[str, Any] = {
                    "frame": int(s.get("frame", 0) or 0),
                    "outside": bool(s.get("outside", False)),
                    "occluded": bool(s.get("occluded", False)),
                }
                if is_poly and len(all_pts) >= 6:
                    kf["shape"] = "polygon"
                    kf["poly"] = all_pts
                    # bbox for hit-testing/selection
                    xs, ys = all_pts[0::2], all_pts[1::2]
                    kf["points"] = [min(xs), min(ys), max(xs), max(ys)]
                else:
                    kf["points"] = all_pts[:4]
                keyframes.append(kf)
            if not keyframes:
                continue
            tracks.append({
                "id": int(tr.get("id", 0) or 0),
                "label_id": int(tr.get("label_id", 0) or 0),
                "frame": int(tr.get("frame", keyframes[0]["frame"]) or 0),
                "keyframes": keyframes,
            })

        shapes: list[dict[str, Any]] = []
        for sh in ann.get("shapes") or []:
            if not isinstance(sh, dict):
                continue
            shapes.append({
                "id": int(sh.get("id", 0) or 0),
                "label_id": int(sh.get("label_id", 0) or 0),
                "frame": int(sh.get("frame", 0) or 0),
                "points": [float(x) for x in (sh.get("points") or [])][:4],
                "type": str(sh.get("type", "rectangle")),
            })

        stop_frame = int(task.get("stop_frame", 0) or 0)
        size = int(task.get("size", 0) or 0)
        return {
            "task": {
                "id": int(task.get("id", 0) or 0),
                "name": task.get("name", ""),
                "mode": task.get("mode", ""),
                "width": int(task.get("width", 0) or 0),
                "height": int(task.get("height", 0) or 0),
                "frame_count": (stop_frame + 1) if stop_frame else size,
                "size": size,
            },
            "labels": [
                {"id": lid, "name": name}
                for lid, name in sorted(label_map.items())
                if name
            ],
            "tracks": tracks,
            "shapes": shapes,
        }

    def cvat_replace_tracks(self, tenant_id: str, project_id: str, task_id: int, tracks: list[dict]) -> int:
        """Replace ALL annotations of a CVAT task with the given tracks.

        Editor semantics: the NF editor owns the task's annotations. We
        clear everything and recreate the tracks so keyframe edits, deletes
        and reorderings apply cleanly (no append duplicates).
        """
        self._cvat_project_row(tenant_id, project_id)

        payload_tracks: list[dict[str, Any]] = []
        for tr in tracks or []:
            kfs = sorted(tr.get("keyframes") or [], key=lambda k: int(k.get("frame", 0) or 0))
            kfs = [k for k in kfs if len(k.get("points") or []) >= 4 or len(k.get("poly") or []) >= 6]
            if not kfs:
                continue
            lid = int(tr.get("label_id", 0) or 0)
            if not lid:
                continue
            shapes = []
            for k in kfs:
                if k.get("shape") == "polygon" and len(k.get("poly") or []) >= 6:
                    shapes.append({
                        "frame": int(k["frame"]),
                        "type": "polygon",
                        "points": [float(x) for x in k["poly"]],
                        "occluded": bool(k.get("occluded", False)),
                        "outside": bool(k.get("outside", False)),
                    })
                else:
                    shapes.append({
                        "frame": int(k["frame"]),
                        "type": "rectangle",
                        "points": [float(x) for x in k["points"][:4]],
                        "occluded": bool(k.get("occluded", False)),
                        "outside": bool(k.get("outside", False)),
                    })
            payload_tracks.append({
                "label_id": lid,
                "frame": int(kfs[0]["frame"]),
                "group": 0,
                "source": "manual",
                "shapes": shapes,
            })

        self.cvat.clear_annotations(task_id)
        if payload_tracks:
            self.cvat.update_annotations(
                task_id, {"tracks": payload_tracks, "shapes": []},
                import_mode="append", action="create",
            )
        return len(payload_tracks)

    def cvat_frame(self, tenant_id: str, project_id: str, task_id: int, frame: int, quality: str = "compressed") -> bytes:
        """Proxy a single frame JPEG from CVAT (for the NF-native editor)."""
        self._cvat_project_row(tenant_id, project_id)
        return self.cvat.download_frame(task_id, frame, quality)

    def sam_segment(
        self, tenant_id: str, image_url: str, image_base64: str,
        points: list, negative_points: list, box: list | None, ml_backend_id: str,
        model_name: str = "", model_version: str = "",
    ) -> dict:
        """Run interactive SAM segmentation on the shared ML backend.

        Thin proxy: the ml-backend /sam/segment endpoint holds the model. The
        SAM weights are user-uploaded and user-selected, so the registry
        model_name/version is resolved to an s3 weights URI and passed as the
        backend's model_path (no SAM weights are bundled in the image).
        """
        import httpx

        weights = self._resolve_backend_weights(tenant_id, model_name, model_version)
        if not weights:
            raise FileNotFoundError(
                "the selected SAM model version has no usable s3 weights; upload a "
                ".pt/.onnx SAM checkpoint and select it in the editor"
            )

        ml_url = ""
        if ml_backend_id and self.ml_backend:
            mb = self.ml_backend.get(tenant_id, ml_backend_id)
            if mb and mb.get("url"):
                ml_url = mb["url"]
        if not ml_url:
            ml_url = config.labelstudio.ml_backend_url
        if not ml_url:
            raise ValueError("ml backend url not configured")
        body: dict[str, Any] = {"model_path": weights}
        if image_url:
            body["image_url"] = image_url
        if image_base64:
            body["image_base64"] = image_base64
        if points:
            body["points"] = points
        if negative_points:
            body["negative_points"] = negative_points
        if box:
            body["box"] = box
        resp = httpx.post(f"{ml_url.rstrip('/')}/sam/segment", json=body, timeout=httpx.Timeout(120.0))
        if resp.status_code != 200:
            try:
                detail = resp.json().get("error", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            raise ValueError(f"sam segment failed: {detail}")
        return resp.json()

    def ls_save_annotation(self, tenant_id: str, project_id: str, task_id: int, result: list[dict]) -> None:
        """Save an annotation back to Label Studio (creates a new annotation)."""
        self.ls.save_annotation(task_id, result)

    # -- pre-annotation -----------------------------------------------------

    def _select_preannotate_model(
        self, tenant_id: str, p: AnnotationProject, ml_backend_id: str,
        model_name: str, model_version: str,
    ) -> tuple[str, str]:
        """Resolve which registry model pre-annotation should use.

        Priority: explicit per-request choice > the project's remembered
        preference > the selected ML backend row's bound model. Returns
        ("", "") when nothing is chosen.
        """
        if model_name and model_version:
            return model_name, model_version
        pref = p.preannotate_model or {}
        if pref.get("model_name") and pref.get("model_version"):
            return str(pref["model_name"]), str(pref["model_version"])
        if ml_backend_id and self.ml_backend:
            mb = self.ml_backend.get(tenant_id, ml_backend_id)
            if mb and mb.get("model_name") and mb.get("model_version"):
                return str(mb["model_name"]), str(mb["model_version"])
        return "", ""

    def preannotate(
        self, tenant_id: str, project_id: str, limit: int, ml_backend_id: str,
        task_ids: list[int], created_by: str, frame_stride: int = 5,
        score_threshold: float | None = None,
        model_name: str = "", model_version: str = "",
    ) -> dict:
        """Create a preannotate async task (worker runs the ML backend inference)."""
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == project_id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p:
            raise ValueError("project not found")
        # Resolve and validate the model up front so a missing choice fails
        # fast with an actionable message instead of a backend file error.
        model_name, model_version = self._select_preannotate_model(
            tenant_id, p, ml_backend_id, model_name, model_version
        )
        if not model_name or not model_version:
            raise ValueError(
                "no pre-annotation model selected: choose a model for this project first"
            )
        weights = self._resolve_backend_weights(tenant_id, model_name, model_version)
        if not weights:
            raise ValueError(
                f"selected model {model_name}@{model_version} has no usable weights; "
                "pick a version with a .pt checkpoint"
            )
        # Effective confidence floor: an explicit per-request value wins;
        # otherwise fall back to the selected backend row's configured
        # threshold, then the 0.25 default (env default 0.5 misses small items).
        if score_threshold is None:
            score_threshold = 0.25
            if ml_backend_id and self.ml_backend:
                mb = self.ml_backend.get(tenant_id, ml_backend_id)
                if mb and mb.get("confidence_threshold"):
                    score_threshold = float(mb["confidence_threshold"])
        score_threshold = min(1.0, max(0.01, float(score_threshold)))
        # Block launching a second concurrent pre-annotation for this project:
        # a running/pending/paused preannotate task already owns it. The user
        # must wait for it or stop it (control API) before starting another.
        for a in self.db.scalars(
            select(AsyncTask).where(
                and_(
                    AsyncTask.tenant_id == tenant_id,
                    AsyncTask.type == "preannotate",
                    AsyncTask.status.in_(["pending", "processing", "paused"]),
                )
            )
        ).all():
            if str((a.params or {}).get("project_id", "")) == str(project_id):
                raise ValueError(
                    "a pre-annotation task is already running for this project; "
                    "wait for it to finish or stop it first"
                )
        params = {
            "project_id": project_id,
            "ml_backend_id": ml_backend_id,
            "task_ids": task_ids,
            "frame_stride": max(1, int(frame_stride or 5)),
            "score_threshold": min(1.0, max(0.01, float(score_threshold or 0.25))),
            # Resolved model choice + its s3 weights (worker injects model_path).
            "model_name": model_name,
            "model_version": model_version,
            "model_path": weights,
        }
        task = AsyncTask(
            id=new_id("async"),
            tenant_id=tenant_id,
            type="preannotate",
            status="pending",
            progress=0,
            params=params,
            created_by=created_by or "anonymous",
        )
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return {
            "task_id": task.id,
            "status": task.status,
            "project_id": project_id,
        }

    def run_preannotate_task(
        self, tenant_id: str, task_id: str, params: dict,
        set_progress: Any = None, gate: Any = None,
    ) -> tuple[int, int, int]:
        """Worker entry: run ML backend predict over unlabeled tasks (all pages).

        `gate`, when provided, is polled before each page/task; it blocks while
        the task is paused and raises AsyncTaskCancelled when stopped, so the
        loop honours the user's pause/resume/stop controls.
        """
        project_id = str(params.get("project_id", ""))
        ml_backend_id = str(params.get("ml_backend_id", "") or "")
        raw_ids = params.get("task_ids") or []
        want = {int(x) for x in raw_ids if isinstance(x, (int, float))}

        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == project_id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p or not p.external_id:
            raise ValueError("project not linked to an annotation tool")

        # Route video-tracking projects to the CVAT preannotate flow.
        use_cvat = (p.task_type == "video-tracking")
        if use_cvat and self.cvat and self.cvat.enabled():
            return self.run_preannotate_cvat_task(tenant_id, task_id, params, set_progress, gate)

        ls_id = int(p.external_id)

        ml_url = self.ls.ml_backend_url
        # Weights resolved at enqueue time (explicit > project preference >
        # backend binding) are authoritative; fall back to the backend row for
        # tasks enqueued before the per-project model preference existed.
        backend_model_path = str(params.get("model_path", "") or "")
        if ml_backend_id and self.ml_backend:
            mb = self.ml_backend.get(tenant_id, ml_backend_id)
            if mb and mb.get("url"):
                ml_url = mb["url"]
                if not backend_model_path:
                    backend_model_path = self._resolve_backend_weights(
                        tenant_id, str(mb.get("model_name") or ""),
                        str(mb.get("model_version") or ""),
                    )
        if not ml_url:
            raise ValueError("ml backend url not configured")
        score_threshold = float(params.get("score_threshold", 0.25) or 0.25)
        from app.clients.labelstudio import inject_predict_attrs
        label_config = inject_predict_attrs(
            build_label_config(p.schema_ or {}),
            score_threshold=score_threshold, model_path=backend_model_path,
        )

        processed, predicted, total = 0, 0, 0
        page = 1
        page_size = 50
        while True:
            if gate:
                gate()
            data = self.ls.list_tasks(ls_id, page, page_size)
            tasks = data.get("tasks", []) or []
            if total == 0:
                total = int(data.get("total", 0))
            for t in tasks:
                tid = int(t.get("id"))
                if want and tid not in want:
                    continue
                # Honour pause/stop between tasks (gate blocks while paused and
                # raises AsyncTaskCancelled when stopped).
                if gate:
                    gate()
                # NOTE: do NOT skip tasks just because is_labeled is true. A task
                # can be marked labeled by an earlier (possibly empty) annotation
                # while the user still wants to re-run pre-annotation to refresh
                # the model's boxes. Predictions are machine state and are
                # overwritten on every run; human annotations are left untouched.
                image = resolve_ls_image(t.get("data", {}).get("image"), ls_id)
                if not image:
                    continue
                resolved = self.preannotate_image_url(image, tenant_id)
                if not resolved:
                    continue
                try:
                    # Pass the LS task id so the ML backend can resolve cloud-storage images.
                    results = self.ls.run_ml_backend_predict(
                        ml_url, [{"id": tid, "data": {"image": resolved}}], ls_id, label_config
                    )
                except Exception:
                    continue
                processed += 1
                for r in results:
                    rs = r.get("result", []) or []
                    score = float(r.get("score", 0) or 0)
                    mv = str(r.get("model_version", "") or "ppyoloe")
                    try:
                        # Replace any prior (e.g. stale/empty) prediction instead of
                        # stacking a duplicate each time pre-annotation is re-run.
                        self.ls.replace_prediction(tid, rs, score, mv)
                        predicted += 1
                    except Exception:
                        pass
                if set_progress and total > 0:
                    set_progress(task_id, min(100, 5 + processed * 95 // total), "processing")
            if page * page_size >= total or not tasks:
                break
            page += 1
        return processed, predicted, total

    def preannotate_image_url(self, image: str, tenant_id: str) -> str:
        """Convert an s3:// image to an HTTP URL the ML backend can reach.

        The ML backend always runs *inside the cluster* and fetches images by
        plain HTTP GET (it cannot sign S3 requests; SeaweedFS denies anonymous
        reads). So we hand it a short-lived presigned URL signed against the
        cluster S3 endpoint. In dev the worker runs on the host via a localhost
        port-forward, but signing is offline and the URL host is the cluster
        endpoint, which the in-cluster consumer resolves directly.
        """
        if image.startswith("http://") or image.startswith("https://"):
            return image
        if image.startswith("s3://"):
            rest = image[len("s3://"):]
            bucket, _, key = rest.partition("/")
            return self.s3.presigned_cluster_get_url(bucket or self.s3.s3_bucket(), key)
        return ""

    def _resolve_backend_weights(self, tenant_id: str, model_name: str, model_version: str) -> str:
        """Resolve an ML backend row's model_name/version to an s3:// weights URI.

        Returns "" when the pair does not match a registered model version
        (the ML backend then falls back to its default weights).
        """
        if not model_name or not model_version:
            return ""
        m = self.db.scalar(
            select(ModelRow).where(
                and_(ModelRow.name == model_name, ModelRow.tenant_id == tenant_id)
            )
        )
        if not m:
            return ""
        v = self.db.scalar(
            select(ModelVersionRow).where(
                and_(
                    ModelVersionRow.model_id == m.id,
                    ModelVersionRow.version == model_version,
                    ModelVersionRow.tenant_id == tenant_id,
                )
            )
        )
        if not v:
            return ""
        md = v.metadata_ if isinstance(v.metadata_, dict) else {}
        weights = str(md.get("weights_uri") or v.file_path or "")
        if weights.startswith("s3://"):
            return weights
        return ""

    # -- batch accept -------------------------------------------------------

    def batch_accept(self, tenant_id: str, project_id: str, task_ids: list[int] | None = None, created_by: str = "anonymous") -> dict:
        """Create a batch_accept async task (predictions -> annotations).
        
        If task_ids is empty or None, process all unlabeled tasks with predictions.
        """
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == project_id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p:
            raise ValueError("project not found")
        
        # If task_ids is empty or None, process all unlabeled tasks with predictions
        params = {"project_id": project_id}
        if task_ids:
            params["task_ids"] = task_ids
        else:
            # Empty task_ids means process all unlabeled tasks with predictions
            params["task_ids"] = []
            params["accept_all"] = True
            
        task = AsyncTask(
            id=new_id("async"),
            tenant_id=tenant_id,
            type="batch_accept",
            status="pending",
            progress=0,
            params=params,
            created_by=created_by or "anonymous",
        )
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return {"task_id": task.id, "status": task.status}

    def run_batch_accept_task(
        self, tenant_id: str, task_id: str, params: dict,
        set_progress: Any = None, gate: Any = None,
    ) -> tuple[int, int, int]:
        """Worker entry: convert predictions to annotations; empty -> confirmed no-target.

        `gate` honours pause/resume/stop exactly as in run_preannotate_task.
        """
        project_id = str(params.get("project_id", ""))
        raw_ids = params.get("task_ids") or []
        want = {int(x) for x in raw_ids if isinstance(x, (int, float))} if params.get("task_ids") else set()
        accept_all = params.get("accept_all", False)

        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == project_id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p or not p.external_id:
            raise ValueError("project not linked to label studio")
        ls_id = int(p.external_id)

        processed, accepted, empty = 0, 0, 0
        page = 1
        page_size = 50
        # Exact total for an explicit task selection; accept-all learns the
        # project total from the first list_tasks page.
        total = len(want) if want else 0
        while True:
            if gate:
                gate()
            data = self.ls.list_tasks(ls_id, page, page_size)
            tasks = data.get("tasks", []) or []
            if not tasks:
                break
            if total == 0:
                total = int(data.get("total", 0))
            for t in tasks:
                tid = int(t.get("id"))
                if t.get("is_labeled"):
                    continue
                # If accept_all is True, process all unlabeled tasks; otherwise filter by task_ids
                if not accept_all and want and tid not in want:
                    continue
                # Honour pause/stop between tasks.
                if gate:
                    gate()
                best: list[Any] = []
                has_pred = False
                if int(t.get("total_predictions", 0) or 0) > 0:
                    detail = self.ls.get_task(tid)
                    preds = detail.get("predictions") or []
                    if preds and preds[0].get("result"):
                        best = preds[0]["result"]
                        has_pred = True
                    try:
                        self.ls.save_annotation(tid, best)
                    except Exception:
                        continue
                    processed += 1
                    if has_pred:
                        accepted += 1
                    else:
                        empty += 1
                    if set_progress and total > 0:
                        # Map the work onto 5..100; the worker reserved 0..5
                        # for task dispatch (same scale as pre-annotation).
                        set_progress(task_id, min(100, 5 + processed * 95 // total), "processing")
            if page * page_size >= int(data.get("total", 0)) or not tasks:
                break
            page += 1
        return processed, accepted, empty

    # -- export -------------------------------------------------------------

    def _resolve_source_version(self, tenant_id: str, project: AnnotationProject) -> Optional[DatasetVersion]:
        """Resolve the dataset version an annotated export is built from.

        Prefers the version pinned on the project. When that row no longer
        exists (the source version was deleted or the dataset re-imported
        after the project was created), fall back to the dataset's newest
        surviving version instead of hard-failing the whole export.
        """
        if not project.dataset_id or not self.dataset:
            return None
        pinned = project.dataset_version
        if pinned:
            row = self.dataset.get_dataset_version(tenant_id, project.dataset_id, int(pinned))
            if row:
                return row
            logger.warning(
                "project %s pins dataset %s version %s which no longer exists; "
                "falling back to the latest existing version",
                project.id, project.dataset_id, pinned,
            )
        return self.db.scalar(
            select(DatasetVersion)
            .where(and_(
                DatasetVersion.dataset_id == project.dataset_id,
                DatasetVersion.tenant_id == tenant_id,
            ))
            .order_by(DatasetVersion.version.desc())
        )

    def export_as_dataset(
        self, tenant_id: str, project_id: str, new_name: str, description: str, format: str, created_by: str
    ) -> dict[str, Any]:
        """Export annotations into a new dataset version (COCO JSON + copied images).

        Routes to CVAT or LS based on the project's annotation_tool field.
        Text projects (NLP task types) export as JSONL instead of COCO.
        """
        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == project_id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p or not p.external_id:
            raise ValueError("project not linked to an annotation tool")
        fmt = (format or "coco").lower()

        # Text projects: classification → plain JSONL, generation/chat →
        # sharegpt JSONL (feeds LLaMA-Factory directly, ADR-011).
        text_types = ("text-classification", "ner", "text-generation", "summarization", "chat-rlhf")
        if p.task_type in text_types:
            return self._export_text(tenant_id, p, new_name, description, created_by)
        if fmt not in ("coco", "yolo"):
            fmt = "coco"

        # Route by task_type: video-tracking -> CVAT, others -> LS
        use_cvat = (p.task_type == "video-tracking")
        if use_cvat and self.cvat and self.cvat.enabled():
            return self._export_cvat(tenant_id, p, new_name, description, fmt, created_by)
        # Default: Label Studio.
        if not self.ls.enabled():
            raise ValueError("label studio not configured")
        ls_id = int(p.external_id)

        # Categories from the project schema.
        categories: list[str] = []
        schema = p.schema_ if isinstance(p.schema_, dict) else {}
        raw_labels = schema.get("labels") or []
        for label in raw_labels:
            categories.append(str(label))
        if not categories:
            categories = ["person", "car", "bus", "truck"]
        cat_map = {c: i for i, c in enumerate(categories)}
        # The category map is extended after parsing so labels present in
        # annotations but missing from the schema (renamed labels, classes
        # imported from data.yaml whose names drifted from the project config)
        # are appended instead of silently dropped — dropping negatives/other
        # classes here produced one-class exports and models that hallucinate
        # the surviving class.
        skipped_geometry = 0

        # Pull annotated tasks from Label Studio.
        raw = self.ls.export_annotations(ls_id)
        coco_images: list[dict[str, Any]] = []
        coco_anns: list[dict[str, Any]] = []
        yolo_images: list[dict[str, Any]] = []
        yolo_labels: dict[str, list[str]] = {}
        ann_idx = 1
        default_w, default_h = 640, 480

        def _img_dims(img_ref: str) -> tuple[int, int]:
            """Resolve an image's real pixel size from the S3 object (JPEG SOF)."""
            if not img_ref.startswith("s3://"):
                return default_w, default_h
            try:
                data = self.s3.read_object(self.s3.s3_uri_to_filer_path(img_ref))
                w, h = _jpeg_dims(data)
                if w and h:
                    return w, h
            except Exception:
                pass
            return default_w, default_h

        # Per-image parsed boxes: (label, [px, py, pw, ph]) in real pixels.
        parsed_images: list[dict[str, Any]] = []
        extra_labels: list[str] = []
        for t in raw:
            data = t.get("data") or {}
            img_ref = str(data.get("image") or "")
            img_name = ""
            if img_ref.startswith("s3://"):
                img_name = img_ref.rstrip("/").rsplit("/", 1)[-1]
            if not img_name:
                img_name = f"img_{int(t.get('inner_id') or 0):04d}.jpg"
            iw, ih = _img_dims(img_ref)

            boxes: list[tuple[str, list[float]]] = []
            for a in t.get("annotations") or []:
                for r in a.get("result") or []:
                    if r.get("type") != "rectanglelabels":
                        continue
                    val = r.get("value") or {}
                    x = float(val.get("x") or 0)
                    y = float(val.get("y") or 0)
                    w = float(val.get("width") or 0)
                    h = float(val.get("height") or 0)
                    rl = val.get("rectanglelabels") or []
                    label = str(rl[0]) if rl else ""
                    if not label:
                        continue
                    if w <= 0 or h <= 0:
                        skipped_geometry += 1
                        continue
                    # Label Studio stores box geometry as percentages (0-100).
                    # Convert to real pixels so COCO bbox / IoU matching works.
                    px = x * iw / 100.0
                    py = y * ih / 100.0
                    pw = w * iw / 100.0
                    ph = h * ih / 100.0
                    boxes.append((label, [px, py, pw, ph]))
                    if label not in cat_map and label not in extra_labels:
                        extra_labels.append(label)
            parsed_images.append({
                "file_name": img_name,
                "width": iw,
                "height": ih,
                "boxes": boxes,
            })

        # Append annotation-only labels after schema labels (indices stable).
        for label in extra_labels:
            cat_map[label] = len(categories)
            categories.append(label)
        if extra_labels:
            logger.warning(
                "export project %s: %d label name(s) absent from schema were "
                "appended to keep their boxes: %s",
                project_id, len(extra_labels), extra_labels,
            )
        if skipped_geometry:
            logger.warning(
                "export project %s: %d region(s) skipped due to invalid geometry",
                project_id, skipped_geometry,
            )

        for img in parsed_images:
            img_name, iw, ih = img["file_name"], img["width"], img["height"]
            img_id = len(coco_images) + 1
            coco_images.append({
                "id": img_id,
                "file_name": img_name,
                "width": iw,
                "height": ih,
            })
            yolo_images.append({"file_name": img_name, "width": iw, "height": ih})

            yolo_lines = []
            for label, (x, y, w, h) in img["boxes"]:
                coco_anns.append({
                    "id": ann_idx,
                    "image_id": img_id,
                    "category_id": cat_map[label],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_idx += 1
                cx = (x + w / 2) / iw
                cy = (y + h / 2) / ih
                nw = w / iw
                nh = h / ih
                yolo_lines.append(f"{cat_map[label]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            if yolo_lines:
                base = img_name.rsplit(".", 1)[0] if "." in img_name else img_name
                yolo_labels[f"{base}.txt"] = yolo_lines

        coco = {
            "info": {"description": description or "annotated export", "version": "1.0"},
            "categories": [{"id": i, "name": c} for c, i in cat_map.items()],
            "images": coco_images,
            "annotations": coco_anns,
        }

        # Resolve the source dataset version URI (for images + version bump).
        src_version = self._resolve_source_version(tenant_id, p)
        if not src_version:
            raise ValueError("source dataset version not found")

        base_uri = (src_version.uri or f"s3://nereidforge/datasets/{p.dataset_id}/").rstrip("/")
        export_sub = f"{project_id}-{int(time.time())}"
        export_uri = f"{base_uri}/annotated/{export_sub}/{fmt}/"
        export_filer = self.s3.s3_uri_to_filer_path(export_uri)

        # Write annotations + optional data.yaml, then copy images so the export is standalone.
        files: list[tuple[str, bytes]] = []
        if fmt == "coco":
            files.append(("annotations.json", json.dumps(coco, indent=2).encode()))
            # train.py aborts without data.yaml even for the COCO layout (it
            # reads class names from there and only converts the boxes from
            # annotations.json), so the COCO export carries both.
            if categories:
                files.append(("data.yaml", build_data_yaml(categories).encode()))
        else:
            for name, lines in yolo_labels.items():
                files.append((f"labels/{name}", "\n".join(lines).encode()))
            if categories:
                files.append(("data.yaml", build_data_yaml(categories).encode()))
        for name, content in files:
            self.s3.put_object(f"{export_filer}/{name}", content)

        # Copy source images into the export directory (annotated version is standalone).
        # Source versions may use the flat images/ layout or preset YOLO splits
        # (train|val|test/images); gather from all of them, de-duplicated by name.
        img_count = 0
        src_filer = self.s3.s3_uri_to_filer_path(src_version.uri)
        if src_filer:
            seen_names: set[str] = set()
            for src_img_dir in ("images", "train/images", "val/images", "test/images"):
                for e in self.s3.list_entries(f"{src_filer}/{src_img_dir}"):
                    lower = e.name.lower()
                    if not lower.endswith((".jpg", ".jpeg", ".png")):
                        continue
                    if e.name in seen_names:
                        continue
                    seen_names.add(e.name)
                    try:
                        data = self.s3.read_object(f"{src_filer}/{src_img_dir}/{e.name}")
                        self.s3.put_object(f"{export_filer}/images/{e.name}", data)
                        img_count += 1
                    except Exception:
                        continue

        # Create a new dataset version row pointing at the export directory.
        next_ver = (self.dataset.get_dataset(tenant_id, p.dataset_id) or {}).get("version") or 0
        next_ver += 1
        v = DatasetVersion(
            id=new_id("dsv"),
            dataset_id=p.dataset_id,
            tenant_id=tenant_id,
            version=next_ver,
            description=f"annotated export from project {project_id}",
            uri=export_uri,
            format=fmt,
            size_bytes=sum(len(c) for _, c in files),
            file_count=len(files) + img_count,
            created_by=created_by or "anonymous",
            parent_version_id=src_version.id if src_version else "",
            source_project_id=project_id,
            is_annotated=True,
            is_draft=False,
        )
        self.db.add(v)
        ds = self.db.scalar(
            select(Dataset).where(and_(Dataset.id == p.dataset_id, Dataset.tenant_id == tenant_id))
        )
        if ds:
            ds.version = next_ver
        self.db.commit()

        return {
            "dataset_id": p.dataset_id,
            "dataset_version": next_ver,
            "name": new_name or f"{p.name}-annotated",
            "format": fmt,
            "uri": export_uri,
        }

    # -- CVAT export --------------------------------------------------------

    def _export_text(
        self, tenant_id: str, p: AnnotationProject, new_name: str,
        description: str, created_by: str,
    ) -> dict[str, Any]:
        """Export LS text annotations as JSONL dataset version.

        classification/ner → one JSON per line: {"text":..., "label"/"entities"}
        generation/chat    → sharegpt: {"conversations":[{"from":...,"value":...}]}
        summarization      → {"text":..., "summary":...}
        """
        ls_id = int(p.external_id)
        raw = self.ls.export_annotations(ls_id)

        rows: list[str] = []
        for t in raw:
            data = t.get("data") or {}
            text = str(data.get("text") or "")
            chat_html = str(data.get("chat") or "")
            anns = t.get("annotations") or []
            result = anns[0].get("result", []) if anns else []

            if p.task_type in ("text-generation", "chat-rlhf"):
                # source turns come from the original task data (chat text);
                # the annotation's TextArea = the target response
                conversations: list[dict] = []
                if chat_html:
                    # strip our <b>[ROLE]</b> markup back into turns
                    import re as _re

                    for m in _re.finditer(r"<b>\[([A-Z]+)\]</b>\s*(.*?)(?=(?:<br/>)|$)", chat_html):
                        conversations.append({"from": m.group(1).lower(), "value": m.group(2)})
                if not conversations and text:
                    conversations = [{"from": "human", "value": text}]
                for r in result:
                    val = (r.get("value") or {})
                    out_text = "".join(str(x) for x in (val.get("text") or []))
                    if r.get("type") == "textarea" and out_text:
                        conversations = conversations + [
                            {"from": "gpt", "value": out_text},
                        ]
                        break
                if len(conversations) >= 2:
                    rows.append(json.dumps({"conversations": conversations}, ensure_ascii=False))
            elif p.task_type == "summarization":
                summary = ""
                for r in result:
                    if r.get("type") == "textarea":
                        summary = "".join(str(x) for x in ((r.get("value") or {}).get("text") or []))
                        break
                if text or summary:
                    rows.append(json.dumps({"text": text, "summary": summary}, ensure_ascii=False))
            else:
                # classification + NER
                labels = [
                    str(x)
                    for r in result
                    for x in ((r.get("value") or {}).get("choices")
                              or (r.get("value") or {}).get("labels") or [])
                ]
                entities = [
                    {
                        "start": int((r.get("value") or {}).get("start", 0) or 0),
                        "end": int((r.get("value") or {}).get("end", 0) or 0),
                        "label": str(((r.get("value") or {}).get("labels") or [""])[0]),
                    }
                    for r in result
                    if r.get("type") == "labels"
                ]
                row: dict[str, Any] = {"text": text}
                if entities:
                    row["entities"] = entities
                elif labels:
                    row["label"] = labels[0] if len(labels) == 1 else labels
                if text or labels or entities:
                    rows.append(json.dumps(row, ensure_ascii=False))

        # Resolve source dataset + write version
        src_version = self._resolve_source_version(tenant_id, p)
        if not src_version:
            raise ValueError("source dataset version not found")
        base_uri = (src_version.uri or f"s3://nereidforge/datasets/{p.dataset_id}/").rstrip("/")
        export_uri = f"{base_uri}/annotated/{p.id}-{int(time.time())}/jsonl/"
        export_filer = self.s3.s3_uri_to_filer_path(export_uri)
        content = "\n".join(rows).encode()
        self.s3.put_object(f"{export_filer}/annotations.jsonl", content)

        next_ver = (self.dataset.get_dataset(tenant_id, p.dataset_id) or {}).get("version") or 0
        next_ver += 1
        v = DatasetVersion(
            id=new_id("dsv"),
            dataset_id=p.dataset_id,
            tenant_id=tenant_id,
            version=next_ver,
            description=f"text annotated export from project {p.id}",
            uri=export_uri,
            format="jsonl",
            size_bytes=len(content),
            file_count=len(rows),
            created_by=created_by or "anonymous",
            parent_version_id=src_version.id if src_version else "",
            source_project_id=p.id,
            is_annotated=True,
            is_draft=False,
        )
        self.db.add(v)
        ds = self.db.scalar(
            select(Dataset).where(and_(Dataset.id == p.dataset_id, Dataset.tenant_id == tenant_id))
        )
        if ds:
            ds.version = next_ver
        self.db.commit()
        return {
            "dataset_id": p.dataset_id,
            "dataset_version": next_ver,
            "name": new_name or f"{p.name}-annotated",
            "format": "jsonl",
            "uri": export_uri,
        }

    def _export_cvat(
        self, tenant_id: str, p: AnnotationProject, new_name: str,
        description: str, fmt: str, created_by: str,
    ) -> dict[str, Any]:
        """Export CVAT annotations into a new dataset version (COCO JSON + PP-YOLOE + images).

        Parses CVAT shapes (rectangle, polygon) and track keyframes into COCO
        bbox / segmentation and PP-YOLOE label files. For video tasks, one COCO
        image entry is created per annotated frame and those frames are
        extracted from CVAT so the export is standalone/trainable.
        """
        cvat_id = int(p.external_id)

        # label_id -> name via the robust client helper (handles the case
        # where the project endpoint returns label names as plain strings).
        label_map: dict[int, str] = self.cvat.resolve_label_map(cvat_id)

        # Categories from schema (fallback to CVAT labels).
        categories: list[str] = []
        schema = p.schema_
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except Exception:
                schema = {}
        elif not isinstance(schema, dict):
            schema = {}
        raw_labels = schema.get("labels") or []
        for label in raw_labels:
            categories.append(str(label))
        if not categories:
            categories = [name for name in label_map.values() if name]
        if not categories:
            categories = ["person", "car", "bus", "truck"]
        cat_map = {c: i for i, c in enumerate(categories)}
        # Reverse: cvat label_id -> category index.
        lid_to_cat: dict[int, int] = {}
        for lid, name in label_map.items():
            if name in cat_map:
                lid_to_cat[lid] = cat_map[name]

        coco_images: list[dict[str, Any]] = []
        coco_anns: list[dict[str, Any]] = []
        yolo_labels: dict[str, list[str]] = {}
        ann_idx = 1
        # Frames to extract from CVAT per task (video mode): {task_id: {frame: file_name}}
        extract_frames: dict[int, dict[int, str]] = {}
        # Probe frames already downloaded while resolving dimensions.
        frame_probe_cache: dict[int, dict[int, bytes]] = {}
        any_image_tasks = False

        def add_image(fname: str, w: int, h: int) -> int:
            img_id = len(coco_images) + 1
            coco_images.append({"id": img_id, "file_name": fname, "width": w, "height": h})
            return img_id

        # Iterate CVAT tasks in the project (paginated).
        page = 1
        while True:
            data = self.cvat.list_tasks(cvat_id, page, page_size=100)
            tasks = data.get("results", []) or []
            for t in tasks:
                tid = int(t.get("id", 0))
                twidth = int(t.get("width", 0) or 640)
                theight = int(t.get("height", 0) or 480)
                mode = str(t.get("mode", "annotation") or "annotation")
                frame_count = int(t.get("size", 0) or int(t.get("stop_frame", 0)) + 1)
                # CVAT often leaves task width/height empty; resolved from
                # the first annotated frame JPEG during extraction below.
                real_w = int(t.get("width") or 0)
                real_h = int(t.get("height") or 0)
                has_image_tasks = False

                # Fetch annotations for this task.
                try:
                    ann = self.cvat.get_annotations(tid)
                except Exception as e:
                    logger.warning("cvat get_annotations task=%s failed: %s", tid, e)
                    ann = {}

                # Collect (frame, shape) pairs from standalone shapes and
                # track keyframes (CVAT interpolates between keyframes; the
                # export keeps keyframes only, which is what was annotated).
                frame_shapes: list[tuple[int, dict]] = []
                for shape in ann.get("shapes") or []:
                    if not isinstance(shape, dict):
                        continue
                    frame_shapes.append((int(shape.get("frame", 0) or 0), shape))
                for track in ann.get("tracks") or []:
                    if not isinstance(track, dict):
                        continue
                    tlid = int(track.get("label_id", 0))
                    for ts in track.get("shapes") or []:
                        if not isinstance(ts, dict) or ts.get("outside"):
                            continue
                        frame_shapes.append((int(ts.get("frame", 0) or 0), {**ts, "label_id": tlid}))
                if not frame_shapes:
                    continue

                img_name = t.get("name", f"task_{tid}")
                if mode == "interpolation" and frame_count > 1:
                    # Video task: one image entry per annotated frame; frames
                    # are extracted from CVAT after the parse loop. Resolve
                    # dimensions from the first annotated frame (CVAT often
                    # leaves task width/height empty for video tasks).
                    annotated_frames = sorted({f for f, _ in frame_shapes})
                    probe_cache: dict[int, bytes] = {}
                    if not real_w or not real_h:
                        try:
                            probe = self.cvat.download_frame(
                                tid, annotated_frames[0], quality="compressed"
                            )
                            dims = _jpeg_dims(probe)
                            if dims:
                                real_w, real_h = dims
                            probe_cache[annotated_frames[0]] = probe
                        except Exception as e:
                            logger.warning("cvat probe frame task=%s failed: %s", tid, e)
                    if not real_w or not real_h:
                        real_w, real_h = 640, 480
                    twidth, theight = real_w, real_h
                    frame_to_id: dict[int, int] = {}
                    fmap = extract_frames.setdefault(tid, {})
                    for f in annotated_frames:
                        fn = f"{img_name}_{f:06d}.jpg"
                        fmap[f] = fn
                        frame_to_id[f] = add_image(fn, twidth, theight)
                    for f, shape in frame_shapes:
                        self._parse_cvat_shape(
                            shape, lid_to_cat, twidth, theight,
                            coco_anns, yolo_labels, frame_to_id[f],
                            f"{img_name}_{f:06d}", ann_idx,
                        )
                        ann_idx += 1
                    frame_probe_cache[tid] = probe_cache
                else:
                    # Image task: single image entry named after the task.
                    has_image_tasks = True
                    if not twidth or not theight:
                        twidth, theight = 640, 480
                    img_id = add_image(f"{img_name}.jpg", twidth, theight)
                    for f, shape in frame_shapes:
                        self._parse_cvat_shape(
                            shape, lid_to_cat, twidth, theight,
                            coco_anns, yolo_labels, img_id, img_name, ann_idx,
                        )
                        ann_idx += 1
                if has_image_tasks:
                    any_image_tasks = True

            if page * 100 >= int(data.get("count", 0)) or not tasks:
                break
            page += 1

        coco = {
            "info": {"description": description or "cvat annotated export", "version": "1.0"},
            "categories": [{"id": i, "name": c} for c, i in cat_map.items()],
            "images": coco_images,
            "annotations": coco_anns,
        }

        # Resolve source dataset version for images + version bump.
        src_version = self._resolve_source_version(tenant_id, p)
        if not src_version:
            raise ValueError("source dataset version not found")
        base_uri = (src_version.uri or f"s3://nereidforge/datasets/{p.dataset_id}/").rstrip("/")
        export_sub = f"{p.id}-{int(time.time())}"
        export_uri = f"{base_uri}/annotated/{export_sub}/{fmt}/"
        export_filer = self.s3.s3_uri_to_filer_path(export_uri)

        # Write annotation files.
        files: list[tuple[str, bytes]] = []
        if fmt == "coco":
            files.append(("annotations.json", json.dumps(coco, indent=2).encode()))
        else:
            for name, lines in yolo_labels.items():
                files.append((f"labels/{name}.txt", "\n".join(lines).encode()))
            if categories:
                files.append(("data.yaml", build_data_yaml(categories).encode()))
        for name, content in files:
            self.s3.put_object(f"{export_filer}/{name}", content)

        # Copy source images only for image-mode tasks (video exports get
        # their frames extracted from CVAT; blindly copying the source
        # images/ dir would drag in unrelated files from other flows).
        img_count = 0
        if any_image_tasks:
            src_filer = self.s3.s3_uri_to_filer_path(src_version.uri)
            if src_filer:
                for e in self.s3.list_entries(f"{src_filer}/images"):
                    lower = e.name.lower()
                    if lower.endswith((".jpg", ".jpeg", ".png")):
                        try:
                            data = self.s3.read_object(f"{src_filer}/images/{e.name}")
                            self.s3.put_object(f"{export_filer}/images/{e.name}", data)
                            img_count += 1
                        except Exception:
                            continue

        # Extract annotated video frames from CVAT (compressed quality is
        # plenty for training and much smaller than originals).
        for tid, fmap in extract_frames.items():
            for f, fname in fmap.items():
                cached = frame_probe_cache.get(tid, {}).get(f)
                if cached is not None:
                    jpeg = cached
                else:
                    try:
                        jpeg = self.cvat.download_frame(tid, f, quality="compressed")
                    except Exception as e:
                        logger.warning("cvat frame extract task=%s frame=%s failed: %s", tid, f, e)
                        continue
                try:
                    self.s3.put_object(f"{export_filer}/images/{fname}", jpeg)
                    img_count += 1
                except Exception as e:
                    logger.warning("s3 write frame %s failed: %s", fname, e)

        # Create dataset version row.
        next_ver = (self.dataset.get_dataset(tenant_id, p.dataset_id) or {}).get("version") or 0
        next_ver += 1
        v = DatasetVersion(
            id=new_id("dsv"),
            dataset_id=p.dataset_id,
            tenant_id=tenant_id,
            version=next_ver,
            description=f"cvat annotated export from project {p.id}",
            uri=export_uri,
            format=fmt,
            size_bytes=sum(len(c) for _, c in files),
            file_count=len(files) + img_count,
            created_by=created_by or "anonymous",
            parent_version_id=src_version.id if src_version else "",
            source_project_id=p.id,
            is_annotated=True,
            is_draft=False,
        )
        self.db.add(v)
        ds = self.db.scalar(
            select(Dataset).where(and_(Dataset.id == p.dataset_id, Dataset.tenant_id == tenant_id))
        )
        if ds:
            ds.version = next_ver
        self.db.commit()

        return {
            "dataset_id": p.dataset_id,
            "dataset_version": next_ver,
            "name": new_name or f"{p.name}-annotated",
            "format": fmt,
            "uri": export_uri,
        }

    @staticmethod
    def _parse_cvat_shape(
        shape: dict, lid_to_cat: dict, width: int, height: int,
        coco_anns: list, yolo_labels: dict, img_id: int, yolo_key: str,
        ann_idx: int,
    ) -> None:
        """Parse a single CVAT shape into COCO annotation + YOLO line.

        Handles rectangle (bbox) and polygon (segmentation) types. YOLO lines
        are grouped under yolo_key (image file base name) so each label file
        matches its image file name.
        """
        stype = shape.get("type", "rectangle")
        lid = int(shape.get("label_id", 0))
        cat_idx = lid_to_cat.get(lid)
        if cat_idx is None:
            return
        points = shape.get("points") or []
        if len(points) < 2:
            return

        if stype == "rectangle" and len(points) >= 4:
            # CVAT rectangle: [x1, y1, x2, y2]
            x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
            bw = x2 - x1
            bh = y2 - y1
            coco_anns.append({
                "id": ann_idx,
                "image_id": img_id,
                "category_id": cat_idx,
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            # YOLO normalized.
            cx = (x1 + bw / 2) / width
            cy = (y1 + bh / 2) / height
            nw = bw / width
            nh = bh / height
            yolo_labels.setdefault(yolo_key, []).append(
                f"{cat_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
            )
        elif stype == "polygon" and len(points) >= 6:
            # CVAT polygon: flat [x1, y1, x2, y2, ...] — at least 3 points.
            coco_anns.append({
                "id": ann_idx,
                "image_id": img_id,
                "category_id": cat_idx,
                "segmentation": [points],
                "iscrowd": 0,
            })
            # YOLO polygon: use bounding box of polygon points.
            xs = points[0::2]
            ys = points[1::2]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            bw, bh = x2 - x1, y2 - y1
            cx = (x1 + bw / 2) / width
            cy = (y1 + bh / 2) / height
            nw = bw / width
            nh = bh / height
            yolo_labels.setdefault(yolo_key, []).append(
                f"{cat_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
            )

    # -- CVAT semi-auto annotation ------------------------------------------

    def run_preannotate_cvat_task(
        self, tenant_id: str, task_id: str, params: dict,
        set_progress: Any = None, gate: Any = None,
    ) -> tuple[int, int, int]:
        """Worker entry: run ML backend predict over CVAT video tasks.

        CVAT 2.x has no prediction layer (ADR-010 D2). Frames are sampled
        with a stride, downloaded from CVAT, re-hosted on SeaweedFS (the ML
        backend has no CVAT credentials), predicted, then linked into tracks
        by greedy IoU matching so CVAT interpolates between keyframes.

        ``gate`` (when provided) is polled before each page/task; it blocks
        while paused and raises AsyncTaskCancelled when stopped, mirroring the
        Label Studio preannotate loop.
        """
        project_id = str(params.get("project_id", ""))
        ml_backend_id = str(params.get("ml_backend_id", "") or "")
        frame_stride = max(1, int(params.get("frame_stride", 5) or 5))
        # Per-request detection threshold passed to the ML backend via the
        # label_config; lower than the env default (0.5) because small
        # objects in video routinely score 0.25-0.5.
        score_threshold = float(params.get("score_threshold", 0.25) or 0.25)

        p = self.db.scalar(
            select(AnnotationProject).where(
                and_(AnnotationProject.id == project_id, AnnotationProject.tenant_id == tenant_id)
            )
        )
        if not p or not p.external_id:
            raise ValueError("project not linked to cvat")
        cvat_id = int(p.external_id)

        # Resolve ML backend URL. Weights resolved at enqueue time (explicit >
        # project preference > backend binding) are authoritative.
        ml_url = ""
        backend_model_path = str(params.get("model_path", "") or "")
        if ml_backend_id and self.ml_backend:
            mb = self.ml_backend.get(tenant_id, ml_backend_id)
            if mb and mb.get("url"):
                ml_url = mb["url"]
                # If the backend row references a model registered in NF,
                # run inference with those weights (s3:// auto-download in
                # the ML backend) instead of the pod's default COCO model.
                if not backend_model_path:
                    backend_model_path = self._resolve_backend_weights(
                        tenant_id,
                        str(mb.get("model_name") or ""),
                        str(mb.get("model_version") or ""),
                    )
        if not ml_url:
            ml_url = config.labelstudio.ml_backend_url
        if not ml_url:
            raise ValueError("ml backend url not configured")

        # Build label_config for ML backend (LS-format XML, since the ML
        # backend is a Label Studio ML driver). Score threshold is injected
        # as a control-tag attr the backend reads per request.
        from app.clients.labelstudio import inject_predict_attrs
        label_config = inject_predict_attrs(
            build_label_config(p.schema_ or {}), score_threshold=score_threshold,
            model_path=backend_model_path,
        )

        # CVAT label name -> id map (robust against str labels).
        name_to_lid: dict[str, int] = {
            name: lid for lid, name in self.cvat.resolve_label_map(cvat_id).items() if name
        }

        bucket = self.s3.s3_bucket()
        processed, predicted, total = 0, 0, 0
        page = 1
        while True:
            if gate:
                gate()
            data = self.cvat.list_tasks(cvat_id, page, page_size=50)
            tasks = data.get("results", []) or []
            if total == 0:
                total = int(data.get("count", 0))
            for t in tasks:
                if gate:
                    gate()
                tid = int(t.get("id", 0))
                detail = self.cvat.get_task(tid)
                twidth = int(detail.get("width", 0) or 0)
                theight = int(detail.get("height", 0) or 0)
                stop_frame = int(detail.get("stop_frame", 0) or 0)
                frame_count = int(detail.get("size", 0) or stop_frame + 1 or 1)

                # Sample frames, download, re-host on S3 for the ML backend.
                # Re-runs are supported: previous machine annotations are
                # replaced (see replace_auto_annotations at write time), so we
                # do NOT skip an already-annotated task here.
                frames = list(range(0, max(frame_count, 1), frame_stride))
                if frames and frames[-1] != frame_count - 1:
                    frames.append(frame_count - 1)
                temp_filer = f"/buckets/{bucket}/preannotate/{project_id}/{tid}"
                ml_tasks: list[dict[str, Any]] = []
                for fi in frames:
                    try:
                        jpeg = self.cvat.download_frame(tid, fi, quality="compressed")
                    except Exception as e:
                        logger.warning("cvat frame download task=%s frame=%s failed: %s", tid, fi, e)
                        continue
                    # CVAT often leaves video task dimensions empty; resolve
                    # them from the first successfully downloaded frame.
                    if not twidth or not theight:
                        dims = _jpeg_dims(jpeg)
                        if dims:
                            twidth, theight = dims
                    if not twidth or not theight:
                        twidth, theight = 640, 480
                    frame_uri = f"s3://{bucket}/preannotate/{project_id}/{tid}/frame_{fi:06d}.jpg"
                    self.s3.put_object(f"{temp_filer}/frame_{fi:06d}.jpg", jpeg)
                    resolved = self.preannotate_image_url(frame_uri, tenant_id)
                    if resolved:
                        ml_tasks.append({"id": fi, "data": {"image": resolved}})
                if not ml_tasks:
                    continue

                # Predict in batches: one huge request for hundreds of
                # frames times out (30s) against the ML backend; batches of
                # 8 keep each call snappy and let one bad batch fail alone.
                raw_results: list[dict[str, Any]] = []
                predict_failed = False
                for off in range(0, len(ml_tasks), 8):
                    chunk = ml_tasks[off:off + 8]
                    try:
                        raw_results.extend(self.cvat.run_ml_backend_predict(
                            ml_url, chunk, cvat_id, label_config
                        ))
                    except Exception as e:
                        logger.warning(
                            "cvat preannotate task=%s predict batch failed (frames %s-%s): %s",
                            tid, chunk[0]["id"], chunk[-1]["id"], e,
                        )
                        predict_failed = predict_failed or off == 0
                if not raw_results and predict_failed:
                    self._cleanup_filer_dir(temp_filer)
                    continue
                processed += 1

                # Per-frame detections in absolute pixel coords. The ML
                # backend returns results POSITIONALLY (aligned with the
                # submitted task list, empty result = no detection), so map
                # by index rather than by an id field that does not exist.
                det_by_frame: dict[int, list[tuple[str, list[float]]]] = {}
                for idx, r in enumerate(raw_results):
                    if not (0 <= idx < len(ml_tasks)):
                        continue
                    fi = int(ml_tasks[idx]["id"])
                    for item in r.get("result", []) or []:
                        if item.get("type") != "rectanglelabels":
                            continue
                        val = item.get("value") or {}
                        rls = val.get("rectanglelabels") or []
                        label_name = str(rls[0]) if rls else ""
                        lid = name_to_lid.get(label_name, 0)
                        if not lid:
                            continue
                        x1 = float(val.get("x", 0)) * twidth / 100.0
                        y1 = float(val.get("y", 0)) * theight / 100.0
                        x2 = (float(val.get("x", 0)) + float(val.get("width", 0))) * twidth / 100.0
                        y2 = (float(val.get("y", 0)) + float(val.get("height", 0))) * theight / 100.0
                        if x2 <= x1 or y2 <= y1:
                            continue
                        # label name keyed for later lid lookup in tracks.
                        det_by_frame.setdefault(fi, []).append(
                            (label_name, [x1, y1, x2, y2])
                        )

                tracks = self._link_detections_to_tracks(det_by_frame)
                payload_tracks: list[dict[str, Any]] = []
                leftover_shapes: list[dict[str, Any]] = []
                for label_name, keyframes in tracks:
                    lid = name_to_lid.get(label_name, 0)
                    if not lid:
                        continue
                    start = keyframes[0][0]
                    if len(keyframes) >= 2:
                        shapes = [
                            {
                                "frame": f,
                                "type": "rectangle",
                                "points": pts,
                                "occluded": False,
                                "outside": False,
                            }
                            for f, pts in keyframes
                        ]
                        # Close the track with an outside keyframe: CVAT
                        # extends tracks without one to the END of the task,
                        # so boxes linger on frames where the object is gone.
                        # The next sampled frame after the last hit is the
                        # evidence point ("looked, not there").
                        last_f, last_pts = keyframes[-1]
                        if last_f < frame_count - 1:
                            end_f = min(frame_count - 1, last_f + frame_stride)
                            shapes.append({
                                "frame": end_f,
                                "type": "rectangle",
                                "points": last_pts,
                                "occluded": False,
                                "outside": True,
                            })
                        payload_tracks.append({
                            "label_id": lid,
                            "frame": start,
                            "group": 0,
                            "source": "auto",
                            "shapes": shapes,
                        })
                    else:
                        # Single-frame detections still land as plain shapes
                        # so sparse hits are not silently dropped.
                        leftover_shapes.append({
                            "type": "rectangle",
                            "source": "auto",
                            "label_id": lid,
                            "frame": start,
                            "points": keyframes[0][1],
                            "occluded": False,
                            "outside": False,
                        })

                if payload_tracks or leftover_shapes:
                    try:
                        self.cvat.replace_auto_annotations(
                            tid, payload_tracks, leftover_shapes,
                        )
                        predicted += 1
                    except Exception as e:
                        logger.warning("cvat write annotations task=%s failed: %s", tid, e)

                self._cleanup_filer_dir(temp_filer)

                if set_progress and total > 0:
                    set_progress(task_id, min(100, 5 + processed * 95 // total), "processing")

            if page * 50 >= total or not tasks:
                break
            page += 1

        return processed, predicted, total

    @staticmethod
    def _link_detections_to_tracks(
        det_by_frame: dict[int, list[tuple[str, list[float]]]],
        iou_threshold: float = 0.3,
    ) -> list[tuple[str, list[tuple[int, list[float]]]]]:
        """Greedy IoU linking of per-frame boxes into tracks (same label).

        Returns [(label_name, [(frame, points), ...])] — each track holds its
        keyframes in frame order. CVAT interpolates rectangles between
        keyframes, so a stride-sampled track plays back smoothly.
        """

        class _Track:
            __slots__ = ("last_box", "keyframes")

            def __init__(self, box: list[float], frame: int) -> None:
                self.last_box = box
                self.keyframes: list[tuple[int, list[float]]] = [(frame, box)]

        def iou(a: list[float], b: list[float]) -> float:
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            return inter / (area_a + area_b - inter)

        frames = sorted(det_by_frame.keys())
        active: dict[str, list[_Track]] = {}
        finished: list[tuple[str, list[tuple[int, list[float]]]]] = []

        # Max frames a track survives without a matching detection before it
        # is closed. With stride sampling, the object may be missed on a few
        # consecutive sampled frames; closing too eagerly shatters one real
        # object into many 1-keyframe fragments.
        max_gap = 3

        for f in frames:
            dets = det_by_frame[f]
            used = [False] * len(dets)
            # Extend existing tracks (greedy best IoU per label).
            for label, lst in list(active.items()):
                for track in lst:
                    best_j, best_iou = -1, iou_threshold
                    for j, (lname, box) in enumerate(dets):
                        if used[j] or lname != label:
                            continue
                        v = iou(track.last_box, box)
                        if v > best_iou:
                            best_j, best_iou = j, v
                    if best_j >= 0:
                        used[best_j] = True
                        box = dets[best_j][1]
                        track.keyframes.append((f, box))
                        track.last_box = box
                # Close tracks that missed too many frames.
                still: list[_Track] = []
                for tr in lst:
                    if f - tr.keyframes[-1][0] > max_gap:
                        finished.append((label, tr.keyframes))
                    else:
                        still.append(tr)
                active[label] = still
            # Spawn new tracks for unmatched detections.
            for j, (lname, box) in enumerate(dets):
                if used[j]:
                    continue
                active.setdefault(lname, []).append(_Track(box, f))

        for label, lst in active.items():
            for tr in lst:
                finished.append((label, tr.keyframes))
        return finished

    def _cleanup_filer_dir(self, filer_dir: str) -> None:
        """Best-effort recursive delete of a temp directory on SeaweedFS."""
        try:
            self.s3.delete_object(filer_dir)
        except Exception:
            pass

    def get_async_task(self, tenant_id: str, id: str) -> Optional[dict]:
        """Get an async task by id (shared with dataset service semantics)."""
        t = self.db.scalar(
            select(AsyncTask).where(and_(AsyncTask.id == id, AsyncTask.tenant_id == tenant_id))
        )
        if not t:
            return None
        return {
            "id": t.id,
            "tenant_id": t.tenant_id,
            "type": t.type,
            "status": t.status,
            "progress": t.progress,
            "params": t.params or {},
            "error_message": t.error_message,
            "created_by": t.created_by,
            "created_at": iso(t.created_at),
            "updated_at": iso(t.updated_at),
        }

    def get_active_async_task(self, tenant_id: str, project_id: str) -> Optional[dict]:
        """Return the project's currently queued/running/paused task.

        Covers annotation tasks (preannotate / batch_accept, matched by
        project_id) and large-dataset imports (ls_import, matched by the
        project's LS project id via external_id). Lets the UI restore the
        inline progress bar + controls after a page refresh.
        """
        proj = self.db.scalar(select(AnnotationProject).where(
            and_(AnnotationProject.id == project_id,
                 AnnotationProject.tenant_id == tenant_id)))
        rows = self.db.scalars(
            select(AsyncTask)
            .where(
                and_(
                    AsyncTask.tenant_id == tenant_id,
                    AsyncTask.type.in_(["preannotate", "batch_accept", "ls_import"]),
                    AsyncTask.status.in_(["pending", "processing", "paused"]),
                )
            )
            .order_by(AsyncTask.created_at.desc())
        ).all()
        for r in rows:
            if r.type == "ls_import":
                if proj and str((r.params or {}).get("ls_id", "")) == str(proj.external_id):
                    return self.get_async_task(tenant_id, r.id)
            elif str((r.params or {}).get("project_id", "")) == str(project_id):
                return self.get_async_task(tenant_id, r.id)
        return None

    def control_async_task(self, tenant_id: str, id: str, action: str) -> Optional[dict]:
        """Pause / resume / cancel a running async task (preannotate / batch_accept).

        Writes the signal to params["_control"] (polled by the worker gate);
        cancel also flips status to "cancelled" up-front so the UI stops at once.
        Returns the updated task dict, or None when not found / action unknown.
        """
        if action not in ("pause", "resume", "cancel"):
            raise ValueError(f"unknown control action: {action}")
        t = self.db.scalar(
            select(AsyncTask).where(and_(AsyncTask.id == id, AsyncTask.tenant_id == tenant_id))
        )
        if not t:
            return None
        params = dict(t.params or {})
        params["_control"] = action
        t.params = params
        if action == "cancel":
            t.status = "cancelled"
        elif action == "resume" and t.status == "paused":
            t.status = "processing"
        elif action == "pause" and t.status == "processing":
            t.status = "paused"
        self.db.commit()
        self.db.refresh(t)
        return self.get_async_task(tenant_id, id)
