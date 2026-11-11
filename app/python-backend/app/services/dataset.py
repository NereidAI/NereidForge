# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Dataset service: datasets, versions, drafts, images, async tasks.

Mirrors the Go DatasetService (internal/service/dataset.go):
- dataset CRUD with draft lifecycle (draft -> finalize -> version)
- version management (create/annotated/copy/delete/stats)
- image listing/deletion and S3-backed download/export
- async task creation and progress tracking
"""
from __future__ import annotations

import hashlib
import io
import os
import re
from typing import Any, Optional

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.clients.kfp import KFPClient
from app.clients.s3 import SeaweedFSClient
from app.core.util import iso, new_id, page_bounds, paginate
from app.db.session import SessionLocal
from app.models.orm import AsyncTask, Dataset, DatasetVersion

# Dataset formats accepted by the registry. `llm-sft` marks a sharegpt-style
# JSONL corpus (optionally with an images/ dir for VLM) that is directly
# trainable by the LLM/VLM trainers; coco/yolo/imagenet are computer-vision
# layouts; `jsonl`/`auto` are user-facing aliases normalized on input.
DATASET_FORMATS = {"coco", "yolo", "llm-sft", "jsonl", "imagenet"}
# Formats whose finalized versions are directly selectable for training.
TRAINABLE_FORMATS = {"llm-sft", "imagenet"}

# Task types implied directly by the dataset format (no annotation project
# needed to infer them).
_FORMAT_TASK_TYPES: dict[str, tuple[str, ...]] = {
    "llm-sft": ("llm", "vlm"),
    "jsonl": ("llm", "vlm"),
    "imagenet": ("classification",),
}

# Annotation-project task types mapped onto training task types. Used to infer
# what a coco/yolo corpus was annotated for (bbox vs mask vs labels).
_PROJECT_TASK_MAP: dict[str, str] = {
    "detection": "detection",
    "image-detection": "detection",
    "segmentation": "segmentation",
    "image-segmentation": "segmentation",
    "classification": "classification",
    "image-classification": "classification",
    "keypoint": "detection",
    "rotation": "detection",
    "mask": "segmentation",
    "video-tracking": "detection",
}


def normalize_format(fmt: str) -> str:
    """Normalize a user-supplied format string; '' keeps the caller default."""
    f = (fmt or "").strip().lower()
    if f in ("", "auto"):
        return ""
    if f == "jsonl":
        return "llm-sft"
    if f not in DATASET_FORMATS:
        raise ValueError(f"unsupported dataset format: {fmt}")
    return f


class AsyncTaskCancelled(Exception):
    """Raised inside a worker task loop when the task was stopped (cancelled)
    by the user via the control API, so the worker aborts without failing."""
    pass


def _dataset_dict(d: Dataset, version_size_bytes: int = 0, task_types: Optional[list[str]] = None) -> dict[str, Any]:
    """Serialize a Dataset to the API response shape.

    The dataset table's size_bytes column is not maintained; report the
    aggregate byte size of all of its versions instead. ``task_types`` lists
    the training task types this corpus can be used for (inferred from its
    format and linked annotation projects); an empty list means a generic
    CV corpus (coco/yolo) with no annotation-derived type.
    """
    return {
        "id": d.id,
        "tenant_id": d.tenant_id,
        "name": d.name,
        "description": d.description,
        "uri": d.uri,
        "format": d.format,
        "task_types": task_types or [],
        "version": d.version,
        "size_bytes": version_size_bytes,
        "file_count": d.file_count,
        "status": d.status,
        "created_by": d.created_by,
        "created_at": iso(d.created_at),
        "updated_at": iso(d.updated_at),
    }


def _version_dict(v: DatasetVersion) -> dict[str, Any]:
    """Serialize a DatasetVersion to the API response shape."""
    return {
        "id": v.id,
        "dataset_id": v.dataset_id,
        "tenant_id": v.tenant_id,
        "version": v.version,
        "description": v.description,
        "uri": v.uri,
        "format": v.format,
        "size_bytes": v.size_bytes,
        "file_count": v.file_count,
        "commit_hash": v.commit_hash,
        "created_by": v.created_by,
        "created_at": iso(v.created_at),
        "parent_version_id": v.parent_version_id,
        "source_project_id": v.source_project_id,
        "is_annotated": v.is_annotated,
        "is_draft": v.is_draft,
    }


def _async_task_dict(t: AsyncTask) -> dict[str, Any]:
    """Serialize an AsyncTask to the API response shape."""
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
        "started_at": iso(t.started_at),
        "completed_at": iso(t.completed_at),
    }


def _uri_for(dataset_id: str, version: int, tenant_id: str) -> str:
    """Build the s3:// URI for a dataset version."""
    from app.clients.s3 import SeaweedFSClient

    return SeaweedFSClient.tenant_uri(
        f"s3://nereidforge/datasets/{dataset_id}/v{version}/", tenant_id
    )


class DatasetService:
    """Manages datasets, versions, images, and async import tasks."""

    def __init__(self, db: Session, s3: SeaweedFSClient, kfp: Optional[KFPClient] = None) -> None:
        self.db = db
        self.s3 = s3
        self.kfp = kfp

    # -- datasets -----------------------------------------------------------

    def create_dataset(
        self, tenant_id: str, name: str, description: str, uri: str, format: str, created_by: str
    ) -> dict[str, Any]:
        """Create a dataset (empty shell; versions are created separately)."""
        # auto/unknown -> empty; content detection fills it on first finalize.
        fmt = normalize_format(format)
        ds = Dataset(
            id=new_id("ds"),
            tenant_id=tenant_id,
            name=name,
            description=description or "",
            uri=uri or f"s3://nereidforge/datasets/{name}/",
            format=fmt,
            version=0,
            created_by=created_by or "anonymous",
        )
        self.db.add(ds)
        self.db.commit()
        self.db.refresh(ds)
        return _dataset_dict(ds, task_types=self._dataset_task_types(tenant_id, [ds]).get(ds.id, []))

    def _version_sizes(self, tenant_id: str, dataset_ids: list[str]) -> dict[str, int]:
        """Aggregate byte size of all versions per dataset id."""
        from sqlalchemy import func

        if not dataset_ids:
            return {}
        rows = self.db.execute(
            select(DatasetVersion.dataset_id, func.coalesce(func.sum(DatasetVersion.size_bytes), 0))
            .where(
                and_(
                    DatasetVersion.tenant_id == tenant_id,
                    DatasetVersion.dataset_id.in_(dataset_ids),
                )
            )
            .group_by(DatasetVersion.dataset_id)
        ).all()
        return {r[0]: int(r[1] or 0) for r in rows}

    def _dataset_task_types(self, tenant_id: str, datasets: list[Dataset]) -> dict[str, list[str]]:
        """Infer compatible training task types per dataset.

        Format-based types (llm-sft -> llm/vlm, imagenet -> classification)
        always apply; coco/yolo corpora additionally inherit the task types of
        the annotation projects their versions were exported from. A coco/yolo
        corpus with no linked project yields an empty list (generic CV: the
        caller decides whether detection/segmentation may use it).
        """
        from app.models.orm import AnnotationProject

        out: dict[str, list[str]] = {}
        ids = [d.id for d in datasets]
        project_rows: dict[str, set[str]] = {}
        if ids:
            rows = self.db.execute(
                select(DatasetVersion.dataset_id, AnnotationProject.task_type)
                .select_from(DatasetVersion)
                .outerjoin(
                    AnnotationProject,
                    and_(
                        AnnotationProject.id == DatasetVersion.source_project_id,
                        AnnotationProject.tenant_id == tenant_id,
                    ),
                )
                .where(
                    and_(
                        DatasetVersion.tenant_id == tenant_id,
                        DatasetVersion.dataset_id.in_(ids),
                    )
                )
            ).all()
            for ds_id, project_tt in rows:
                mapped = _PROJECT_TASK_MAP.get(project_tt or "")
                if mapped:
                    project_rows.setdefault(ds_id, set()).add(mapped)
        for d in datasets:
            types = list(_FORMAT_TASK_TYPES.get(d.format or "", ()))
            if d.format in ("coco", "yolo"):
                types.extend(sorted(project_rows.get(d.id, set())))
            out[d.id] = types
        return out

    def list_datasets(
        self, tenant_id: str, page: int, page_size: int, search: str = "", status: str = ""
    ) -> tuple[list[dict], int]:
        """List datasets with optional search/status filters and pagination."""
        from sqlalchemy import func

        query = select(Dataset).where(Dataset.tenant_id == tenant_id)
        count_q = select(func.count(Dataset.id)).where(Dataset.tenant_id == tenant_id)
        if search:
            query = query.where(Dataset.name.ilike(f"%{search}%"))
            count_q = count_q.where(Dataset.name.ilike(f"%{search}%"))
        if status:
            query = query.where(Dataset.status == status)
            count_q = count_q.where(Dataset.status == status)
        total = self.db.scalar(count_q) or 0
        limit, offset = page_bounds(page, page_size)
        rows = self.db.scalars(query.order_by(Dataset.created_at.desc()).limit(limit).offset(offset)).all()
        sizes = self._version_sizes(tenant_id, [d.id for d in rows])
        task_types = self._dataset_task_types(tenant_id, rows)
        return [
            _dataset_dict(d, sizes.get(d.id, 0), task_types.get(d.id, [])) for d in rows
        ], total

    def get_dataset(self, tenant_id: str, id: str) -> Optional[dict]:
        """Get a dataset by id."""
        d = self.db.scalar(
            select(Dataset).where(and_(Dataset.id == id, Dataset.tenant_id == tenant_id))
        )
        if not d:
            return None
        sizes = self._version_sizes(tenant_id, [d.id])
        task_types = self._dataset_task_types(tenant_id, [d])
        return _dataset_dict(d, sizes.get(d.id, 0), task_types.get(d.id, []))

    def update_dataset(self, tenant_id: str, id: str, patch: dict[str, Any]) -> Optional[dict]:
        """Update dataset fields."""
        d = self.db.scalar(
            select(Dataset).where(and_(Dataset.id == id, Dataset.tenant_id == tenant_id))
        )
        if not d:
            return None
        for k, v in patch.items():
            if hasattr(d, k):
                setattr(d, k, v)
        self.db.commit()
        self.db.refresh(d)
        return _dataset_dict(d, task_types=self._dataset_task_types(tenant_id, [d]).get(d.id, []))

    def delete_dataset(self, tenant_id: str, id: str) -> bool:
        """Delete a dataset if not referenced by live annotation projects."""
        refs = self.dataset_references(tenant_id, id)
        if refs:
            raise ValueError("dataset is referenced by active annotation projects")
        d = self.db.scalar(
            select(Dataset).where(and_(Dataset.id == id, Dataset.tenant_id == tenant_id))
        )
        if not d:
            return False
        # cascade delete versions
        self.db.query(DatasetVersion).filter(
            and_(DatasetVersion.dataset_id == id, DatasetVersion.tenant_id == tenant_id)
        ).delete()
        self.db.delete(d)
        self.db.commit()
        return True

    # -- versions -----------------------------------------------------------

    def create_draft(self, tenant_id: str, dataset_id: str, description: str, created_by: str) -> Optional[dict]:
        """Create a draft version (is_draft=True), reusing an existing draft if present."""
        existing = self.db.scalar(
            select(DatasetVersion).where(
                and_(
                    DatasetVersion.dataset_id == dataset_id,
                    DatasetVersion.tenant_id == tenant_id,
                    DatasetVersion.is_draft.is_(True),
                )
            )
        )
        if existing:
            return _version_dict(existing)
        ds = self.db.scalar(
            select(Dataset).where(and_(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id))
        )
        if not ds:
            return None
        next_ver = (ds.version or 0) + 1
        uri = _uri_for(dataset_id, next_ver, tenant_id)
        v = DatasetVersion(
            id=new_id("dsv"),
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            version=0,  # draft carries version 0 until finalized
            description=description or "draft",
            uri=uri,
            # Start undecided; finalize_draft() content detection assigns the
            # real format from the actual objects (jsonl/yolo/coco/imagenet).
            format=ds.format or "",
            commit_hash=self._hash(f"{dataset_id}:draft:{created_by}"),
            created_by=created_by,
            is_draft=True,
        )
        self.db.add(v)
        self.db.commit()
        self.db.refresh(v)
        return _version_dict(v)

    def finalize_draft(self, tenant_id: str, dataset_id: str, draft_id: str) -> Optional[dict]:
        """Finalize a draft into a read-only version, assigning the next version number.

        Format/trainability are derived from the draft's actual objects: a draft
        containing any .jsonl file becomes an `llm-sft` trainable version (covers
        sharegpt LLM and image+JSONL VLM corpora uploaded straight to S3).
        """
        v = self.db.get(DatasetVersion, draft_id)
        if not v or v.tenant_id != tenant_id or not v.is_draft:
            return None
        ds = self.db.scalar(
            select(Dataset).where(and_(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id))
        )
        if not ds:
            return None
        self._apply_content_detection(v)
        next_ver = (ds.version or 0) + 1
        v.version = next_ver
        v.is_draft = False
        if v.format:
            ds.format = v.format
        ds.version = next_ver
        self.db.commit()
        self.db.refresh(v)
        return _version_dict(v)

    def _apply_content_detection(self, v: DatasetVersion) -> None:
        """Inspect a version's S3 objects and set format/is_annotated in place.

        Detection rules, highest priority first: any .jsonl -> llm-sft; any
        labels/*.txt -> yolo; an annotations.json at the root or under a coco/
        folder -> coco. Trainable corpora are marked is_annotated so they show
        up in the training form. A plain image corpus keeps its prior state.
        """
        try:
            filer = self.s3.s3_uri_to_filer_path(v.uri or "")
            files = self.s3.list_files_with_size(filer) if filer else []
        except Exception:
            files = []
        if files:
            v.file_count = len(files)
            v.size_bytes = sum(size for _, size in files)
        names = [name.lower() for name, _ in files]
        has_jsonl = any(name.endswith(".jsonl") for name in names)
        if has_jsonl:
            v.format = "llm-sft"
            # SFT versions are directly trainable; the training UI selects on
            # is_annotated, which doubles as "finalized and trainable".
            v.is_annotated = True
            return
        has_yolo_labels = any(
            name == "labels/" or bool(re.search(r"(^|/)labels/.+\.txt$", name))
            for name in names
        )
        if has_yolo_labels:
            v.format = "yolo"
            v.is_annotated = True
            return
        has_coco_json = any(
            name == "annotations.json"
            or bool(re.search(r"(^|/)coco/annotations\.json$", name))
            for name in names
        )
        if has_coco_json:
            v.format = "coco"
            v.is_annotated = True
            return
        # ImageFolder classification: <split>/<class>/<file> (canonical
        # importer output) or flat images/<class>/<file> trees.
        img_exts = r"(?:jpe?g|png|bmp|webp)"
        has_class_folders = any(
            re.search(rf"(^|/)(train|val|valid|test)/[^/]+/[^/]+\.{img_exts}$", name)
            or re.search(rf"(^|/)images/[^/]+/[^/]+\.{img_exts}$", name)
            for name in names
        )
        if has_class_folders:
            v.format = "imagenet"
            v.is_annotated = True

    def register_s3_version(
        self,
        tenant_id: str,
        dataset_id: str,
        uri: str,
        fmt: str,
        description: str,
        created_by: str,
    ) -> dict[str, Any]:
        """Register an existing S3 prefix as a finalized trainable version.

        No bytes are copied: the version row points at the external prefix
        (e.g. a pre-staged sharegpt corpus). Only the configured bucket is
        accepted, and the prefix must contain at least one object. Format
        auto-detection runs when `fmt` is empty/``auto`` (JSONL → llm-sft).
        """
        ds = self.db.scalar(
            select(Dataset).where(and_(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id))
        )
        if not ds:
            raise ValueError("dataset not found")
        raw = (uri or "").strip().rstrip("/") + "/"
        if not raw.startswith("s3://"):
            raise ValueError("uri must be an s3:// URI")
        bucket = raw[len("s3://"):].split("/", 1)[0]
        if not bucket or bucket != self.s3.s3_bucket():
            raise ValueError(f"uri bucket must be {self.s3.s3_bucket()}")
        # Non-default tenants may only register prefixes inside their own
        # tenant namespace; the default tenant owns the bucket root.
        if tenant_id != "default" and not raw.startswith(f"s3://{bucket}/{tenant_id}/"):
            raise ValueError(f"uri must stay under s3://{bucket}/{tenant_id}/")
        filer = self.s3.s3_uri_to_filer_path(raw)
        try:
            files = self.s3.list_files_with_size(filer)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"cannot list S3 prefix: {e}") from e
        if not files:
            raise ValueError("S3 prefix is empty or does not exist")
        wanted = normalize_format(fmt)
        next_ver = (ds.version or 0) + 1
        v = DatasetVersion(
            id=new_id("dsv"),
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            version=next_ver,
            description=description or f"registered {raw}",
            uri=raw,
            format=wanted or ds.format or "",
            file_count=len(files),
            size_bytes=sum(size for _, size in files),
            commit_hash=self._hash(f"{dataset_id}:register:{raw}:{created_by}"),
            created_by=created_by,
            is_draft=False,
        )
        self._apply_content_detection(v)
        # Explicitly requested llm-sft wins even if detection disagrees.
        if wanted:
            v.format = wanted
            if wanted in TRAINABLE_FORMATS:
                v.is_annotated = True
        if v.format:
            ds.format = v.format
        ds.version = next_ver
        self.db.add(v)
        self.db.commit()
        self.db.refresh(v)
        return _version_dict(v)

    def copy_version_to_draft(self, tenant_id: str, dataset_id: str, version_id: str, created_by: str) -> Optional[dict]:
        """Copy a finalized version's data into a new draft."""
        src = self.db.get(DatasetVersion, version_id)
        if not src or src.tenant_id != tenant_id:
            return None
        ds = self.db.scalar(
            select(Dataset).where(and_(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id))
        )
        if not ds:
            return None
        next_ver = (ds.version or 0) + 1
        uri = _uri_for(dataset_id, next_ver, tenant_id)
        # copy objects from source URI to new draft URI
        src_filer = self.s3.s3_uri_to_filer_path(src.uri)
        dst_filer = self.s3.s3_uri_to_filer_path(uri)
        if src_filer and dst_filer:
            for f in self.s3.walk(src_filer):
                rel = f[len(src_filer):].lstrip("/")
                if rel:
                    data = self.s3.read_object(f)
                    self.s3.put_object(f"{dst_filer}/{rel}", data)
        draft = DatasetVersion(
            id=new_id("dsv"),
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            version=0,
            description=f"copy of v{src.version}",
            uri=uri,
            format=src.format or "yolo",
            file_count=src.file_count,
            size_bytes=src.size_bytes,
            commit_hash=self._hash(f"{dataset_id}:copy:{version_id}"),
            created_by=created_by,
            is_draft=True,
        )
        self.db.add(draft)
        self.db.commit()
        self.db.refresh(draft)
        return _version_dict(draft)

    def list_dataset_versions(self, tenant_id: str, dataset_id: str, annotated_only: bool = False) -> list[dict]:
        """List versions of a dataset, optionally only annotated ones.

        Each serialized version carries has_videos (S3 probe of the
        videos/ prefix) so the frontend can steer video-tracking project
        creation to versions that actually contain videos.
        """
        query = select(DatasetVersion).where(
            and_(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.tenant_id == tenant_id,
            )
        )
        if annotated_only:
            query = query.where(DatasetVersion.is_annotated.is_(True))
        rows = self.db.scalars(query.order_by(DatasetVersion.version.desc())).all()
        out = []
        for v in rows:
            d = _version_dict(v)
            d["has_videos"] = self._version_has_videos(v)
            out.append(d)
        return out

    def list_dataset_versions_paged(
        self, tenant_id: str, dataset_id: str, page: int = 1, page_size: int = 20, annotated_only: bool = False
    ) -> tuple[list[dict], int]:
        """Paginated version list for the dataset detail page."""
        from sqlalchemy import func

        query = select(DatasetVersion).where(
            and_(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.tenant_id == tenant_id,
            )
        )
        count_q = select(func.count(DatasetVersion.id)).where(
            and_(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.tenant_id == tenant_id,
            )
        )
        if annotated_only:
            query = query.where(DatasetVersion.is_annotated.is_(True))
            count_q = count_q.where(DatasetVersion.is_annotated.is_(True))
        total = self.db.scalar(count_q) or 0
        limit, offset = page_bounds(page, page_size)
        rows = self.db.scalars(
            query.order_by(DatasetVersion.version.desc()).limit(limit).offset(offset)
        ).all()
        out = []
        for v in rows:
            d = _version_dict(v)
            d["has_videos"] = self._version_has_videos(v)
            out.append(d)
        return out, total

    def _version_has_videos(self, v: DatasetVersion) -> bool:
        """Cheap check whether a version's videos/ dir contains any file."""
        filer = self.s3.s3_uri_to_filer_path(v.uri or "")
        if not filer:
            return False
        try:
            return any(
                not e.is_dir for e in self.s3.list_entries(f"{filer}/videos")
            )
        except Exception:
            return False

    def get_dataset_version(self, tenant_id: str, dataset_id: str, version: int) -> Optional[DatasetVersion]:
        """Fetch a DatasetVersion row by dataset+version."""
        return self.db.scalar(
            select(DatasetVersion).where(
                and_(
                    DatasetVersion.dataset_id == dataset_id,
                    DatasetVersion.tenant_id == tenant_id,
                    DatasetVersion.version == version,
                )
            )
        )

    def get_version_dict(self, tenant_id: str, dataset_id: str, version: int) -> Optional[dict]:
        """Get a version as an API dict."""
        v = self.get_dataset_version(tenant_id, dataset_id, version)
        return _version_dict(v) if v else None

    def update_dataset_version(self, tenant_id: str, dataset_id: str, version: int, description: str) -> Optional[dict]:
        """Update a version's description."""
        v = self.get_dataset_version(tenant_id, dataset_id, version)
        if not v:
            return None
        v.description = description
        self.db.commit()
        self.db.refresh(v)
        return _version_dict(v)

    def delete_dataset_version(self, tenant_id: str, dataset_id: str, version: int) -> bool:
        """Delete a version and its S3 objects."""
        v = self.get_dataset_version(tenant_id, dataset_id, version)
        if not v:
            return False
        if v.is_draft:
            return self.delete_draft(tenant_id, dataset_id, v.id)
        filer = self.s3.s3_uri_to_filer_path(v.uri)
        if filer:
            self.s3.delete_object(filer)
        self.db.delete(v)
        self.db.commit()
        return True

    def delete_draft(self, tenant_id: str, dataset_id: str, draft_id: str) -> bool:
        """Delete a draft version and its S3 objects."""
        v = self.db.get(DatasetVersion, draft_id)
        if not v or v.tenant_id != tenant_id or not v.is_draft:
            return False
        filer = self.s3.s3_uri_to_filer_path(v.uri)
        if filer:
            self.s3.delete_object(filer)
        self.db.delete(v)
        self.db.commit()
        return True

    # -- images -------------------------------------------------------------

    def list_version_images(
        self, tenant_id: str, dataset_id: str, version: int, page: int = 1, page_size: int = 48
    ) -> tuple[list[str], int]:
        """List image/video file names under a version's images/ and videos/ directories.

        Returns a (page, total) tuple so preview grids can paginate instead of
        loading the full image set (datasets can have thousands of images).
        """
        v = self.get_dataset_version(tenant_id, dataset_id, version)
        if not v or not v.uri:
            return [], 0
        filer = self.s3.s3_uri_to_filer_path(v.uri)
        if not filer:
            return [], 0
        results: list[str] = []
        for sub in ("images", "videos"):
            sub_dir = f"{filer}/{sub}"
            entries = self.s3.list_entries(sub_dir)
            for e in entries:
                if not e.is_dir:
                    results.append(f"{sub}/{e.name}")
        results.sort()
        total = len(results)
        page = max(1, page)
        page_size = max(1, min(page_size, 500))
        start = (page - 1) * page_size
        return results[start:start + page_size], total

    # File extensions that are safe to preview as text in the browser.
    _TEXT_EXTS = {
        ".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".tsv",
        ".cfg", ".ini", ".conf", ".py", ".sh", ".log", ".names",
    }

    def _version_filer_base(self, tenant_id: str, dataset_id: str, version: int) -> Optional[tuple[str, Any]]:
        """Resolve a version row and its S3 filer base path, or (None, None)."""
        v = self.get_dataset_version(tenant_id, dataset_id, version)
        if not v or not v.uri:
            return None, None
        filer = self.s3.s3_uri_to_filer_path(v.uri)
        if not filer:
            return None, None
        return filer, v

    def list_version_files(self, tenant_id: str, dataset_id: str, version: int) -> Optional[list[dict[str, Any]]]:
        """List every object under a version prefix for the file browser.

        Returns a flat list of {path, name, size, ext, is_text, is_media}
        (path is relative to the version root). Returns None if the version
        does not exist. A single paginated LIST supplies sizes (no HEADs).
        """
        filer, _ = self._version_filer_base(tenant_id, dataset_id, version)
        if filer is None:
            return None
        files: list[dict[str, Any]] = []
        for rel, size in self.s3.list_files_with_size(filer):
            ext = os.path.splitext(rel)[1].lower()
            files.append({
                "path": rel,
                "name": os.path.basename(rel),
                "size": size,
                "ext": ext,
                "is_text": ext in self._TEXT_EXTS,
                "is_media": ext in {".jpg", ".jpeg", ".png", ".gif", ".webp",
                                    ".bmp", ".tif", ".tiff",
                                    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv"},
            })
        return files

    @staticmethod
    def _parse_card(text: str) -> dict[str, Any]:
        """Split a README.md into {raw, readme, frontmatter}.

        frontmatter comes from an optional leading YAML block delimited by
        ``---`` lines (HuggingFace-style); readme is the markdown body.
        """
        frontmatter: dict[str, Any] = {}
        body = text
        if text.startswith("---"):
            lines = text.splitlines()
            end = None
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    end = i
                    break
            if end is not None:
                yaml_block = "\n".join(lines[1:end])
                try:
                    import yaml
                    parsed = yaml.safe_load(yaml_block)
                    if isinstance(parsed, dict):
                        frontmatter = parsed
                except Exception:
                    frontmatter = {}
                body = "\n".join(lines[end + 1:]).lstrip("\n")
        return {"raw": text, "readme": body, "frontmatter": frontmatter}

    def get_version_card(self, tenant_id: str, dataset_id: str, version: int) -> Optional[dict[str, Any]]:
        """Read the version's README.md dataset card (HuggingFace-style).

        Returns {raw, readme, frontmatter}; an empty card when no README.md
        exists. Returns None only when the version itself is missing.
        """
        filer, _ = self._version_filer_base(tenant_id, dataset_id, version)
        if filer is None:
            return None
        try:
            raw = self.s3.read_object(f"{filer}/README.md")
        except Exception:
            return {"raw": "", "readme": "", "frontmatter": {}}
        return self._parse_card(raw.decode("utf-8", errors="replace"))

    def save_version_card(self, tenant_id: str, dataset_id: str, version: int, content: str) -> Optional[dict[str, Any]]:
        """Write the version's README.md dataset card and return the parsed card.

        The caller sends the full markdown (optionally including a ``---`` YAML
        frontmatter block); we store it verbatim at the version root. Returns
        None when the version is missing.
        """
        filer, _ = self._version_filer_base(tenant_id, dataset_id, version)
        if filer is None:
            return None
        text = content or ""
        self.s3.put_object(f"{filer}/README.md", text.encode("utf-8"))
        return self._parse_card(text)

    def read_version_file(self, tenant_id: str, dataset_id: str, version: int, path: str) -> Optional[dict[str, Any]]:
        """Read a single text file from a version (preview pane content).

        Returns {path, content} for text-ish files, None if missing / binary.
        """
        filer, _ = self._version_filer_base(tenant_id, dataset_id, version)
        if filer is None:
            return None
        ext = os.path.splitext(path)[1].lower()
        if ext not in self._TEXT_EXTS:
            return None
        # Path traversal protection: never escape the version prefix.
        safe = os.path.normpath("/" + path).lstrip("/")
        if safe.startswith(".."):
            return None
        try:
            raw = self.s3.read_object(f"{filer}/{safe}")
        except Exception:
            return None
        return {"path": safe, "content": raw.decode("utf-8", errors="replace")}

    def delete_version_image(self, tenant_id: str, dataset_id: str, version: int, img_name: str) -> bool:
        """Delete a single image/video object from a version."""
        v = self.get_dataset_version(tenant_id, dataset_id, version)
        if not v or not v.uri:
            return False
        filer = self.s3.s3_uri_to_filer_path(v.uri)
        if not filer:
            return False
        safe = os.path.basename(img_name)
        # Try both images/ and videos/ subfolders
        deleted = False
        for sub in ("images", "videos"):
            if self.s3.delete_object(f"{filer}/{sub}/{safe}"):
                deleted = True
        if deleted:
            v.file_count = max(0, (v.file_count or 1) - 1)
            self.db.commit()
            return True
        return False

    # -- references ---------------------------------------------------------

    def dataset_references(self, tenant_id: str, dataset_id: str) -> list[str]:
        """Return live annotation projects referencing the dataset (blocking delete)."""
        from app.models.orm import AnnotationProject

        rows = self.db.scalars(
            select(AnnotationProject).where(
                and_(
                    AnnotationProject.tenant_id == tenant_id,
                    AnnotationProject.dataset_id == dataset_id,
                    AnnotationProject.status.notin_(["deleted", "cancelled"]),
                )
            )
        ).all()
        return [p.name for p in rows]

    # -- download / export --------------------------------------------------

    def stream_version_zip(self, tenant_id: str, dataset_id: str, version: int, out: Any) -> int:
        """Stream a version's data as a zip, returning bytes written."""
        v = self.get_dataset_version(tenant_id, dataset_id, version)
        if not v or not v.uri:
            return 0
        filer = self.s3.s3_uri_to_filer_path(v.uri)
        buf = io.BytesIO()
        self.s3.download_to_zip(filer, buf)
        data = buf.getvalue()
        out.write(data)
        return len(data)

    # -- async tasks --------------------------------------------------------

    def import_dataset_files(self, tenant_id: str, dataset_id: str, file_paths: list[str], created_by: str) -> dict[str, Any]:
        """Create a dataset_import async task for the worker to process."""
        draft = self.create_draft(tenant_id, dataset_id, "import draft", created_by)
        dest_uri = draft["uri"] if draft else _uri_for(dataset_id, (self.db.scalar(
            select(Dataset.version).where(Dataset.id == dataset_id)
        ) or 0) + 1, tenant_id)
        params = {"dataset_id": dataset_id, "dest_uri": dest_uri, "file_paths": file_paths}
        task = AsyncTask(
            id=new_id("async"),
            tenant_id=tenant_id,
            type="dataset_import",
            status="pending",
            progress=0,
            params=params,
            created_by=created_by,
        )
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return _async_task_dict(task)

    def list_async_tasks(self, tenant_id: str, page: int, page_size: int) -> dict[str, Any]:
        """List async tasks with pagination."""
        from sqlalchemy import func

        total = self.db.scalar(
            select(func.count(AsyncTask.id)).where(AsyncTask.tenant_id == tenant_id)
        ) or 0
        limit, offset = page_bounds(page, page_size)
        rows = self.db.scalars(
            select(AsyncTask)
            .where(AsyncTask.tenant_id == tenant_id)
            .order_by(AsyncTask.created_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        items = [_async_task_dict(t) for t in rows]
        return paginate(items, total, page, page_size)

    def get_async_task(self, tenant_id: str, id: str) -> Optional[dict]:
        """Get an async task by id."""
        t = self.db.scalar(
            select(AsyncTask).where(and_(AsyncTask.id == id, AsyncTask.tenant_id == tenant_id))
        )
        return _async_task_dict(t) if t else None

    def set_async_task_progress(self, id: str, progress: int, status: str) -> None:
        """Update async task progress/status (used by the worker).

        Progress is always written, but the status is NOT allowed to overwrite a
        control/terminal state: a user pause (paused) or stop (cancelled) set by
        the API must survive an in-flight worker progress callback, otherwise a
        stopped task would flip back to "processing" before the gate aborts it.
        """
        t = self.db.get(AsyncTask, id)
        if t:
            t.progress = progress
            if t.status not in ("paused", "cancelled", "completed", "failed"):
                t.status = status
            self.db.commit()

    def fail_async_task(self, id: str, msg: str) -> None:
        """Mark an async task as failed."""
        t = self.db.get(AsyncTask, id)
        if t and t.status != "cancelled":
            t.status = "failed"
            t.error_message = msg
            self.db.commit()

    def complete_async_task(self, id: str) -> None:
        """Mark an async task as completed (progress 100)."""
        t = self.db.get(AsyncTask, id)
        if t and t.status != "cancelled":
            t.status = "completed"
            t.progress = 100
            self.db.commit()

    def merge_async_task_params(self, id: str, extra: dict[str, Any]) -> None:
        """Merge extra fields into an async task's params (used by worker)."""
        t = self.db.get(AsyncTask, id)
        if t:
            params = dict(t.params or {})
            params.update(extra)
            t.params = params
            self.db.commit()

    def control_async_task(self, tenant_id: str, id: str, action: str) -> Optional[dict]:
        """Apply a user control signal (pause / resume / cancel) to an async task.

        The signal is written to params["_control"]; the running worker loop
        polls it via async_task_control_gate(). Cancel also flips the status to
        "cancelled" up-front so the UI stops immediately (the worker aborts on
        its next gate check).
        """
        t = self.db.scalar(
            select(AsyncTask).where(and_(AsyncTask.id == id, AsyncTask.tenant_id == tenant_id))
        )
        if not t:
            return None
        params = dict(t.params or {})
        if action in ("pause", "resume", "cancel"):
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
        return _async_task_dict(t)

    def async_task_control_gate(self, id: str) -> None:
        """Polled by a worker task loop before/while processing each item.

        - cancel: raise AsyncTaskCancelled so the worker stops the loop cleanly.
        - pause: block (sleeping) until a resume or cancel arrives.
        - resume / absent: return immediately.

        Reads a fresh row each call on a short-lived session so the signal set by
        the API process (a different session) is always visible.
        """
        import time as _time

        while True:
            session = SessionLocal()
            try:
                t = session.get(AsyncTask, id)
                if not t:
                    return
                if t.status == "cancelled":
                    raise AsyncTaskCancelled(f"async task {id} cancelled by user")
                control = (t.params or {}).get("_control")
                status = t.status
            finally:
                session.close()
            if control == "cancel" or status == "cancelled":
                raise AsyncTaskCancelled(f"async task {id} cancelled by user")
            if control == "pause" or status == "paused":
                _time.sleep(2)
                continue
            # running / resumed
            return

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _hash(s: str) -> str:
        """Return a hex SHA-256 of the input string."""
        return hashlib.sha256(s.encode()).hexdigest()
