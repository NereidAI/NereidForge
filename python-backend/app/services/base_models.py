# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Base-model pool service: user uploads of CV checkpoints and LLM model dirs.

The pool lives under ``s3://<bucket>/base-models/`` with two fixed namespaces:
``cv/<file>.pt`` for single PyTorch checkpoints and ``llm/<name>/`` for
HF-style model directories. One ``base_models`` row maps 1:1 to an uploaded
file (cv) or directory (llm), giving the training form a managed alternative
to checkpoints baked into trainer images.
"""
from __future__ import annotations

import logging
import posixpath
import re
import zipfile
from typing import Any, BinaryIO, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.clients.s3 import SeaweedFSClient
from app.core.util import new_id, page_bounds
from app.models.orm import BaseModel

logger = logging.getLogger("nereidforge.basemodels")

# Checkpoint suffixes accepted for cv uploads (framework recorded separately).
_CV_SUFFIXES = (".pt", ".pth", ".h5", ".hdf5", ".keras", ".pb", ".onnx")
# Legacy pool entries (before cv/llm split) live at the pool root and are
# migrated into llm/ by reconcile_with_s3.
_LEGACY_ROOT = "base-models"
_CV_ROOT = "base-models/cv"
_LLM_ROOT = "base-models/llm"

# Enumerations for pool metadata (kept in one place so router and UI agree).
FRAMEWORKS = ("pytorch", "tensorflow", "onnx")
CV_TASKS = ("detection", "segmentation", "classification")
LLM_TASKS = ("llm", "vlm")
STATUSES = ("original", "develop", "production")
SOURCES = ("manual", "s3-reconcile", "seed")


def infer_framework(filename: str) -> str:
    """Infer the framework of a checkpoint from its file suffix."""
    lowered = filename.lower()
    if lowered.endswith((".pt", ".pth", ".safetensors", ".bin")):
        return "pytorch"
    if lowered.endswith((".h5", ".hdf5", ".keras", ".pb")):
        return "tensorflow"
    if lowered.endswith(".onnx"):
        return "onnx"
    return "pytorch"


def infer_cv_task(filename: str) -> str:
    """Infer the CV task from a stock checkpoint name (yolo11n-seg.pt, ...)."""
    stem = filename.lower()
    if "-seg." in stem or "-seg-" in stem:
        return "segmentation"
    if "-cls." in stem or "-cls-" in stem:
        return "classification"
    return "detection"


# Weight suffixes from which a stock architecture id can be derived.
_STOCK_WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors")
# Ultralytics naming families whose architecture id is fully encoded in the
# checkpoint file name: yolov8n.pt, yolov8n-seg.pt, yolov5nu.pt, yolo11x-cls.pt,
# yolov9c.pt, yolov10b.pt, yolo12n.pt plus Baidu RT-DETR (rtdetr-l.pt,
# rtdetr-v2-x.pt). These archs are NOT part of the NF torch trainer catalog
# (torchvision/timm/smp only): they target customer-owned ultralytics trainer
# images (ADR-016 BYOI). The match must stay deterministic — arbitrary user
# checkpoint names must never be mistaken for architecture ids.
_STOCK_ARCH_RE = re.compile(
    r"^(?:"
    r"yolo(?:v?\d{1,2})[nstmxlceb](?:u)?(?:-(?:seg|cls|pose|obb))?"
    r"|rtdetr(?:-v2)?-[a-z0-9]+"
    r")$"
)


def infer_stock_arch(filename: str) -> str:
    """Derive the stock architecture id encoded in a checkpoint file name.

    Returns the ultralytics-style arch id (``"yolov8n"``, ``"yolo11x-seg"``)
    when *filename* follows a stock naming family, or an empty string for
    arbitrary user checkpoint names / non-checkpoint paths. The basename is
    matched so ``s3://bucket/base-models/cv/yolov8n.pt`` resolves the same as
    a bare ``yolov8n.pt``; S3 keys always use POSIX separators.
    """
    base = posixpath.basename(str(filename or "").strip().lower())
    for suffix in _STOCK_WEIGHT_SUFFIXES:
        if base.endswith(suffix):
            stem = base[: -len(suffix)]
            return stem if _STOCK_ARCH_RE.match(stem) else ""
    return ""


class BaseModelError(ValueError):
    """Raised on invalid pool upload input (bad name, missing config.json)."""


class BaseModelService:
    """CRUD for the base-model pool on top of Postgres + S3."""

    def __init__(self, db: Session, s3: SeaweedFSClient, bucket: str = "nereidforge") -> None:
        self.db = db
        self.s3 = s3
        self.bucket = bucket

    # -- paths --------------------------------------------------------------

    def _cv_filer(self, filename: str) -> str:
        """Return the filer path for a cv checkpoint."""
        return f"/buckets/{self.bucket}/{_CV_ROOT}/{filename}"

    def _llm_filer(self, name: str) -> str:
        """Return the filer prefix for an llm model directory."""
        return f"/buckets/{self.bucket}/{_LLM_ROOT}/{name}"

    def _cv_uri(self, filename: str) -> str:
        """Return the s3 URI for a cv checkpoint."""
        return f"s3://{self.bucket}/{_CV_ROOT}/{filename}"

    def _llm_uri(self, name: str) -> str:
        """Return the s3 URI for an llm model directory."""
        return f"s3://{self.bucket}/{_LLM_ROOT}/{name}/"

    # -- validation ---------------------------------------------------------

    @staticmethod
    def _safe_segment(value: str) -> str:
        """Validate a single user-supplied path segment (file/dir name)."""
        v = str(value or "").strip()
        if not v or v in (".", "..") or "/" in v or "\\" in v or v.startswith("."):
            raise BaseModelError("invalid name: a single non-empty file/dir segment is required")
        if any(ch in v for ch in ("\x00",)):
            raise BaseModelError("invalid name")
        return v

    def _check_cv_filename(self, filename: str) -> str:
        """Validate a cv checkpoint filename including its suffix."""
        name = self._safe_segment(filename)
        if not name.lower().endswith(_CV_SUFFIXES):
            raise BaseModelError(f"cv uploads must be a {'/'.join(_CV_SUFFIXES)} model file")
        return name

    @staticmethod
    def _safe_relpath(rel: str) -> str:
        """Validate an in-archive relative path (zip entry / llm file)."""
        p = str(rel or "").replace("\\", "/").strip().lstrip("/")
        if not p:
            return ""
        # Normalise and reject any path escaping the target directory (zip slip).
        normalized = posixpath.normpath(p)
        if normalized in (".", "..") or normalized.startswith("../") or "/../" in normalized:
            raise BaseModelError("unsafe path inside archive")
        return normalized

    # -- listing ------------------------------------------------------------

    def list_paged(
        self, tenant_id: str, kind: str = "", search: str = "",
        page: int = 1, page_size: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        """List pool rows (newest first), filtered by kind and/or name search."""
        stmt = select(BaseModel).where(BaseModel.tenant_id == tenant_id)
        if kind:
            stmt = stmt.where(BaseModel.kind == kind)
        if search:
            stmt = stmt.where(BaseModel.name.ilike(f"%{search.strip()}%"))
        total = int(self.db.scalar(select(func.count()).select_from(stmt.subquery())) or 0)
        limit, offset = page_bounds(page, page_size)
        rows = self.db.scalars(
            stmt.order_by(BaseModel.created_at.desc()).limit(limit).offset(offset)
        ).all()
        return [self._dict(r) for r in rows], total

    def list_for_training(self, tenant_id: str, kind: str) -> list[dict[str, Any]]:
        """List all rows of one kind for the training-form selectors."""
        rows = self.db.scalars(
            select(BaseModel)
            .where(BaseModel.tenant_id == tenant_id, BaseModel.kind == kind)
            .order_by(BaseModel.name)
        ).all()
        return [self._dict(r) for r in rows]

    def get(self, tenant_id: str, row_id: str) -> Optional[BaseModel]:
        """Get one pool row scoped to the tenant."""
        return self.db.scalar(
            select(BaseModel).where(
                BaseModel.id == row_id, BaseModel.tenant_id == tenant_id
            )
        )

    # -- upload: cv ---------------------------------------------------------

    def upload_cv(
        self, tenant_id: str, filename: str, fileobj: BinaryIO, actor: str,
        framework: str = "", task: str = "", status: str = "develop",
        description: str = "",
    ) -> dict[str, Any]:
        """Upload one checkpoint to cv/ and upsert its pool row."""
        name = self._check_cv_filename(filename)
        filer = self._cv_filer(name)
        self.s3.upload_fileobj(filer, fileobj)
        uri = self._cv_uri(name)
        size = self.s3.object_size(uri)
        row = self._upsert_row(
            tenant_id, "cv", name, uri, size, actor,
            framework=framework or infer_framework(name),
            task=task or infer_cv_task(name),
            status=status, description=description, source="manual",
        )
        logger.info("base model cv uploaded: %s (%d bytes)", uri, size)
        return self._dict(row)

    # -- upload: llm --------------------------------------------------------

    def upload_llm_zip(
        self, tenant_id: str, name: str, fileobj: BinaryIO, actor: str,
        framework: str = "pytorch", task: str = "llm", status: str = "develop",
        description: str = "",
    ) -> dict[str, Any]:
        """Extract an uploaded HF-model zip into llm/<name>/ and upsert the row.

        The zip is read straight from the seekable spooled upload (never fully
        buffered in RAM); members are written one at a time.
        """
        dir_name = self._safe_segment(name)
        fileobj.seek(0)
        try:
            zf = zipfile.ZipFile(fileobj)
        except zipfile.BadZipFile as exc:
            raise BaseModelError("uploaded file is not a valid zip") from exc
        prefix = self._llm_filer(dir_name)
        total = 0
        n_files = 0
        with zf:
            pairs: list[tuple[zipfile.ZipInfo, str]] = []
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rel = self._safe_relpath(info.filename)
                if rel:
                    pairs.append((info, rel))
            self._require_config_names([rel for _, rel in pairs])
            for info, rel in pairs:
                with zf.open(info) as member:
                    size = self._stream_member(f"{prefix}/{rel}", member)
                total += size
                n_files += 1
        uri = self._llm_uri(dir_name)
        row = self._upsert_row(
            tenant_id, "llm", dir_name, uri, total, actor,
            framework=framework, task=task, status=status,
            description=description, source="manual",
        )
        logger.info("base model llm zip uploaded: %s (%d files, %d bytes)",
                    uri, n_files, total)
        return self._dict(row)

    def upload_llm_files(
        self,
        tenant_id: str,
        name: str,
        files: list[tuple[str, BinaryIO]],
        actor: str,
        framework: str = "pytorch",
        task: str = "llm",
        status: str = "develop",
        description: str = "",
    ) -> dict[str, Any]:
        """Upload individual files into llm/<name>/ (relative paths kept).

        Files stream to S3 one at a time; config.json must be among them.
        """
        dir_name = self._safe_segment(name)
        prepared = [(self._safe_relpath(rel), fileobj) for rel, fileobj in files]
        prepared = [(rel, fileobj) for rel, fileobj in prepared if rel]
        self._require_config_names([rel for rel, _ in prepared])
        prefix = self._llm_filer(dir_name)
        total = 0
        for rel, fileobj in prepared:
            fileobj.seek(0)
            total += self._stream_member(f"{prefix}/{rel}", fileobj)
        uri = self._llm_uri(dir_name)
        row = self._upsert_row(
            tenant_id, "llm", dir_name, uri, total, actor,
            framework=framework, task=task, status=status,
            description=description, source="manual",
        )
        logger.info("base model llm files uploaded: %s (%d files, %d bytes)",
                    uri, len(prepared), total)
        return self._dict(row)

    def _stream_member(self, filer_path: str, fileobj: BinaryIO) -> int:
        """Stream one member to S3 and return its stored byte size."""
        self.s3.upload_fileobj(filer_path, fileobj)
        return self.s3.object_size(filer_path)

    @staticmethod
    def _require_config_names(rels: list[str]) -> None:
        """Reject an llm upload lacking config.json at its root."""
        if "config.json" not in rels:
            raise BaseModelError(
                "llm model directory must include config.json at its root"
            )

    def _write_llm_members(self, prefix: str, members: list[tuple[str, bytes]]) -> int:
        """Write extracted members under the llm prefix; return total bytes."""
        total = 0
        for rel, content in members:
            self.s3.put_object(f"{prefix}/{rel}", content)
            total += len(content)
        return total

    @staticmethod
    def _require_config(members: list[tuple[str, bytes]]) -> None:
        """Reject an llm upload lacking config.json at its root."""
        if not any(rel == "config.json" for rel, _ in members):
            raise BaseModelError(
                "llm model directory must include config.json at its root"
            )

    # -- delete -------------------------------------------------------------

    def delete(self, tenant_id: str, row_id: str) -> bool:
        """Delete a pool row plus its S3 file (cv) or prefix (llm)."""
        row = self.get(tenant_id, row_id)
        if row is None:
            return False
        filer = self.s3.s3_uri_to_filer_path(row.uri)
        if row.kind == "cv":
            self.s3.delete_object(filer)
        else:
            for path in self.s3.walk(filer):
                self.s3.delete_object(path)
        self.db.delete(row)
        self.db.commit()
        logger.info("base model deleted: %s/%s", row.kind, row.name)
        return True

    # -- upsert -------------------------------------------------------------

    def _upsert_row(
        self, tenant_id: str, kind: str, name: str, uri: str,
        size_bytes: int, actor: str,
        framework: str = "pytorch", task: str = "detection",
        status: str = "develop", description: str = "", source: str = "manual",
    ) -> BaseModel:
        """Create a pool row or replace the existing one in place."""
        row = self.db.scalar(select(BaseModel).where(
            BaseModel.tenant_id == tenant_id,
            BaseModel.kind == kind,
            BaseModel.name == name,
        ))
        if row is None:
            row = BaseModel(
                id=new_id("bm"), tenant_id=tenant_id, kind=kind, name=name,
                uri=uri, size_bytes=size_bytes, uploaded_by=actor,
                framework=framework, task=task, status=status,
                description=description, source=source,
            )
            self.db.add(row)
        else:
            row.uri = uri
            row.size_bytes = size_bytes
            row.uploaded_by = actor
            row.framework = framework
            row.task = task
            row.status = status
            row.description = description
            row.source = source
        # Persist immediately: uploads are multi-file API requests and every
        # other NF service commits explicitly (no request-level commit).
        self.db.commit()
        return row

    # -- reconcile (legacy entries + backfill) ------------------------------

    def reconcile_with_s3(self, tenant_id: str = "default") -> int:
        """Align pool rows with what S3 already holds.

        Migrates legacy root-level model directories into ``llm/`` (copy then
        delete), and backfills missing rows for files already present under
        ``cv/`` or ``llm/``. Returns the number of rows ensured.
        """
        ensured = 0
        root = f"/buckets/{self.bucket}/{_LEGACY_ROOT}"
        try:
            entries = self.s3.list_entries(root)
        except Exception as exc:  # noqa: BLE001
            logger.warning("base model reconcile skipped: %s", exc)
            return 0
        for e in entries:
            if not e.is_dir or e.name in ("cv", "llm"):
                continue
            self._migrate_legacy_dir(e.name, tenant_id)
        ensured += self._backfill_kind(tenant_id, "cv")
        ensured += self._backfill_kind(tenant_id, "llm")
        return ensured

    def _migrate_legacy_dir(self, name: str, tenant_id: str) -> None:
        """Move a legacy base-models/<name>/ directory into llm/<name>/."""
        src_prefix = f"/buckets/{self.bucket}/{_LEGACY_ROOT}/{name}"
        dst_prefix = self._llm_filer(name)
        for path in self.s3.walk(src_prefix):
            rel = path[len(src_prefix):].lstrip("/")
            try:
                content = self.s3.read_object(path)
            except Exception:
                # Another replica migrating concurrently may have already
                # copied and deleted this object; nothing to do for it.
                continue
            self.s3.put_object(f"{dst_prefix}/{rel}", content)
            self.s3.delete_object(path)
        logger.info("migrated legacy base model %s -> llm/", name)

    def _backfill_kind(self, tenant_id: str, kind: str) -> int:
        """Create missing pool rows from existing cv/ or llm/ S3 content."""
        prefix = f"/buckets/{self.bucket}/base-models/{kind}"
        count = 0
        if kind == "cv":
            files = self.s3.list_files_with_size(prefix)
            for rel, size in files:
                uri = self._cv_uri(rel)
                if self._row_exists(tenant_id, kind, rel):
                    continue
                self.db.add(BaseModel(
                    id=new_id("bm"), tenant_id=tenant_id, kind=kind, name=rel,
                    uri=uri, size_bytes=size, uploaded_by="s3-reconcile",
                    framework=infer_framework(rel), task=infer_cv_task(rel),
                    status="original", source="s3-reconcile",
                ))
                count += 1
            self.db.flush()
            return count
        for e in self.s3.list_entries(prefix):
            if not e.is_dir or self._row_exists(tenant_id, kind, e.name):
                continue
            dir_prefix = f"{prefix}/{e.name}"
            size = sum(sz for _, sz in self.s3.list_files_with_size(dir_prefix))
            self.db.add(BaseModel(
                id=new_id("bm"), tenant_id=tenant_id, kind=kind, name=e.name,
                uri=self._llm_uri(e.name), size_bytes=size,
                uploaded_by="s3-reconcile", framework="pytorch", task="llm",
                status="original", source="s3-reconcile",
            ))
            count += 1
        self.db.flush()
        return count

    def _row_exists(self, tenant_id: str, kind: str, name: str) -> bool:
        """Return whether a pool row for the triple already exists."""
        return self.db.scalar(select(BaseModel.id).where(
            BaseModel.tenant_id == tenant_id,
            BaseModel.kind == kind,
            BaseModel.name == name,
        )) is not None

    # -- output -------------------------------------------------------------

    @staticmethod
    def _dict(r: BaseModel) -> dict[str, Any]:
        """Serialize a pool row for API responses."""
        return {
            "id": r.id,
            "kind": r.kind,
            "name": r.name,
            "uri": r.uri,
            "size_bytes": int(r.size_bytes or 0),
            "framework": r.framework or "pytorch",
            "task": r.task or "detection",
            "status": r.status or "develop",
            "description": r.description or "",
            "source": r.source or "manual",
            "uploaded_by": r.uploaded_by,
            "created_at": r.created_at.isoformat() if getattr(r, "created_at", None) else None,
        }
