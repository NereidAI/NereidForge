# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Dataset routes: datasets, versions, drafts, images, async tasks, image proxies."""
from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.helpers import error, get_svc, not_found
from app.core.config import config
from app.core.util import paginate
from app.services import Services

router = APIRouter()


@router.post("/datasets/notebook-staging/upload")
async def upload_notebook_staging(request: Request, svc: Services = Depends(get_svc)):
    """Stage a Kale/Jupyter .ipynb for a notebook (batch) job.

    Uses the fixed, UI-known prefix s3://<bucket>/notebook-staging/v1/<file>
    (ADR-013 Case 2): the frontend derives the notebook_uri from the staged
    file name and the worker reads the object back when it compiles and
    submits the job. The original file name is kept as the last path segment
    so both sides can agree on the uri without a stored mapping.
    """
    form = await request.form()
    files = form.getlist("files")
    if not files:
        raise error(400, "no files uploaded")
    bucket = svc.s3.s3_bucket()
    filer = f"/buckets/{bucket}/notebook-staging/v1"
    uris: list[str] = []
    for f in files:
        if not (f.filename or "").endswith(".ipynb"):
            raise error(400, f"not a .ipynb notebook: {f.filename}")
        data = await f.read()
        if not data:
            raise error(400, f"empty notebook file: {f.filename}")
        safe = os.path.basename(f.filename or "notebook.ipynb")
        svc.s3.put_object(f"{filer}/{safe}", data)
        uris.append(f"s3://{bucket}/notebook-staging/v1/{safe}")
    return {"status": "ok", "uris": uris}


class DatasetCreate(BaseModel):
    """Create dataset request body."""
    name: str
    description: Optional[str] = ""
    uri: Optional[str] = ""
    # auto: the format is decided by content detection on the first finalized
    # version, so creation no longer asks for a format that may be wrong.
    format: Optional[str] = "auto"


class DatasetUpdate(BaseModel):
    """Update dataset request body."""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None


class VersionCreate(BaseModel):
    """Create version request body."""
    uri: Optional[str] = ""
    format: Optional[str] = "coco"
    description: Optional[str] = ""


class VersionRegister(BaseModel):
    """Register an existing S3 prefix as a finalized trainable version."""
    uri: str
    format: Optional[str] = "auto"
    description: Optional[str] = ""


class VersionUpdate(BaseModel):
    """Update version request body."""
    description: Optional[str] = None


class CardContent(BaseModel):
    """Save dataset card (README.md) request body; full markdown incl. yaml frontmatter."""
    content: str = ""


@router.post("/datasets")
async def create_dataset(req: DatasetCreate, request: Request, svc: Services = Depends(get_svc)):
    """Create a new dataset."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    try:
        return svc.dataset.create_dataset(tenant, req.name, req.description or "", req.uri or "", req.format or "auto", user)
    except ValueError as e:
        raise error(400, str(e))


@router.get("/datasets")
async def list_datasets(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    search: str = "",
    status: str = "",
    svc: Services = Depends(get_svc),
):
    """List datasets with filters and pagination."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    items, total = svc.dataset.list_datasets(tenant, page, page_size, search, status)
    return paginate(items, total, page, page_size)


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Get a single dataset."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    ds = svc.dataset.get_dataset(tenant, dataset_id)
    if not ds:
        raise not_found()
    return ds


@router.patch("/datasets/{dataset_id}")
async def update_dataset(dataset_id: str, req: DatasetUpdate, request: Request, svc: Services = Depends(get_svc)):
    """Update a dataset."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    ds = svc.dataset.update_dataset(tenant, dataset_id, patch)
    if not ds:
        raise not_found()
    return ds


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Delete a dataset (blocked when referenced by live annotation projects)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    try:
        ok = svc.dataset.delete_dataset(tenant, dataset_id)
    except ValueError as e:
        raise error(409, str(e))
    if not ok:
        raise not_found()
    return {"deleted": True, "dataset_id": dataset_id}


@router.get("/datasets/{dataset_id}/references")
async def dataset_references(dataset_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Return live annotation projects referencing the dataset."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    return {"references": svc.dataset.dataset_references(tenant, dataset_id)}


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Stream the dataset's latest version as a zip."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    ds = svc.dataset.get_dataset(tenant, dataset_id)
    if not ds:
        raise not_found()
    vers = svc.dataset.list_dataset_versions(tenant, dataset_id)
    if not vers:
        raise not_found()
    ver = vers[0]
    def gen():
        svc.dataset.stream_version_zip(tenant, dataset_id, ver["version"], None)
        # read via s3 walk to stream bytes
    data = _read_version_bytes(svc, tenant, dataset_id, ver["version"])
    return StreamingResponse(iter([data]), media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename={dataset_id}.zip"})


def _read_version_bytes(svc: Services, tenant: str, dataset_id: str, version: int) -> bytes:
    """Read a version's zip bytes (helper for download/export)."""
    import io

    buf = io.BytesIO()
    svc.dataset.stream_version_zip(tenant, dataset_id, version, buf)
    return buf.getvalue()


# -- versions ---------------------------------------------------------------


@router.post("/datasets/{dataset_id}/versions")
async def create_version(dataset_id: str, req: VersionCreate, request: Request, svc: Services = Depends(get_svc)):
    """Create a dataset version."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    ds = svc.dataset.get_dataset(tenant, dataset_id)
    if not ds:
        raise not_found()
    ver = svc.dataset.create_draft(tenant, dataset_id, req.description or "new version", "anonymous")
    return ver


@router.get("/datasets/{dataset_id}/versions")
async def list_versions(
    dataset_id: str, request: Request, annotated: int = 0,
    page: int = 1, page_size: int = 20, svc: Services = Depends(get_svc),
):
    """List a dataset's versions (paginated)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    items, total = svc.dataset.list_dataset_versions_paged(
        tenant, dataset_id, page, page_size, annotated_only=bool(annotated)
    )
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("/datasets/{dataset_id}/versions/draft")
async def create_draft(dataset_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Create a draft version."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    draft = svc.dataset.create_draft(tenant, dataset_id, "draft", user)
    if not draft:
        raise not_found()
    return draft


@router.post("/datasets/{dataset_id}/versions/register")
async def register_version(dataset_id: str, req: VersionRegister, request: Request, svc: Services = Depends(get_svc)):
    """Register an existing S3 prefix (no copy) as a finalized trainable version.

    Used for pre-staged corpora such as sharegpt JSONL + images/ VLM datasets.
    """
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    try:
        return svc.dataset.register_s3_version(
            tenant, dataset_id, req.uri, req.format or "auto", req.description or "", user
        )
    except ValueError as e:
        raise error(400, str(e))


@router.post("/datasets/{dataset_id}/versions/draft/{draft_id}/finalize")
async def finalize_draft(dataset_id: str, draft_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Finalize a draft into a read-only version."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    v = svc.dataset.finalize_draft(tenant, dataset_id, draft_id)
    if not v:
        raise not_found()
    return v


@router.delete("/datasets/{dataset_id}/versions/draft/{draft_id}")
async def delete_draft(dataset_id: str, draft_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Delete a draft version."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    if not svc.dataset.delete_draft(tenant, dataset_id, draft_id):
        raise not_found()
    return {"deleted": True}


@router.get("/datasets/{dataset_id}/versions/{version_id}")
async def get_version(dataset_id: str, version_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Get a dataset version by version number."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    v = svc.dataset.get_dataset_version(tenant, dataset_id, int(version_id))
    if not v:
        raise not_found()
    from app.services.dataset import _version_dict

    return _version_dict(v)


@router.patch("/datasets/{dataset_id}/versions/{version_id}")
async def update_version(dataset_id: str, version_id: str, req: VersionUpdate, request: Request, svc: Services = Depends(get_svc)):
    """Update a version's description."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    v = svc.dataset.update_dataset_version(tenant, dataset_id, int(version_id), req.description or "")
    if not v:
        raise not_found()
    return v


@router.delete("/datasets/{dataset_id}/versions/{version_id}")
async def delete_version(dataset_id: str, version_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Delete a version."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    if not svc.dataset.delete_dataset_version(tenant, dataset_id, int(version_id)):
        raise not_found()
    return {"deleted": True}


@router.post("/datasets/{dataset_id}/versions/{version_id}/copy-to-draft")
async def copy_to_draft(dataset_id: str, version_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Copy a version's data into a new draft."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    draft = svc.dataset.copy_version_to_draft(tenant, dataset_id, version_id, user)
    if not draft:
        raise not_found()
    return draft


@router.get("/datasets/{dataset_id}/versions/{version_id}/images")
async def list_version_images(dataset_id: str, version_id: str, request: Request, page: int = 1, page_size: int = 48, svc: Services = Depends(get_svc)):
    """List images in a version (paginated; preview grids should never load all)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    images, total = svc.dataset.list_version_images(tenant, dataset_id, int(version_id), page, page_size)
    return {"dataset_id": dataset_id, "version": int(version_id), "images": images, "total": total, "page": page, "page_size": page_size}


@router.delete("/datasets/{dataset_id}/versions/{version_id}/images/{image_name}")
async def delete_version_image(dataset_id: str, version_id: str, image_name: str, request: Request, svc: Services = Depends(get_svc)):
    """Delete a single image from a version."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    if not svc.dataset.delete_version_image(tenant, dataset_id, int(version_id), image_name):
        raise not_found()
    return {"deleted": True}


@router.get("/datasets/{dataset_id}/versions/{version_id}/files")
async def list_version_files(dataset_id: str, version_id: str, request: Request, svc: Services = Depends(get_svc)):
    """List all files (recursive) under a version for the file browser."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    files = svc.dataset.list_version_files(tenant, dataset_id, int(version_id))
    if files is None:
        raise not_found()
    return {"dataset_id": dataset_id, "version": int(version_id), "files": files, "total": len(files)}


@router.get("/datasets/{dataset_id}/versions/{version_id}/card")
async def get_version_card(dataset_id: str, version_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Read the version's README.md dataset card (markdown + yaml frontmatter)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    card = svc.dataset.get_version_card(tenant, dataset_id, int(version_id))
    if card is None:
        raise not_found()
    return {"dataset_id": dataset_id, "version": int(version_id), **card}


@router.put("/datasets/{dataset_id}/versions/{version_id}/card")
async def save_version_card(dataset_id: str, version_id: str, body: CardContent, request: Request, svc: Services = Depends(get_svc)):
    """Write the version's README.md dataset card (full markdown incl. frontmatter)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    card = svc.dataset.save_version_card(tenant, dataset_id, int(version_id), body.content)
    if card is None:
        raise not_found()
    return {"dataset_id": dataset_id, "version": int(version_id), **card}


@router.get("/datasets/{dataset_id}/versions/{version_id}/file")
async def read_version_file(dataset_id: str, version_id: str, request: Request, path: str, svc: Services = Depends(get_svc)):
    """Read a single text file's content for the preview pane."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    result = svc.dataset.read_version_file(tenant, dataset_id, int(version_id), path)
    if result is None:
        raise not_found()
    return {"dataset_id": dataset_id, "version": int(version_id), **result}



# -- async tasks ------------------------------------------------------------

@router.get("/async-tasks")
async def list_async_tasks(request: Request, page: int = 1, page_size: int = 20, svc: Services = Depends(get_svc)):
    """List async tasks with pagination."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    return svc.dataset.list_async_tasks(tenant, page, page_size)


@router.get("/async-tasks/{task_id}")
async def get_async_task(task_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Get one async task (imports, packaging, deploy verification, ...)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    task = svc.dataset.get_async_task(tenant, task_id)
    if not task:
        from app.api.helpers import not_found
        raise not_found()
    return task


@router.post("/datasets/{dataset_id}/upload")
async def upload_dataset(dataset_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Upload files to a dataset (creates a dataset_import async task).

    Files are staged straight into S3 (SeaweedFS) under a per-request prefix
    instead of a pod-local temp dir, because the async worker that consumes
    them runs in a different container. The ORIGINAL filename is kept as the
    last path segment so the worker can derive the object name from it (no
    upload-prefix leaks into dataset/CVAT task names).
    """
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    form = await request.form()
    files = form.getlist("files")
    if not files:
        raise error(400, "no files uploaded")
    max_files = config.storage.upload_max_files
    max_bytes = config.storage.upload_max_bytes
    if len(files) > max_files:
        raise error(400, f"too many files: {len(files)} exceeds max {max_files}")
    staging = f".uploads/{uuid.uuid4().hex}"
    bucket = svc.s3.s3_bucket()
    filer = f"/buckets/{bucket}/{staging}"
    uris: list[str] = []
    total = 0
    for f in files:
        data = await f.read()
        total += len(data)
        if total > max_bytes:
            raise error(413, f"upload too large: {total} bytes exceeds max {max_bytes}")
        safe = os.path.basename(f.filename or "file")
        svc.s3.put_object(f"{filer}/{safe}", data)
        uris.append(f"s3://{bucket}/{staging}/{safe}")
    task = svc.dataset.import_dataset_files(tenant, dataset_id, uris, user)
    return {"status": task["status"], "task_id": task["id"]}
