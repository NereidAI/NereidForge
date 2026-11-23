# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Project routes: annotation projects, LS tasks, pre-annotation, batch accept."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from app.api.helpers import error, get_svc, not_found
from app.core.util import paginate
from app.services import Services

router = APIRouter()


class ProjectCreate(BaseModel):
    """Create annotation project request body."""
    name: str
    description: Optional[str] = ""
    task_type: Optional[str] = "detection"
    annotation_tool: Optional[str] = "label-studio"
    schema: Optional[dict] = None
    dataset_id: Optional[str] = None
    dataset_version: Optional[int] = 1
    import_uri: Optional[str] = ""
    # Assigned annotator (a users.id); empty means unassigned.
    assignee_id: Optional[str] = ""


class ProjectUpdate(BaseModel):
    """Update annotation project request body."""
    name: Optional[str] = None
    description: Optional[str] = None
    schema: Optional[dict] = None
    assignee_id: Optional[str] = None
    # Persisted per-project pre-annotation model choice
    # {"model_name": str, "model_version": str}.
    preannotate_model: Optional[dict] = None


class SaveAnnotation(BaseModel):
    """Save annotation request body."""
    result: list


class BatchAccept(BaseModel):
    """Batch accept request body."""
    task_ids: list[int]


class ExportRequest(BaseModel):
    """Export dataset request body."""
    name: Optional[str] = None
    description: Optional[str] = ""
    format: Optional[str] = "coco"


@router.post("/projects")
async def create_project(req: ProjectCreate, request: Request, svc: Services = Depends(get_svc)):
    """Create an annotation project (provisions a CVAT or LS project by task_type)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    host = request.headers.get("Host", "localhost")
    try:
        p = svc.project.create(
            tenant, host, req.name, req.description or "", req.annotation_tool or "label-studio",
            req.task_type or "detection", req.schema or {}, req.import_uri or "",
            req.dataset_id or "", req.dataset_version or 1, user, req.assignee_id or "",
        )
    except ValueError as e:
        raise error(400, str(e))
    return {
        "project_id": p["id"],
        "external_id": p["external_id"],
        "external_url": p["external_url"],
        "project": p,
    }


@router.get("/projects")
async def list_projects(
    request: Request, page: int = 1, page_size: int = 20,
    dataset_id: str = "", status: str = "", assignee_id: str = "", search: str = "",
    svc: Services = Depends(get_svc),
):
    """List annotation projects with filters/search."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    items, total = svc.project.list(tenant, page, page_size, dataset_id, status, assignee_id, search)
    return paginate(items, total, page, page_size)


@router.get("/projects/{id}")
async def get_project(id: str, request: Request, svc: Services = Depends(get_svc)):
    """Get an annotation project."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    p = svc.project.get(tenant, id)
    if not p:
        raise not_found()
    svc.project.sync_progress(tenant, id)
    return svc.project.get(tenant, id)


@router.patch("/projects/{id}")
async def update_project(id: str, req: ProjectUpdate, request: Request, svc: Services = Depends(get_svc)):
    """Update an annotation project."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    p = svc.project.update(tenant, id, patch)
    if not p:
        raise not_found()
    return p


@router.delete("/projects/{id}")
async def delete_project(id: str, request: Request, svc: Services = Depends(get_svc)):
    """Delete an annotation project."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    try:
        if not svc.project.delete(tenant, id):
            raise not_found()
    except ValueError as e:
        raise error(409, str(e))
    return {"deleted": True}


@router.post("/projects/{id}/start")
async def start_project(id: str, request: Request, svc: Services = Depends(get_svc)):
    """Start an annotation project."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    p = svc.project.start(tenant, id)
    if not p:
        raise not_found()
    return p


@router.post("/projects/{id}/complete")
async def complete_project(id: str, request: Request, svc: Services = Depends(get_svc)):
    """Complete an annotation project."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    p = svc.project.complete(tenant, id)
    if not p:
        raise not_found()
    return p


@router.post("/projects/{id}/cancel")
async def cancel_project(id: str, request: Request, svc: Services = Depends(get_svc)):
    """Cancel an annotation project."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    p = svc.project.cancel(tenant, id)
    if not p:
        raise not_found()
    return p


class AddLabelsRequest(BaseModel):
    """Add one or more labels to a project while annotating."""
    labels: list[str]


@router.post("/projects/{id}/labels")
async def add_project_labels(id: str, req: AddLabelsRequest, request: Request, svc: Services = Depends(get_svc)):
    """Append labels to an existing project (synced to LS label_config / CVAT)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    try:
        p = svc.project.add_labels(tenant, id, req.labels)
    except ValueError as e:
        raise error(400, str(e))
    if not p:
        raise not_found()
    return p


@router.post("/projects/{id}/import-tasks")
async def reimport_tasks(id: str, request: Request, svc: Services = Depends(get_svc)):
    """Re-import tasks into the project's LS project."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    host = request.headers.get("Host", "localhost")
    n = svc.project.reimport_tasks(tenant, host, id)
    return {"imported": n}


# -- LS tasks ---------------------------------------------------------------


@router.get("/projects/{id}/ls-tasks")
async def ls_tasks(
    id: str, request: Request, page: int = 1, page_size: int = 20,
    svc: Services = Depends(get_svc),
):
    """List Label Studio tasks for the project."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    ls_id, items, total = svc.project.ls_tasks(tenant, id, page, page_size)
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "ls_project_id": ls_id,
    }


@router.post("/projects/{id}/ls-tasks/{task_id}/annotations")
async def save_annotation(id: str, task_id: int, req: SaveAnnotation, request: Request, svc: Services = Depends(get_svc)):
    """Save an annotation back to LS."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    svc.project.ls_save_annotation(tenant, id, task_id, req.result)
    return {"saved": True, "task_id": task_id}


# -- CVAT tasks -------------------------------------------------------------

@router.get("/projects/{id}/cvat-tasks")
async def cvat_tasks(
    id: str, request: Request, page: int = 1, page_size: int = 20,
    svc: Services = Depends(get_svc),
):
    """List CVAT tasks for the project."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    cvat_id, items, total = svc.project.cvat_tasks(tenant, id, page, page_size)
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "cvat_project_id": cvat_id,
    }

@router.get("/projects/{id}/cvat-tasks/{task_id}")
async def cvat_task_detail(id: str, task_id: int, request: Request, svc: Services = Depends(get_svc)):
    """Get CVAT task details with annotations."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    task = svc.project.cvat_task_detail(tenant, id, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.post("/projects/{id}/cvat-tasks/{task_id}/annotations")
async def cvat_save_annotation(id: str, task_id: int, req: SaveAnnotation, request: Request, svc: Services = Depends(get_svc)):
    """Save an annotation back to CVAT."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    svc.project.cvat_save_annotation(tenant, id, task_id, req.result)
    return {"saved": True, "task_id": task_id}


@router.get("/projects/{id}/cvat-tasks/{task_id}/annotation")
async def cvat_task_annotation(id: str, task_id: int, request: Request, svc: Services = Depends(get_svc)):
    """Full annotation state of a CVAT task for the NF-native editor."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    try:
        return svc.project.cvat_task_annotation_detail(tenant, id, task_id)
    except ValueError as e:
        raise error(404, str(e))


class KfPayload(BaseModel):
    """One track keyframe: rectangle bbox in absolute pixel coords, or a
    polygon (shape='polygon' + flat vertex list in poly)."""
    frame: int
    points: list[float]
    outside: bool = False
    occluded: bool = False
    shape: Optional[str] = None  # rectangle | polygon
    poly: Optional[list[float]] = None


class TrackPayload(BaseModel):
    """One track submitted by the NF-native editor."""
    id: Optional[int] = None
    label_id: int
    keyframes: list[KfPayload]


class ReplaceTracksRequest(BaseModel):
    """Replace-all-annotations request body for the NF-native editor."""
    tracks: list[TrackPayload]


@router.put("/projects/{id}/cvat-tasks/{task_id}/annotations")
async def cvat_replace_annotations(id: str, task_id: int, req: ReplaceTracksRequest, request: Request, svc: Services = Depends(get_svc)):
    """Replace all annotations of a CVAT task with the editor's tracks."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    try:
        n = svc.project.cvat_replace_tracks(
            tenant, id, task_id,
            [t.model_dump() for t in req.tracks],
        )
    except ValueError as e:
        raise error(404, str(e))
    return {"saved": True, "task_id": task_id, "tracks": n}


@router.get("/projects/{id}/cvat-tasks/{task_id}/frames/{frame}")
def cvat_frame(id: str, task_id: int, frame: int, request: Request, quality: str = "compressed", svc: Services = Depends(get_svc)):
    """Proxy a single frame image from CVAT (authenticated XHR -> blob).

    This is a *sync* endpoint on purpose: the CVAT client uses blocking httpx.
    Declaring it ``async`` would run the blocking download on the event loop
    and serialize every concurrent frame prefetch during playback (measured
    ~6 fps aggregate). FastAPI runs a plain ``def`` route in a worker thread,
    so concurrent frame requests proceed in parallel.
    """
    tenant = request.headers.get("X-Tenant-Id", "default")
    try:
        data = svc.project.cvat_frame(tenant, id, task_id, frame, quality)
    except ValueError as e:
        raise error(404, str(e))
    # Video tasks return PNG even at quality=compressed; image sets return
    # JPEG. Sniff the magic bytes so the browser gets the right content type.
    media_type = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return Response(content=data, media_type=media_type)


# -- pre-annotation ---------------------------------------------------------
class PreannotateRequest(BaseModel):
    limit: int = 0
    ml_backend_id: str = ""
    task_ids: str = ""
    frame_stride: int = 5
    # None = use the selected ML backend's configured threshold (or 0.25).
    score_threshold: float | None = None
    # Optional explicit model choice; overrides the project's remembered
    # pre-annotation preference when both are supplied.
    model_name: str = ""
    model_version: str = ""

@router.post("/projects/{id}/preannotate")
async def preannotate(
    id: str, request: Request, req: PreannotateRequest, svc: Services = Depends(get_svc),
):
    """Trigger semi-auto pre-annotation (async task).

    frame_stride applies to CVAT video tasks only: sample every Nth frame
    for ML inference; tracks are linked between sampled keyframes.
    score_threshold overrides the ML backend's per-request detection floor;
    when omitted the backend row's threshold is used (default 0.25, as the
    env default 0.5 misses small objects).
    """
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    ids = [int(x) for x in req.task_ids.split(",") if x.strip().isdigit()] if req.task_ids else []
    try:
        task = svc.project.preannotate(
            tenant, id, req.limit, req.ml_backend_id, ids, user,
            frame_stride=req.frame_stride,
            score_threshold=req.score_threshold,
            model_name=req.model_name,
            model_version=req.model_version,
        )
    except ValueError as e:
        raise error(409, str(e))
    return {"task_id": task["task_id"], "status": task["status"]}


@router.post("/projects/{id}/batch-accept")
async def batch_accept(id: str, req: BatchAccept, request: Request, svc: Services = Depends(get_svc)):
    """Batch-accept predictions as annotations (async task)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    task = svc.project.batch_accept(tenant, id, req.task_ids, user)
    return {"task_id": task["task_id"], "status": task["status"]}


@router.get("/projects/{id}/async-tasks/active")
async def get_active_async_task(id: str, request: Request, svc: Services = Depends(get_svc)):
    """Return the project's currently running/paused annotation task, if any
    (so the UI can restore the inline progress bar after a refresh)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    task = svc.project.get_active_async_task(tenant, id)
    return {"task": task}


@router.get("/projects/{id}/async-tasks/{async_task_id}")
async def get_async_task(id: str, async_task_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Get an async task's progress."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    task = svc.project.get_async_task(tenant, async_task_id)
    if not task:
        raise not_found()
    return task


class AsyncTaskControl(BaseModel):
    """Body for pausing / resuming / cancelling a running async task."""
    action: str  # "pause" | "resume" | "cancel"


@router.post("/projects/{id}/async-tasks/{async_task_id}/control")
async def control_async_task(
    id: str, async_task_id: str, req: AsyncTaskControl,
    request: Request, svc: Services = Depends(get_svc),
):
    """Pause / resume / stop a running pre-annotation (or batch-accept) task."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    try:
        task = svc.project.control_async_task(tenant, async_task_id, req.action)
    except ValueError as e:
        raise error(400, str(e))
    if not task:
        raise not_found()
    return task


@router.post("/projects/{id}/export-dataset")
async def export_dataset(id: str, req: ExportRequest, request: Request, svc: Services = Depends(get_svc)):
    """Export annotations to a new dataset version (COCO/PP-YOLOE)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    try:
        out = svc.project.export_as_dataset(
            tenant, id, req.name or "", req.description or "", req.format or "coco", user,
        )
    except ValueError as e:
        raise error(400, str(e))
    return {"exported": True, **out}


@router.get("/projects/{id}/image")
async def project_image(id: str, path: str, request: Request, svc: Services = Depends(get_svc)):
    """Proxy an image from the project's LS store (placeholder)."""
    raise error(404, "image not found")
