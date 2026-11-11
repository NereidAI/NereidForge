# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""API router for the S3 base-model pool.

Endpoints manage user-uploaded CV checkpoints and LLM model directories:
paginated listing (management page), multipart upload (cv files / llm zip or
multi-file), and deletion. Uploads run in the worker threadpool so the event
loop keeps answering probes while files stream to S3.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.helpers import error, get_svc, not_found
from app.core.util import paginate
from app.services import Services

router = APIRouter()
logger = logging.getLogger("nereidforge.api.basemodels")


@router.get("/base-model-pool")
async def list_pool(
    request: Request,
    kind: str = "", search: str = "", page: int = 1, page_size: int = 20,
    svc: Services = Depends(get_svc),
):
    """List pool rows (newest first), filtered by kind and/or name search."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    items, total = svc.base_model.list_paged(tenant, kind, search, page, page_size)
    return paginate(items, total, page, page_size)


@router.post("/base-model-pool")
async def upload_pool(
    request: Request,
    kind: str = Form(...),
    name: str = Form(""),
    framework: str = Form(""),
    task: str = Form(""),
    status: str = Form("develop"),
    description: str = Form(""),
    files: list[UploadFile] = File(...),
    svc: Services = Depends(get_svc),
):
    """Upload base models into the pool (multipart, one or more files).

    kind=cv: every file is stored by its own filename under ``cv/``; one row
    per file. kind=llm: a single ``.zip`` is extracted into ``llm/<name>/``
    (name defaults to the zip stem), while multiple files are stored under
    ``llm/<name>/`` preserving their relative paths. Framework/task are
    inferred when omitted; status and description are recorded as given.
    """
    from app.services.base_models import (
        FRAMEWORKS, LLM_TASKS, STATUSES, CV_TASKS, BaseModelError,
    )

    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    if not files:
        raise error(400, "at least one file is required")
    if status not in STATUSES:
        raise error(400, f"status must be one of: {', '.join(STATUSES)}")
    if framework and framework not in FRAMEWORKS:
        raise error(400, f"framework must be one of: {', '.join(FRAMEWORKS)}")
    if task:
        valid_tasks = (*CV_TASKS, *LLM_TASKS)
        if task not in valid_tasks:
            raise error(400, f"task must be one of: {', '.join(valid_tasks)}")

    def _persist():
        # Runs off the event loop: streams uploads straight to S3.
        if kind == "cv":
            out = [
                svc.base_model.upload_cv(
                    tenant, f.filename or "", f.file, user,
                    framework=framework, task=task, status=status,
                    description=description,
                )
                for f in files
            ]
            return {"items": out, "total": len(out)}
        if kind != "llm":
            raise error(400, "kind must be cv or llm")
        is_zip = len(files) == 1 and (files[0].filename or "").lower().endswith(".zip")
        if is_zip:
            dir_name = name or (files[0].filename or "")[: -len(".zip")]
            row = svc.base_model.upload_llm_zip(
                tenant, dir_name, files[0].file, user,
                framework=framework or "pytorch", task=task or "llm",
                status=status, description=description,
            )
        else:
            if not name:
                raise error(400, "name is required for multi-file llm uploads")
            pairs = [(f.filename or "", f.file) for f in files]
            row = svc.base_model.upload_llm_files(
                tenant, name, pairs, user,
                framework=framework or "pytorch", task=task or "llm",
                status=status, description=description,
            )
        return {"items": [row], "total": 1}

    try:
        return await run_in_threadpool(_persist)
    except BaseModelError as e:
        raise error(400, str(e))


@router.delete("/base-model-pool/{row_id}")
async def delete_pool(request: Request, row_id: str, svc: Services = Depends(get_svc)):
    """Delete one pool row plus its S3 file (cv) or directory (llm)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    deleted = await run_in_threadpool(svc.base_model.delete, tenant, row_id)
    if not deleted:
        raise not_found()
    return {"deleted": True}
