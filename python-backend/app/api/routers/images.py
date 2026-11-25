# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Image proxy routes (auth-exempt so browser <img> tags can load images)."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from app.services import Services
from app.api.helpers import get_svc, error

router = APIRouter()


def _content_type(name: str) -> str:
    """Guess the content type from a file extension."""
    ext = os.path.splitext(name)[1].lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".flv": "video/x-flv",
        ".wmv": "video/x-ms-wmv",
        ".m4v": "video/x-m4v",
    }.get(ext, "application/octet-stream")


@router.get("/s3-object")
async def s3_object(uri: str, svc: Services = Depends(get_svc)):
    """Proxy an S3 object as HTTP bytes (browser-reachable, no cluster IP exposure)."""
    if not uri or not uri.startswith("s3://"):
        raise error(400, "uri is required (s3://...)")
    filer = svc.s3.s3_uri_to_filer_path(uri)
    if not filer:
        raise error(400, "invalid s3 uri")
    try:
        data = svc.s3.read_object(filer)
    except Exception:
        raise error(404, "object not found")
    return Response(
        content=data,
        media_type=_content_type(uri),
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/datasets/{dataset_id}/image")
async def dataset_image(dataset_id: str, path: str, tenant: str = "default", svc: Services = Depends(get_svc)):
    """Proxy an image from a dataset version (browser-reachable, auth-exempt)."""
    print(f"DEBUG: dataset_id={dataset_id}, path={path}, tenant={tenant}")
    ds = svc.dataset.get_dataset(tenant, dataset_id)
    print(f"DEBUG: ds={ds}")
    if not ds:
        raise error(404, "dataset not found")
    ver = svc.dataset.get_dataset_version(tenant, dataset_id, ds["version"])
    print(f"DEBUG: ver={ver}")
    if not ver or not ver.uri:
        raise error(404, "dataset version not found")
    filer_base = svc.s3.s3_uri_to_filer_path(ver.uri)
    print(f"filer_base={filer_base}")
    if not filer_base:
        raise error(404, "invalid version uri")
    # Path traversal protection.
    safe_path = os.path.normpath("/" + path).lstrip("/")
    filer = f"{filer_base}/{safe_path}"
    print(f"filer={filer}")
    try:
        data = svc.s3.read_object(filer)
    except Exception as e:
        print(f"read_object error: {e}")
        raise error(404, "image not found")
    return Response(
        content=data,
        media_type=_content_type(path),
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
        },
    )
