# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for annotation-project route error contracts."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.routers.projects import cvat_task_detail


def test_missing_cvat_task_raises_http_404() -> None:
    """A missing CVAT task must raise the documented 404, not NameError."""
    services = SimpleNamespace(
        project=SimpleNamespace(
            cvat_task_detail=lambda tenant_id, project_id, task_id: None,
        ),
    )
    request = SimpleNamespace(headers={"X-Tenant-Id": "default"})

    with pytest.raises(HTTPException) as caught:
        asyncio.run(cvat_task_detail("project-id", 42, request, services))

    assert caught.value.status_code == 404
    assert caught.value.detail == "Task not found"
