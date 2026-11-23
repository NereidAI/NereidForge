# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for scalable Label Studio task-list loading.

The studio requires complete annotation and prediction arrays, but fetching one
task detail per list item makes a normal 20-task page exceed the UI deadline.
The contract here preserves full results through one LS list request while
keeping unrelated list consumers lightweight.
"""

import types
from typing import get_type_hints

from app.clients.labelstudio import LabelStudioClient
from app.services.project import ProjectService


class _FakeDB:
    """_FakeDB returns the linked annotation project for the service lookup."""

    def __init__(self, project) -> None:
        self.project = project

    def scalar(self, statement):
        """scalar ignores the SQL expression and returns the configured project."""
        return self.project


class _FakeLS:
    """Serve one full-list page and reject the former per-task detail fan-out."""

    def __init__(self) -> None:
        self.include_results = None
        self.detail_calls = 0

    def list_tasks(self, project_id, page=1, page_size=50, include_results=False):
        """Return embedded results that mirror LS fields=all serialization."""
        self.include_results = include_results
        return {
            "total": 1,
            "tasks": [{
                "id": 7,
                "inner_id": 3,
                "data": {"image": "s3://nereidforge/datasets/example/image.jpg"},
                "is_labeled": True,
                "total_annotations": 1,
                "total_predictions": 1,
                "annotations": [{
                    "id": 11,
                    "result": [{"type": "rectanglelabels"}],
                    "created_at": "2026-10-30T00:00:00Z",
                }],
                "predictions": [{
                    "id": 12,
                    "result": [{"type": "rectanglelabels"}],
                    "score": 0.75,
                    "model_version": "yolo",
                }],
                "created_at": "2026-10-29T00:00:00Z",
                "updated_at": "2026-10-30T00:00:00Z",
            }],
        }

    def get_task(self, task_id):
        """Fail if the service reintroduces one detail request per list item."""
        self.detail_calls += 1
        raise AssertionError("studio list must not fetch individual task details")


def test_list_tasks_only_embeds_results_on_explicit_request():
    """Keep lightweight callers unchanged while exposing LS fields=all explicitly."""
    client = LabelStudioClient.__new__(LabelStudioClient)
    calls = []

    def fake_get(path, params=None):
        calls.append((path, params))
        return {"tasks": [], "total": 0}

    client._get = fake_get  # type: ignore[assignment]

    client.list_tasks(1, 2, 20)
    client.list_tasks(1, 2, 20, include_results=True)

    assert calls == [
        ("/api/tasks/", {"project": 1, "page": 2, "page_size": 20}),
        ("/api/tasks/", {"project": 1, "page": 2, "page_size": 20, "fields": "all"}),
    ]


def test_ls_tasks_uses_embedded_results_without_detail_requests():
    """Return complete studio data from one list call without the N+1 fallback."""
    project = types.SimpleNamespace(
        id="prj-1", tenant_id="default", external_id="1",
    )
    service = ProjectService.__new__(ProjectService)
    service.db = _FakeDB(project)
    service.ls = _FakeLS()

    ls_id, items, total = service.ls_tasks("default", project.id, 1, 20)

    assert ls_id == 1
    assert total == 1
    assert service.ls.include_results is True
    assert service.ls.detail_calls == 0
    assert items[0]["annotations"] == [{
        "id": 11,
        "result": [{"type": "rectanglelabels"}],
        "created_at": "2026-10-30T00:00:00Z",
    }]
    assert items[0]["predictions"] == [{
        "id": 12,
        "result": [{"type": "rectanglelabels"}],
        "score": 0.75,
        "model_version": "yolo",
    }]
    assert items[0]["image_url"] == "s3://nereidforge/datasets/example/image.jpg"


def test_project_service_type_hints_resolve_label_studio_client() -> None:
    """Dependency introspection must resolve the constructor's LS annotation."""
    hints = get_type_hints(ProjectService.__init__)

    assert hints["ls"] is LabelStudioClient
