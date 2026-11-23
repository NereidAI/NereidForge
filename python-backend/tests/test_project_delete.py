# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for annotation project deletion semantics.

Exported dataset versions are independent data: deletion must detach their
provenance link (source_project_id -> NULL) rather than block forever, while
an active training job consuming such a version still blocks the deletion.
"""

import types

import pytest

from app.services.project import ProjectService


class _FakeDB:
    """Sequence-driven DB double for ProjectService.delete queries."""

    def __init__(self, project, async_task=None, jobs=None, exported=None,
                 job_version=None) -> None:
        self._project = project
        self._async_task = async_task
        self._jobs = jobs or []
        self._exported = exported or []
        self._job_version = job_version
        self._scalar_calls = 0
        self.deleted = None

    def scalar(self, stmt):
        """Return values in the order ProjectService.delete queries them."""
        self._scalar_calls += 1
        if self._scalar_calls == 1:
            return self._project
        if self._scalar_calls == 2:
            return self._async_task
        return self._job_version

    def scalars(self, stmt):
        """First call lists active jobs, second lists exported versions."""

        class _Result:
            def __init__(self, items) -> None:
                self.items = items

            def all(self):
                return self.items

        self._scalars_calls = getattr(self, "_scalars_calls", 0) + 1
        if self._scalars_calls == 1:
            return _Result(self._jobs)
        return _Result(self._exported)

    def delete(self, obj) -> None:
        self.deleted = obj

    def commit(self) -> None:
        pass


def _service(db: _FakeDB) -> ProjectService:
    """Build a ProjectService double without external clients."""
    svc = ProjectService.__new__(ProjectService)
    svc.db = db
    svc.ls = types.SimpleNamespace(enabled=lambda: False)
    svc.cvat = None
    return svc


def test_delete_detaches_exported_version_and_succeeds():
    """Exported versions keep their data but lose the project link."""
    project = types.SimpleNamespace(
        id="prj1", tenant_id="default", external_id=None, task_type="detection"
    )
    exported = types.SimpleNamespace(id="dv1", source_project_id="prj1")
    db = _FakeDB(project, exported=[exported])

    assert _service(db).delete("default", "prj1") is True

    assert exported.source_project_id is None
    assert db.deleted is project


def test_delete_blocked_by_active_training_job():
    """An active job using an exported version keeps the project alive."""
    project = types.SimpleNamespace(
        id="prj1", tenant_id="default", external_id=None, task_type="detection"
    )
    job = types.SimpleNamespace(
        id="job1",
        params={"dataset_id": "ds", "dataset_version": "1"},
    )
    version = types.SimpleNamespace(id="dv1")
    db = _FakeDB(project, jobs=[job], job_version=version)

    with pytest.raises(ValueError, match="used by training job job1"):
        _service(db).delete("default", "prj1")

    assert db.deleted is None
