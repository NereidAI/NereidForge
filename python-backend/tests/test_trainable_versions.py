# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelService.list_trainable_versions.

The training form base-model selector must resolve a trainable (.pt)
checkpoint per registered version and flag ONNX-only versions as
non-trainable.
"""

import types
from datetime import datetime, timezone

from app.services.models import ModelService


class _FakeResult:
    """QueryResult double."""

    def __init__(self, items) -> None:
        self.items = items

    def all(self):
        return self.items


class _FakeDB:
    """Sequence-driven DB double (models call, then one versions call/model)."""

    def __init__(self, queues) -> None:
        self._queues = list(queues)

    def scalars(self, stmt):
        return _FakeResult(self._queues.pop(0))


class _FakeS3:
    """S3 double mapping anchor dir -> sibling file dicts."""

    def __init__(self, files: dict[str, list[dict]]) -> None:
        self.files = files

    def list_files_in_dir(self, uri: str) -> list[dict]:
        prefix = uri[: -len("probe")] if uri.endswith("probe") else uri
        return self.files.get(prefix, [])


def _model(name="m1", task_type="detection"):
    return types.SimpleNamespace(
        id=name + "-id", name=name, task_type=task_type, tenant_id="default",
    )


def _version(name, metadata=None, file_path="",
             created_at=datetime(2026, 9, 22, 14, 30, tzinfo=timezone.utc)):
    return types.SimpleNamespace(
        id="v-id", version=name, metadata_=metadata or {},
        file_path=file_path, size_bytes=10, status="ready",
        created_at=created_at,
    )


def _service(queues, s3_files=None):
    svc = ModelService.__new__(ModelService)
    svc.db = _FakeDB(queues)
    svc.s3 = _FakeS3(s3_files or {})
    return svc


def test_weights_uri_resolves_trainable_checkpoint():
    v = _version("20260922-143000", metadata={
        "weights_uri": "s3://nereidforge/models/job/best.pt",
    })
    m = _model()
    svc = _service([[m], [v]])

    out = svc.list_trainable_versions("default")

    assert len(out) == 1
    assert out[0]["trainable"] is True
    assert out[0]["checkpoint_uri"] == "s3://nereidforge/models/job/best.pt"


def test_onnx_only_version_is_non_trainable():
    v = _version(
        "20260922-143000",
        file_path="s3://nereidforge/models/job/out.onnx",
    )
    m = _model()
    anchor = "s3://nereidforge/models/job/"
    svc = _service(
        [[m], [v]],
        {anchor: [{"name": "out.onnx", "uri": anchor + "out.onnx"}]},
    )

    out = svc.list_trainable_versions("default")

    assert out[0]["trainable"] is False
    assert out[0]["checkpoint_uri"] == ""


def test_best_checkpoint_preferred_from_sibling_scan():
    v = _version("20260922-143000", metadata={
        "weights_uri": "s3://nereidforge/models/job/last.pt",
    })
    m = _model()
    anchor = "s3://nereidforge/models/job/"
    svc = _service(
        [[m], [v]],
        {anchor: [
            {"name": "best.pt", "uri": anchor + "best.pt"},
            {"name": "last.pt", "uri": anchor + "last.pt"},
        ]},
    )

    out = svc.list_trainable_versions("default")

    assert out[0]["checkpoint_uri"] == anchor + "best.pt"


def test_task_type_filter_is_passed_through():
    svc = _service([[], []])
    svc.list_trainable_versions("default", "detection")
