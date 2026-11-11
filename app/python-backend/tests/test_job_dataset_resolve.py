# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for JobService training/eval dataset URI resolution.

The resolver must recognize every canonical layout the dataset importer can
produce (flat and split YOLO, flat and split ImageFolder) and only fall back
to the shared legacy yolo-train prefix when the version is unrecognizable.
"""

import types

from app.services.jobs import JobService


class _FakeS3:
    """S3 double keyed on exact object keys (including directory markers)."""

    def __init__(self, existing: set[str]):
        self.existing = existing

    @staticmethod
    def tenant_uri(uri: str, tenant_id: str) -> str:
        return uri

    @staticmethod
    def s3_uri_to_filer_path(uri: str) -> str:
        return "/buckets/" + uri[len("s3://"):].rstrip("/")

    def read_object(self, path: str):
        if path in self.existing:
            return b"x"
        raise FileNotFoundError(path)


class _FakeDatasets:
    """DatasetService double returning one fixed version."""

    def __init__(self, uri: str):
        self.uri = uri

    @staticmethod
    def get_dataset(tenant_id: str, ds_id: str):
        return {"version": 1}

    def get_dataset_version(self, tenant_id: str, ds_id: str, ver: int):
        return types.SimpleNamespace(uri=self.uri)


def _service(existing: set[str], uri: str = "s3://nereidforge/datasets/ds/v1") -> JobService:
    svc = JobService.__new__(JobService)
    svc.s3 = _FakeS3(existing)
    svc.dataset = _FakeDatasets(uri)

    class _DB:
        @staticmethod
        def scalar(stmt):
            return types.SimpleNamespace(uri=uri)

    svc.db = _DB()
    return svc


_BASE = "/buckets/nereidforge/datasets/ds/v1"


def test_resolve_split_yolo_returns_version_root():
    svc = _service({
        f"{_BASE}/data.yaml",
        f"{_BASE}/train/images/",
        f"{_BASE}/train/labels/",
        f"{_BASE}/val/images/",
        f"{_BASE}/val/labels/",
    })
    uri = svc.resolve_training_data_uri(
        "default", {"dataset_id": "ds", "dataset_version": "1"},
        types.SimpleNamespace(id="j1"))
    assert uri == "s3://nereidforge/datasets/ds/v1/"


def test_resolve_flat_yolo_returns_version_root():
    svc = _service({
        f"{_BASE}/data.yaml", f"{_BASE}/images/", f"{_BASE}/labels/"})
    uri = svc.resolve_training_data_uri(
        "default", {"dataset_id": "ds", "dataset_version": "1"},
        types.SimpleNamespace(id="j1"))
    assert uri == "s3://nereidforge/datasets/ds/v1/"


def test_resolve_split_imagenet_returns_version_root():
    svc = _service({
        f"{_BASE}/class_names.json", f"{_BASE}/train/",
        f"{_BASE}/train/cat/a.jpg",
        f"{_BASE}/val/cat/b.jpg"})
    uri = svc.resolve_training_data_uri(
        "default", {"dataset_id": "ds", "dataset_version": "1"},
        types.SimpleNamespace(id="j1"))
    assert uri == "s3://nereidforge/datasets/ds/v1/"


def test_resolve_flat_imagenet_returns_version_root():
    svc = _service({
        f"{_BASE}/class_names.json", f"{_BASE}/images/",
        f"{_BASE}/images/cat/a.jpg"})
    uri = svc.resolve_training_data_uri(
        "default", {"dataset_id": "ds", "dataset_version": "1"},
        types.SimpleNamespace(id="j1"))
    assert uri == "s3://nereidforge/datasets/ds/v1/"


def test_resolve_unknown_version_falls_back_to_legacy_prefix():
    svc = _service({f"{_BASE}/readme.txt"})
    uri = svc.resolve_training_data_uri(
        "default", {"dataset_id": "ds", "dataset_version": "1"},
        types.SimpleNamespace(id="j1"))
    assert uri == "s3://nereidforge/yolo-train/"


def test_resolve_eval_split_yolo_does_not_redirect_to_annotated_export():
    svc = _service({
        f"{_BASE}/data.yaml",
        f"{_BASE}/train/images/",
        f"{_BASE}/val/images/",
        f"{_BASE}/annotated/old/coco/annotations.json",
    })
    uri = svc._resolve_eval_dataset_uri(
        "default", {"dataset_id": "ds", "dataset_version": "1"})
    assert uri == "s3://nereidforge/datasets/ds/v1/"
