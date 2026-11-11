# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Multipart model-version routes persist validated bundles and enforce capabilities."""

from __future__ import annotations

import io
import json
import types
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.helpers import get_svc
from app.api.routers.models import router


class _MemS3:
    """Capture streamed objects without requiring a storage service."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_fileobj(self, filer: str, stream) -> None:
        """Store bytes at the exact filer key emitted by the route."""
        self.objects[filer] = stream.read()

    def object_size(self, filer: str) -> int:
        """Return a stored object's byte length."""
        return len(self.objects.get(filer, b""))


class _ClosingMemS3(_MemS3):
    """Emulate storage implementations that own and close an uploaded stream."""

    def upload_fileobj(self, filer: str, stream) -> None:
        """Capture bytes and close the source to verify context-manager cleanup."""
        try:
            super().upload_fileobj(filer, stream)
        finally:
            stream.close()


class _FailingClosingMemS3(_ClosingMemS3):
    """Close the source and fail to verify that cleanup preserves storage errors."""

    def upload_fileobj(self, filer: str, stream) -> None:
        """Raise the original storage failure after taking ownership of the stream."""
        super().upload_fileobj(filer, stream)
        raise RuntimeError("storage failed")


class _RouteModel:
    """Record version persistence while exposing the model route dependencies."""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.hub: list[tuple] = []
        self.version: dict | None = None

    def get_model(self, _tenant: str, _name: str) -> dict:
        """Resolve the test model."""
        return {"id": "model-id", "name": "demo", "framework": "pytorch"}

    def get_version(self, *_args) -> dict | None:
        """Return the configured version for deployment capability checks."""
        return self.version

    def create_version(self, *args) -> dict:
        """Capture the canonical primary URI, size, and enriched metadata."""
        keys = (
            "tenant", "model_id", "version", "description", "file_path",
            "size_bytes", "metadata", "created_by",
        )
        call = dict(zip(keys, args))
        self.created.append(call)
        return {
            "id": "mv-1", "model_id": "model-id", "version": call["version"],
            "description": call["description"], "file_path": call["file_path"],
            "size_bytes": call["size_bytes"], "metadata": call["metadata"],
            "status": "ready", "artifacts": [], "artifacts_count": 0,
            "created_by": "tester", "created_at": "2026-11-06T00:00:00Z",
        }

    def register_to_hub(self, *args) -> bool:
        """Record best-effort hub registration."""
        self.hub.append(args)
        return True


def _client(model: _RouteModel, s3: _MemS3) -> TestClient:
    """Mount the real model router with only external services doubled."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_svc] = lambda: types.SimpleNamespace(model=model, s3=s3)
    return TestClient(app)


def _zip(entries: dict[str, bytes]) -> bytes:
    """Build an archive payload for multipart tests."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return output.getvalue()


def test_artifact_catalog_is_public_and_typed() -> None:
    """The frontend can discover accepted containers and capabilities from the backend."""
    client = _client(_RouteModel(), _MemS3())
    response = client.get("/api/v1/models/artifact-formats")
    assert response.status_code == 200
    payload = response.json()
    assert ".tar.gz" in payload["archive_suffixes"]
    assert any(item["id"] == "openvino-ir" for item in payload["formats"])


def test_multipart_ir_bundle_streams_every_member_and_enriches_metadata() -> None:
    """Two uploaded IR companions become one canonical version artifact set."""
    model = _RouteModel()
    s3 = _MemS3()
    client = _client(model, s3)
    response = client.post(
        "/api/v1/models/demo/versions",
        data={"version": "v1", "description": "IR", "metadata": json.dumps({"input_shape": [1, 3, 640, 640]})},
        files=[
            ("file", ("model.xml", b"graph", "application/octet-stream")),
            ("file", ("model.bin", b"weights", "application/octet-stream")),
        ],
        headers={"X-Tenant-Id": "default", "X-Username": "tester"},
    )
    assert response.status_code == 200, response.text
    assert s3.objects == {
        "/buckets/nereidforge/models/demo/v1/model.xml": b"graph",
        "/buckets/nereidforge/models/demo/v1/model.bin": b"weights",
    }
    created = model.created[0]
    assert created["file_path"] == "s3://nereidforge/models/demo/v1/model.xml"
    assert created["size_bytes"] == len(b"graphweights")
    assert created["metadata"]["artifact_format"] == "openvino-ir"
    assert created["metadata"]["artifact_layout"] == "multi-file"
    assert created["metadata"]["bundle_uri"] == "s3://nereidforge/models/demo/v1/"
    assert created["metadata"]["primary_file"] == "model.xml"
    assert created["metadata"]["input_shape"] == [1, 3, 640, 640]
    assert model.hub


def test_multipart_ir_bundle_allows_storage_to_close_each_source() -> None:
    """A closed multipart source is a valid post-upload state, not a route failure."""
    model = _RouteModel()
    s3 = _ClosingMemS3()
    client = _client(model, s3)
    response = client.post(
        "/api/v1/models/demo/versions",
        data={"version": "v1", "metadata": "{}"},
        files=[
            ("file", ("model.xml", b"graph", "application/octet-stream")),
            ("file", ("model.bin", b"weights", "application/octet-stream")),
        ],
    )
    assert response.status_code == 200, response.text
    assert len(s3.objects) == 2
    assert len(model.created) == 1


def test_closed_source_cleanup_does_not_mask_storage_failure() -> None:
    """The original storage exception remains visible when its source is closed."""
    client = _client(_RouteModel(), _FailingClosingMemS3())
    with pytest.raises(RuntimeError, match="storage failed"):
        client.post(
            "/api/v1/models/demo/versions",
            data={"version": "v1", "metadata": "{}"},
            files=[("file", ("model.onnx", b"onnx", "application/octet-stream"))],
        )


def test_zip_upload_extracts_before_registration() -> None:
    """A ZIP container is not registered as the primary model itself."""
    model = _RouteModel()
    s3 = _MemS3()
    client = _client(model, s3)
    archive = _zip({"wrapper/model.onnx": b"onnx", "wrapper/labels.txt": b"face\n"})
    response = client.post(
        "/api/v1/models/demo/versions",
        data={"version": "v2", "metadata": "{}"},
        files=[("file", ("model.zip", archive, "application/zip"))],
    )
    assert response.status_code == 200, response.text
    assert set(s3.objects) == {
        "/buckets/nereidforge/models/demo/v2/model.onnx",
        "/buckets/nereidforge/models/demo/v2/labels.txt",
    }
    assert model.created[0]["metadata"]["artifact_layout"] == "zip"


def test_metadata_only_registration_cannot_forge_validated_bundle_claims() -> None:
    """The legacy no-file path retains custom metadata but not reserved upload proof."""
    model = _RouteModel()
    client = _client(model, _MemS3())
    response = client.post(
        "/api/v1/models/demo/versions",
        data={
            "version": "legacy",
            "metadata": json.dumps({
                "custom": "kept",
                "artifact_format": "onnx",
                "artifact_capabilities": ["download", "deploy"],
                "bundle_uri": "s3://nereidforge/other/tenant/",
                "primary_file": "secret.onnx",
            }),
        },
    )
    assert response.status_code == 200, response.text
    metadata = model.created[0]["metadata"]
    assert metadata["custom"] == "kept"
    assert "artifact_format" not in metadata
    assert "artifact_capabilities" not in metadata
    assert "bundle_uri" not in metadata
    assert "primary_file" not in metadata


def test_invalid_archive_writes_no_objects_and_no_version() -> None:
    """Validation finishes before the first externally visible storage write."""
    model = _RouteModel()
    s3 = _MemS3()
    client = _client(model, s3)
    response = client.post(
        "/api/v1/models/demo/versions",
        data={"version": "bad", "metadata": "{}"},
        files=[("file", ("bad.zip", _zip({"../model.onnx": b"x"}), "application/zip"))],
    )
    assert response.status_code == 400
    assert "unsafe archive path" in response.json()["detail"]
    assert s3.objects == {}
    assert model.created == []


def test_package_listing_exposes_fp32_revision_and_digest() -> None:
    """NL can obtain the complete package identity before downloading bytes."""
    model = _RouteModel()
    model.version = {
        "metadata": {
            "packages": [{
                "precision": "fp32", "status": "ready", "revision": "mv-real",
                "package_uri": "s3://nereidforge/packages/m/nereid-model-fp32.tar.gz",
                "package_sha256": "c" * 64, "package_bytes": 123,
                "devices": [], "quality_gate_passed": None,
            }],
        },
    }
    client = _client(model, _MemS3())
    response = client.get("/api/v1/models/demo/versions/2021dec/packages")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "model_id": "model-id", "version": "2021dec",
        "packages": [{
            "precision": "FP32", "status": "ready", "revision": "mv-real",
            "package_uri": "s3://nereidforge/packages/m/nereid-model-fp32.tar.gz",
            "package_sha256": "c" * 64, "package_bytes": 123,
            "detection_count": None, "devices": [], "quality_gate_passed": None,
            "map50_baseline": None, "map50_candidate": None,
        }],
    }


def test_package_listing_has_typed_openapi_identity_contract() -> None:
    """Generated FastAPI schema retains FP32 revision and digest discovery fields."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    schema = app.openapi()
    operation = schema["paths"][
        "/api/v1/models/{model_name}/versions/{version}/packages"
    ]["get"]
    response_ref = operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert response_ref.endswith("/EdgePackageListResponse")
    properties = schema["components"]["schemas"]["EdgePackageSummaryResponse"]["properties"]
    assert "revision" in properties
    assert "package_sha256" in properties


def test_registration_only_artifact_cannot_enter_deploy_task() -> None:
    """New OM metadata rejects deployment before any cluster or async-task access."""
    model = _RouteModel()
    model.version = {
        "file_path": "s3://nereidforge/models/demo/v3/model.om",
        "metadata": {
            "artifact_format": "ascend-om",
            "artifact_capabilities": ["download"],
        },
    }
    client = _client(model, _MemS3())
    response = client.post(
        "/api/v1/models/demo/versions/v3/deploy",
        json={"target": "huawei", "runtime_type": "kserve-cann"},
    )
    assert response.status_code == 400
    assert "registration/download only" in response.json()["detail"]
