# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Model-version artifact preparation, archive safety, and capability tests."""

from __future__ import annotations

import io
import tarfile
import zipfile
from types import SimpleNamespace

import pytest
from starlette.datastructures import UploadFile

from app.clients.s3 import SeaweedFSClient
from app.services.model_format import artifact_capability_permitted, artifact_format_catalog
from app.services.model_packages import ModelPackageService
from app.services.model_upload import ArtifactUploadError, prepare_artifact_upload
from app.services.models import ModelService


def _upload(name: str, data: bytes = b"model") -> UploadFile:
    """Build an in-memory multipart file with the production UploadFile contract."""
    return UploadFile(file=io.BytesIO(data), filename=name)


def _zip(entries: dict[str, bytes]) -> bytes:
    """Build a ZIP archive without filesystem artifacts."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return output.getvalue()


def _tar_gz(entries: dict[str, bytes]) -> bytes:
    """Build a gzip tar archive with regular files only."""
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def _contents(prepared) -> dict[str, bytes]:
    """Read every prepared file through the same streaming API used by the router."""
    result = {}
    for artifact in prepared.files:
        with artifact.open() as stream:
            result[artifact.relative_path] = stream.read()
    return result


def test_catalog_distinguishes_executable_capabilities() -> None:
    """The public registry exposes formats without claiming unsupported workflows."""
    catalog = artifact_format_catalog()
    formats = {entry["id"]: entry for entry in catalog["formats"]}

    assert set(catalog["archive_suffixes"]) == {".zip", ".tar.gz", ".tgz"}
    assert "deploy" in formats["openvino-ir"]["capabilities"]
    assert "deploy" in formats["onnx"]["capabilities"]
    assert "deploy" not in formats["tensorrt-plan"]["capabilities"]
    assert "deploy" not in formats["ascend-om"]["capabilities"]
    assert "optimize" in formats["onnx"]["capabilities"]
    assert "package" in formats["onnx"]["capabilities"]
    assert "optimize" not in formats["pytorch-checkpoint"]["capabilities"]
    assert ".xml" in formats["openvino-ir"]["extensions"]
    assert ".bin" in formats["openvino-ir"]["extensions"]
    for format_id in ("tflite", "tensorrt-plan", "ascend-om", "gguf"):
        assert formats[format_id]["capabilities"] == ["download"]
    assert "trial" not in formats["openvino-ir"]["capabilities"]
    assert "optimize" not in formats["openvino-ir"]["capabilities"]


def test_capability_predicate_only_restrictions_new_metadata() -> None:
    """Legacy rows remain unchanged while a recorded list is fully authoritative."""
    assert artifact_capability_permitted({}, "deploy") is True
    assert artifact_capability_permitted({"artifact_capabilities": "legacy"}, "deploy") is True
    assert artifact_capability_permitted({"artifact_capabilities": []}, "download") is False
    assert artifact_capability_permitted({"artifact_capabilities": ["download"]}, "download") is True
    assert artifact_capability_permitted({"artifact_capabilities": ["download"]}, "trial") is False


def test_single_onnx_preserves_streaming_upload_contract() -> None:
    """A single portable model needs no temporary extraction or extra metadata."""
    with prepare_artifact_upload([_upload("model.onnx", b"onnx-bytes")]) as prepared:
        assert prepared.primary_path == "model.onnx"
        assert prepared.format_id == "onnx"
        assert prepared.layout == "single-file"
        assert prepared.size_bytes == len(b"onnx-bytes")
        assert _contents(prepared) == {"model.onnx": b"onnx-bytes"}


def test_multipart_openvino_ir_requires_and_finds_companion() -> None:
    """Separate XML and BIN uploads form one IR artifact with two stored files."""
    uploads = [_upload("model.xml", b"graph"), _upload("model.bin", b"weights")]
    with prepare_artifact_upload(uploads) as prepared:
        assert prepared.primary_path == "model.xml"
        assert prepared.format_id == "openvino-ir"
        assert prepared.layout == "multi-file"
        assert prepared.size_bytes == len(b"graphweights")
        assert _contents(prepared) == {"model.xml": b"graph", "model.bin": b"weights"}


def test_ir_without_same_basename_bin_is_rejected() -> None:
    """An XML graph alone is not a complete deployable OpenVINO artifact."""
    with pytest.raises(ArtifactUploadError, match="same-basename .bin"):
        with prepare_artifact_upload([_upload("model.xml")]):
            pass


def test_zip_strips_one_wrapper_and_preserves_nested_ir_layout() -> None:
    """A regular vendor archive is unwrapped and retains meaningful subdirectories."""
    archive = _zip({
        "bundle/repository/demo/1/model.xml": b"graph",
        "bundle/repository/demo/1/model.bin": b"weights",
        "bundle/labels.txt": b"face\n",
    })
    with prepare_artifact_upload([_upload("ir.zip", archive)]) as prepared:
        assert prepared.primary_path == "repository/demo/1/model.xml"
        assert prepared.format_id == "openvino-ir"
        assert prepared.layout == "zip"
        assert _contents(prepared) == {
            "repository/demo/1/model.xml": b"graph",
            "repository/demo/1/model.bin": b"weights",
            "labels.txt": b"face\n",
        }


def test_tar_gz_tensorrt_records_constraints_without_deploy_capability() -> None:
    """Compiled engine metadata is retained while registration remains non-deployable."""
    archive = _tar_gz({"engine/model.plan": b"plan", "engine/labels.txt": b"person\n"})
    metadata = {"runtime_requirements": {"tensorrt": "8.6", "gpu_compute_cap": "8.6"}}
    with prepare_artifact_upload([_upload("engine.tar.gz", archive)], metadata=metadata) as prepared:
        assert prepared.primary_path == "model.plan"
        assert prepared.format_id == "tensorrt-plan"
        assert prepared.layout == "tar.gz"
        assert "deploy" not in prepared.capabilities
        assert prepared.runtime_requirements["gpu_compute_cap"] == "8.6"


def test_single_ascend_om_is_registration_and_download_only() -> None:
    """Ascend artifacts can enter the registry before a CANN serving adapter exists."""
    with prepare_artifact_upload([_upload("model.om", b"om")]) as prepared:
        assert prepared.format_id == "ascend-om"
        assert "download" in prepared.capabilities
        assert "deploy" not in prepared.capabilities
        assert "convert" not in prepared.capabilities


def test_explicit_primary_file_resolves_multi_artifact_bundle() -> None:
    """Ambiguity requires an explicit existing primary path instead of filename guessing."""
    archive = _zip({"a.onnx": b"a", "b.onnx": b"b"})
    with pytest.raises(ArtifactUploadError, match="primary_file"):
        with prepare_artifact_upload([_upload("models.zip", archive)]):
            pass
    with prepare_artifact_upload(
        [_upload("models.zip", archive)], metadata={"primary_file": "b.onnx"}
    ) as prepared:
        assert prepared.primary_path == "b.onnx"
        assert prepared.format_id == "onnx"


@pytest.mark.parametrize("member", ["../model.onnx", "/tmp/model.onnx", "a/../../model.onnx"])
def test_zip_rejects_traversal_and_absolute_paths(member: str) -> None:
    """Archive members cannot escape the staging directory or become absolute keys."""
    with pytest.raises(ArtifactUploadError, match="unsafe archive path"):
        with prepare_artifact_upload([_upload("bad.zip", _zip({member: b"x"}))]):
            pass


def test_tar_rejects_symbolic_links() -> None:
    """Links are rejected rather than followed or materialized during extraction."""
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        link = tarfile.TarInfo("model.onnx")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)
    with pytest.raises(ArtifactUploadError, match="unsupported archive member"):
        with prepare_artifact_upload([_upload("bad.tar.gz", output.getvalue())]):
            pass


def test_archive_rejects_duplicates_and_enforces_expanded_limits() -> None:
    """Normalized collisions and decompression bombs fail before any S3 write."""
    duplicate = io.BytesIO()
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("model.onnx", b"a")
        archive.writestr("./model.onnx", b"b")
    with pytest.raises(ArtifactUploadError, match="duplicate"):
        with prepare_artifact_upload([_upload("duplicate.zip", duplicate.getvalue())]):
            pass

    with pytest.raises(ArtifactUploadError, match="expanded size"):
        with prepare_artifact_upload(
            [_upload("large.zip", _zip({"model.onnx": b"12345"}))], max_bytes=4
        ):
            pass


def test_archive_cannot_be_mixed_with_loose_files() -> None:
    """One request has one unambiguous container boundary."""
    with pytest.raises(ArtifactUploadError, match="cannot be combined"):
        with prepare_artifact_upload([
            _upload("models.zip", _zip({"model.onnx": b"x"})),
            _upload("labels.txt", b"label\n"),
        ]):
            pass


def test_unknown_or_incomplete_bundle_fails_explicitly() -> None:
    """Generic archives are not silently registered as ready executable models."""
    with pytest.raises(ArtifactUploadError, match="no supported primary artifact"):
        with prepare_artifact_upload([_upload("docs.zip", _zip({"README.txt": b"docs"}))]):
            pass
    with pytest.raises(ArtifactUploadError, match="unsupported model artifact"):
        with prepare_artifact_upload([_upload("weights.unknown")]):
            pass


class _BundleS3Client:
    """Provide prefix listings and object reads for recursive bundle tests."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.list_kwargs: list[dict] = []

    def list_objects_v2(self, **kwargs) -> dict:
        """Emulate the S3 prefix page used by the production paginator."""
        self.list_kwargs.append(kwargs)
        prefix = kwargs.get("Prefix", "")
        return {
            "Contents": [
                {"Key": key, "Size": len(data), "LastModified": "2026-11-06T00:00:00"}
                for key, data in sorted(self.objects.items())
                if key.startswith(prefix) and not key.endswith("/")
            ],
            "IsTruncated": False,
        }

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        """Return one object body through the boto3 response shape."""
        assert Bucket == "nereidforge"
        return {"Body": io.BytesIO(self.objects[Key])}


class _ScalarDB:
    """Resolve the one ORM-like version row queried by bundle access methods."""

    def __init__(self, row) -> None:
        self.row = row

    def scalar(self, _query):
        """Return the configured row regardless of the generated SQL expression."""
        return self.row


def _bundle_service() -> tuple[ModelService, SeaweedFSClient, dict]:
    """Build a real service/S3 path around an in-memory object store."""
    objects = {
        "models/demo/v1/repository/demo/1/model.onnx": b"onnx",
        "models/demo/v1/repository/demo/1/labels.txt": b"face\n",
        "models/demo/v1/README.txt": b"docs",
        "models/demo/v2/outside.onnx": b"outside",
    }
    s3 = SeaweedFSClient.__new__(SeaweedFSClient)
    s3.client = _BundleS3Client(objects)
    version = {
        "file_path": "s3://nereidforge/models/demo/v1/repository/demo/1/model.onnx",
        "metadata": {
            "bundle_uri": "s3://nereidforge/models/demo/v1/",
            "primary_file": "repository/demo/1/model.onnx",
            "artifact_capabilities": ["download", "trial"],
        },
    }
    row = SimpleNamespace(
        file_path=version["file_path"], metadata_=version["metadata"],
    )
    return ModelService(_ScalarDB(row), s3), s3, version


def test_bundle_listing_and_download_use_root_and_allow_nested_paths() -> None:
    """The complete archive remains browsable without permitting prefix escapes."""
    service, s3, _version = _bundle_service()
    files = service.list_version_files("default", "model-id", "v1")
    assert files is not None
    assert {item["name"] for item in files} == {
        "repository/demo/1/model.onnx", "repository/demo/1/labels.txt", "README.txt",
    }
    assert all("Delimiter" not in call for call in s3.client.list_kwargs)
    assert service.s3.count_files_in_dir(_version["metadata"]["bundle_uri"]) == 3

    output = io.BytesIO()
    uri = service.stream_version_file(
        "default", "model-id", "v1", "repository/demo/1/labels.txt", output,
    )
    assert uri == "s3://nereidforge/models/demo/v1/repository/demo/1/labels.txt"
    assert output.getvalue() == b"face\n"
    assert service.stream_version_file(
        "default", "model-id", "v1", "../v2/outside.onnx", io.BytesIO(),
    ) is None
    assert service.stream_version_file(
        "default", "model-id", "v1", "/repository/demo/1/model.onnx", io.BytesIO(),
    ) is None


def test_legacy_metadata_cannot_redirect_the_bundle_root() -> None:
    """A bundle_uri without backend-issued capabilities is not an authorization root."""
    service, _s3, version = _bundle_service()
    del version["metadata"]["artifact_capabilities"]
    files = service.list_version_files("default", "model-id", "v1")
    assert files is not None
    assert {item["name"] for item in files} == {"model.onnx", "labels.txt"}


def test_trial_resolves_an_exact_nested_artifact_from_the_live_bundle() -> None:
    """Nested trial selection follows the same live-list authorization as download."""
    service, _s3, version = _bundle_service()
    uri, is_pt = service._resolve_trial_artifact(
        version, artifact_path="repository/demo/1/model.onnx",
    )
    assert uri == version["file_path"]
    assert is_pt is False
    with pytest.raises(ValueError, match="invalid artifact path"):
        service._resolve_trial_artifact(version, artifact_path="../v2/outside.onnx")


@pytest.mark.parametrize("format_id", ["tflite", "tensorrt-plan", "ascend-om", "gguf", "openvino-ir"])
def test_trial_capability_is_checked_before_storage_or_ml_access(format_id: str) -> None:
    """An explicit path cannot bypass a version's missing trial capability."""
    service, s3, version = _bundle_service()
    version["metadata"]["artifact_format"] = format_id
    version["metadata"]["artifact_capabilities"] = ["download"]
    with pytest.raises(ValueError, match="does not support image trials"):
        service._resolve_trial_artifact(version, artifact_path="repository/demo/1/model.onnx")
    assert s3.client.list_kwargs == []


def test_package_capability_is_checked_before_request_or_task_creation() -> None:
    """Registration-only artifacts cannot create package requests or async tasks."""
    writes: list[str] = []
    additions: list[object] = []
    commits: list[bool] = []
    model = SimpleNamespace(id="model-id")
    version = SimpleNamespace(
        version="v1",
        metadata_={"artifact_format": "ascend-om", "artifact_capabilities": ["download"]},
    )
    rows = iter([model, version])
    db = SimpleNamespace(
        scalar=lambda _query: next(rows),
        add=lambda task: additions.append(task),
        commit=lambda: commits.append(True),
    )
    s3 = SimpleNamespace(put_object=lambda path, _data: writes.append(path))
    service = ModelPackageService(s3, SimpleNamespace(db=db))
    with pytest.raises(ValueError, match="does not support edge packaging"):
        service.enqueue("default", "demo", "v1", "fp16", "tester")
    assert writes == []
    assert additions == []
    assert commits == []


def test_resolve_direct_onnx_package_uses_actual_version_identity() -> None:
    """Uploaded ONNX uses its real NF IDs and no synthetic dataset/revision."""
    model = SimpleNamespace(id="m-sface", task_type="embedding")
    version = SimpleNamespace(
        id="mv-real-sface", version="2021dec",
        file_path="s3://nereidforge/models/sface/2021dec/model.onnx",
        metadata_={
            "artifact_format": "onnx", "family": "sface",
            "artifact_capabilities": ["download", "package"],
        },
    )
    s3 = SimpleNamespace(object_size=lambda uri: 38696353 if uri.endswith(".onnx") else 0)
    service = ModelPackageService(s3, SimpleNamespace(db=SimpleNamespace()))

    request, uris = service.resolve_request(
        "default", model, version, "fp32", "task-sface",
    )

    assert request["model"] == {
        "id": "m-sface", "version": "2021dec", "revision": "mv-real-sface",
        "task": "embedding",
    }
    assert request["source"]["onnx_uri"] == version.file_path
    assert request["source"]["family_hint"] == "sface"
    assert "dataset_version_uri" not in request["source"]
    assert request["hardware"]["profile"] == "onnx-structural"
    assert request["hardware"]["verify_devices"] == []
    assert uris["package_uri"].endswith("/nereid-model-fp32.tar.gz")


def test_ready_fp32_package_retains_revision_for_consumers() -> None:
    """FP32 packages are downloadable and expose identity without tar inspection."""
    records = [{
        "precision": "fp32", "status": "ready", "revision": "mv-real-sface",
        "package_uri": "s3://nereidforge/packages/m-sface/nereid-model-fp32.tar.gz",
        "package_sha256": "a" * 64,
    }]
    assert ModelPackageService.available_packages({"packages": records}) == records


def test_record_result_restores_complete_package_locator_and_revision() -> None:
    """Result consumption can rehydrate a complete record without a pending row."""
    result = {
        "schema": "nereidforge.package-result/1", "request_id": "task-sface",
        "precision": "FP32", "status": "succeeded",
        "package": {
            "uri": "s3://nereidforge/packages/m-sface/nereid-model-fp32.tar.gz",
            "bytes": 123, "sha256": "b" * 64,
        },
        "sidecar_uri": "s3://nereidforge/packages/m-sface/nereid-model-fp32.sidecar.json",
        "staging": {"uri": "s3://nereidforge/packages/m-sface/staging/fp32/"},
        "verification": {"passed": True, "numeric_status": "not_run", "devices": []},
    }
    updates = []
    service = ModelPackageService.__new__(ModelPackageService)
    service._read_json_uri = lambda _uri: result
    service._update_record = lambda *args, **kwargs: updates.append((args, kwargs))
    task = SimpleNamespace(
        id="task-sface", tenant_id="default",
        params={
            "model_id": "m-sface", "version": "2021dec", "precision": "fp32",
            "revision": "mv-real-sface", "hardware_profile": "onnx-structural",
            "request_uri": "s3://nereidforge/packages/m-sface/requests/task-sface.json",
            "result_uri": "s3://nereidforge/packages/m-sface/results/task-sface.result.json",
            "package_uri": result["package"]["uri"],
            "sidecar_uri": result["sidecar_uri"],
            "staging_uri": result["staging"]["uri"],
        },
    )

    record = service.record_result(task)

    assert record["status"] == "ready"
    assert record["revision"] == "mv-real-sface"
    assert record["package_uri"] == result["package"]["uri"]
    assert record["request_uri"] == task.params["request_uri"]
    assert record["result_uri"] == task.params["result_uri"]
    assert record["task_id"] == "task-sface"
    assert updates
