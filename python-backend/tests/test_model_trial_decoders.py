# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Focused behavior tests for model trial decoder inspection and persistence."""

from __future__ import annotations

import json
import types
from copy import deepcopy
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from app.api.helpers import get_svc
from app.api.routers.models import router
from app.core.config import config
from app.services.models import ModelService, ModelTrialError


ONNX_URI = "s3://nereidforge/models/demo/v1/model.onnx"
ALT_ONNX_URI = "s3://nereidforge/models/demo/v1/custom.onnx"
PT_URI = "s3://nereidforge/models/demo/v1/best.pt"
OTHER_URI = "s3://nereidforge/models/demo/v1/other.onnx"


def _version(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a serialized model version with an ONNX-first artifact choice."""
    if metadata is None:
        metadata = {"onnx_uri": ONNX_URI, "weights_uri": PT_URI}
    return {"file_path": ONNX_URI, "metadata": metadata}


def _inspection(artifact: str = ONNX_URI) -> dict[str, Any]:
    """Build the ML Backend inspection contract with representative tensors."""
    return {
        "artifact": artifact,
        "inputs": [{
            "name": "images", "shape": [1, 3, "height", "width"],
            "dtype": "float32", "backend_detail": "preserved",
        }],
        "outputs": [{"name": "logits", "shape": [1, 2], "dtype": "float32"}],
        "decoders": [{
            "id": "classification", "label": "Classification",
            "task": "classification", "matched": True, "score": 0.9,
        }],
        "auto_decoder": {
            "id": "classification", "task": "classification", "reason": "logits",
        },
        "mapping_required": False,
        "names": {"0": "cat", "1": "dog"},
    }


def _trial_result() -> dict[str, Any]:
    """Build a normalized trial response including a task-specific extra field."""
    return {
        "task": "classification",
        "kind": "classification",
        "width": 640,
        "height": 480,
        "names": {"0": "cat", "1": "dog"},
        "boxes": [],
        "probs": {"class_index": 0, "label": "cat", "score": 0.9, "top5": []},
        "time_ms": 1.5,
        "decoder": {"mode": "auto", "id": "classification", "task": "classification"},
        "task_specific": "preserved",
    }


class _FakeS3:
    """Record artifact-prefix lookups and return only explicit live siblings."""

    def __init__(self, files: list[dict[str, str]] | None = None) -> None:
        self.files = files or []
        self.anchors: list[str] = []

    def list_files_in_dir(self, anchor: str) -> list[dict[str, str]]:
        """Return the configured siblings for a validated explicit selection."""
        self.anchors.append(anchor)
        return self.files


def _service(
    version: dict[str, Any], files: list[dict[str, str]] | None = None
) -> ModelService:
    """Build a lightweight service with version lookup and persistence doubled."""
    service = ModelService.__new__(ModelService)
    service.s3 = _FakeS3(files)
    service.get_version = lambda *_args: version
    service.saved_calls = []
    service.save_trial_decoder = lambda *args: service.saved_calls.append(args)
    return service


def _mock_post(monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response]) -> list[dict]:
    """Mock sequential ML responses and capture URL, JSON, and timeout arguments."""
    calls: list[dict] = []
    remaining = iter(responses)

    def fake_post(url: str, json: dict, timeout: httpx.Timeout) -> httpx.Response:
        calls.append({"url": url, "json": json, "timeout": timeout})
        return next(remaining)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(config.labelstudio, "ml_backend_url", "http://ml-backend/")
    return calls


def _response(status: int, payload: dict[str, Any]) -> httpx.Response:
    """Build a real httpx response so JSON and text behavior match production."""
    return httpx.Response(
        status, json=payload, request=httpx.Request("POST", "http://ml-backend/try")
    )


class _RouteModel:
    """Service double exposing only the model methods used by trial routes."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.trial_calls: list[tuple[tuple, dict]] = []
        self.inspect_calls: list[tuple[tuple, dict]] = []

    def get_model(self, _tenant: str, _name: str) -> dict[str, str]:
        """Resolve the public model name to its internal identifier."""
        return {"id": "model-id", "name": "demo"}

    def try_predict(self, *args, **kwargs) -> dict[str, Any]:
        """Capture trial arguments and return a normalized result or test error."""
        self.trial_calls.append((args, kwargs))
        if self.error:
            raise self.error
        return _trial_result()

    def inspect_trial(self, *args, **kwargs) -> dict[str, Any]:
        """Return an inspection with only the explicitly supplied saved fields."""
        self.inspect_calls.append((args, kwargs))
        return {
            **_inspection(),
            "saved_decoder": {"mode": "auto"},
            "input_shape": [3, 224, 224],
        }


def _client(model: _RouteModel) -> TestClient:
    """Mount the real model router with a lightweight service-container double."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_svc] = lambda: types.SimpleNamespace(model=model)
    return TestClient(app)


@pytest.mark.parametrize("format_id", ["tflite", "tensorrt-plan", "ascend-om", "gguf", "openvino-ir"])
def test_trial_and_inspection_reject_missing_capability_before_s3_or_ml(
    monkeypatch, format_id,
):
    """Both public workflows share one early gate; explicit paths cannot bypass it."""
    artifact = f"s3://nereidforge/models/demo/v1/model.{format_id}"
    version = {
        "file_path": artifact,
        "metadata": {
            "artifact_format": format_id,
            "artifact_capabilities": ["download"],
        },
    }
    service = _service(version, [{"name": f"model.{format_id}", "uri": artifact}])
    monkeypatch.setattr(
        httpx, "post", lambda *_args, **_kwargs: pytest.fail("ML must not be called"),
    )

    with pytest.raises(ValueError, match="does not support image trials"):
        service.inspect_trial("tenant", "model-id", "v1", artifact_path=artifact.rsplit("/", 1)[-1])
    with pytest.raises(ValueError, match="does not support image trials"):
        service.try_predict(
            "tenant", "model-id", "v1", "aW1hZ2U=",
            artifact_path=artifact.rsplit("/", 1)[-1],
        )
    assert service.s3.anchors == []
    assert service.saved_calls == []


def test_schema_uses_selected_artifact_and_returns_saved_config_and_input_shape(monkeypatch):
    """Schema inspection resolves a live sibling and looks up config by its URI."""
    saved = {"mode": "builtin", "decoder_id": "custom-onnx"}
    version = _version({
        "onnx_uri": ONNX_URI,
        "weights_uri": PT_URI,
        "source": "uploaded",
        "architecture": "resnet18",
        "input_shape": [3, 224, 224],
        "trial_decoders": {ALT_ONNX_URI: saved, ONNX_URI: {"mode": "auto"}},
    })
    service = _service(version, [{"name": "custom.onnx", "uri": ALT_ONNX_URI}])
    calls = _mock_post(monkeypatch, [_response(200, _inspection(ALT_ONNX_URI))])

    result = service.inspect_trial(
        "tenant", "model-id", "v1", artifact_path="custom.onnx"
    )

    assert calls[0]["url"] == "http://ml-backend/try/inspect"
    assert calls[0]["json"] == {
        "model_path": ALT_ONNX_URI,
        "source": "uploaded",
        "architecture": "resnet18",
        "input_shape": [3, 224, 224],
    }
    assert calls[0]["timeout"].read == 180.0
    assert service.s3.anchors == [ONNX_URI]
    assert result["artifact"] == ALT_ONNX_URI
    assert result["saved_decoder"] == saved
    assert result["saved_decoder"] is not saved
    assert result["input_shape"] == [3, 224, 224]


def test_schema_defaults_source_to_auto_and_keeps_canonical_artifact(monkeypatch):
    """Missing family metadata stays neutral and ML-local paths never replace the S3 identity."""
    service = _service(_version({"onnx_uri": ONNX_URI}))
    calls = _mock_post(monkeypatch, [_response(200, _inspection("/tmp/downloaded-model.onnx"))])

    result = service.inspect_trial("tenant", "model-id", "v1")

    assert calls[0]["json"]["source"] == "auto"
    assert result["artifact"] == ONNX_URI


def test_trial_forwards_explicit_decoder_top_k_and_upload_hints(monkeypatch):
    """An explicit decoder and top_k reach ML with all existing inference options."""
    version = _version({
        "onnx_uri": ONNX_URI,
        "weights_uri": PT_URI,
        "source": "uploaded",
        "architecture": "custom-net",
        "input_shape": [3, 320, 320],
    })
    service = _service(version)
    calls = _mock_post(monkeypatch, [_response(200, _trial_result())])
    decoder = {
        "mode": "custom",
        "task": "classification",
        "input": {"tensor": "images", "width": 320, "height": 320},
        "output": {"output": "logits", "class_axis": 1},
    }

    result = service.try_predict(
        "tenant", "model-id", "v1", "aW1hZ2U=",
        conf=0.4, iou=0.6, class_index=2, decoder_config=decoder, top_k=7,
    )

    assert result["task_specific"] == "preserved"
    assert calls[0]["url"] == "http://ml-backend/try/predict"
    assert calls[0]["json"] == {
        "model_path": ONNX_URI,
        "source": "uploaded",
        "architecture": "custom-net",
        "input_shape": [3, 320, 320],
        "image_base64": "aW1hZ2U=",
        "conf": 0.4,
        "iou": 0.6,
        "cam": False,
        "decoder": decoder,
        "top_k": 7,
        "class_index": 2,
    }
    assert service.saved_calls == [
        ("tenant", "model-id", "v1", ONNX_URI, decoder)
    ]
    assert service.saved_calls[0][-1] is not decoder


@pytest.mark.parametrize(
    ("saved", "expected"),
    [
        ({"mode": "builtin", "decoder_id": "saved-decoder"},
         {"mode": "builtin", "decoder_id": "saved-decoder"}),
        (None, {"mode": "auto"}),
    ],
)
def test_trial_reuses_saved_decoder_or_defaults_to_auto(monkeypatch, saved, expected):
    """Omitting decoder_config reuses only the selected artifact's saved entry."""
    decoders = {OTHER_URI: {"mode": "builtin", "decoder_id": "wrong-artifact"}}
    if saved is not None:
        decoders[ONNX_URI] = saved
    service = _service(_version({
        "onnx_uri": ONNX_URI,
        "weights_uri": PT_URI,
        "trial_decoders": decoders,
    }))
    calls = _mock_post(monkeypatch, [_response(200, _trial_result())])

    service.try_predict("tenant", "model-id", "v1", "aW1hZ2U=")

    assert calls[0]["json"]["decoder"] == expected
    assert calls[0]["json"]["top_k"] == 5
    assert service.saved_calls[0][-1] == expected


class _FakeDB:
    """Capture the locked version query and commits for persistence testing."""

    def __init__(self, row: Any) -> None:
        self.row = row
        self.statement = None
        self.commits = 0

    def scalar(self, statement):
        """Return the tenant-scoped row selected by the service."""
        self.statement = statement
        return self.row

    def commit(self) -> None:
        """Record the successful metadata transaction."""
        self.commits += 1


def test_successful_selection_persistence_preserves_siblings_and_unrelated_metadata():
    """Saving one URI neither mutates sibling entries nor touches dead artifacts."""
    existing_metadata = {
        "trial_decoders": {
            OTHER_URI: {"mode": "builtin", "decoder_id": "other"},
        },
        "deployment": {"service": "keep-me", "nested": {"ready": True}},
    }
    artifacts = [{"legacy": "unchanged"}]
    row = types.SimpleNamespace(metadata_=existing_metadata, artifacts=artifacts)
    db = _FakeDB(row)
    service = ModelService(db, _FakeS3())
    before = deepcopy(existing_metadata)
    selected = {
        "mode": "custom", "task": "classification",
        "input": {"tensor": "images", "mean": [0.0]},
    }

    service.save_trial_decoder(
        "tenant", "model-id", "v1", ONNX_URI, selected
    )

    assert existing_metadata == before
    assert row.metadata_ is not existing_metadata
    assert row.metadata_["deployment"] == before["deployment"]
    assert row.metadata_["trial_decoders"][OTHER_URI] == before["trial_decoders"][OTHER_URI]
    assert row.metadata_["trial_decoders"][ONNX_URI] == selected
    assert row.artifacts is artifacts
    assert db.commits == 1
    selected["input"]["mean"][0] = 99.0
    assert row.metadata_["trial_decoders"][ONNX_URI]["input"]["mean"] == [0.0]
    compiled = str(db.statement.compile(dialect=postgresql.dialect()))
    assert "model_versions.tenant_id" in compiled
    assert "FOR UPDATE" in compiled


def test_failed_inference_preserves_structured_422_and_does_not_persist(monkeypatch):
    """ML mapping failures retain status/detail/code and never write a selection."""
    service = _service(_version())
    _mock_post(monkeypatch, [_response(422, {
        "error": "decoder mapping required",
        "code": "decoder_mapping_required",
    })])

    with pytest.raises(ModelTrialError) as caught:
        service.try_predict("tenant", "model-id", "v1", "aW1hZ2U=")

    assert caught.value.status_code == 422
    assert caught.value.detail == "decoder mapping required"
    assert caught.value.code == "decoder_mapping_required"
    assert service.saved_calls == []


def test_trial_route_parses_decoder_json_and_forwards_top_k():
    """The multipart API parses decoder JSON and keeps normalized extra fields."""
    model = _RouteModel()
    decoder = {"mode": "builtin", "decoder_id": "classification"}

    response = _client(model).post(
        "/api/v1/models/demo/versions/v1/try?path=model.onnx",
        files={"image": ("image.jpg", b"image-bytes", "image/jpeg")},
        data={"decoder_config": json.dumps(decoder), "top_k": "7"},
    )

    assert response.status_code == 200
    assert response.json()["task_specific"] == "preserved"
    args, kwargs = model.trial_calls[0]
    assert args[:3] == ("default", "model-id", "v1")
    assert kwargs["decoder_config"] == decoder
    assert kwargs["top_k"] == 7
    assert kwargs["artifact_path"] == "model.onnx"


@pytest.mark.parametrize("top_k", [0, 101])
def test_trial_route_rejects_out_of_range_top_k(top_k):
    """Top-K bounds reject invalid requests before model inference begins."""
    model = _RouteModel()

    response = _client(model).post(
        "/api/v1/models/demo/versions/v1/try",
        files={"image": ("image.jpg", b"image-bytes", "image/jpeg")},
        data={"top_k": str(top_k)},
    )

    assert response.status_code == 422
    assert model.trial_calls == []


@pytest.mark.parametrize("raw", ["not-json", "[]", "null"])
def test_trial_route_rejects_decoder_config_that_is_not_an_object(raw):
    """Malformed and non-object decoder fields fail before service inference."""
    model = _RouteModel()

    response = _client(model).post(
        "/api/v1/models/demo/versions/v1/try",
        files={"image": ("image.jpg", b"image-bytes", "image/jpeg")},
        data={"decoder_config": raw},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "decoder_config must be a JSON object"}
    assert model.trial_calls == []


def test_trial_route_preserves_decoder_mapping_required_422():
    """The public error envelope exposes both a useful detail and decoder code."""
    model = _RouteModel(ModelTrialError(
        422, "decoder mapping required", "decoder_mapping_required"
    ))

    response = _client(model).post(
        "/api/v1/models/demo/versions/v1/try",
        files={"image": ("image.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "decoder mapping required",
        "error": "decoder mapping required",
        "code": "decoder_mapping_required",
    }


def test_schema_route_forwards_path_without_adding_null_decoder_fields():
    """The typed schema response retains ML extras and the exact saved shape."""
    model = _RouteModel()

    response = _client(model).get(
        "/api/v1/models/demo/versions/v1/try/schema?path=model.onnx"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["saved_decoder"] == {"mode": "auto"}
    assert payload["input_shape"] == [3, 224, 224]
    assert payload["inputs"][0]["backend_detail"] == "preserved"
    assert payload["decoders"][0]["score"] == 0.9
    assert payload["auto_decoder"]["reason"] == "logits"
    assert model.inspect_calls[0][1]["artifact_path"] == "model.onnx"


def test_explicit_traversal_path_is_rejected_before_ml_or_persistence(monkeypatch):
    """An explicit traversal path cannot escape the live version artifact set."""
    service = _service(_version())
    calls = _mock_post(monkeypatch, [])

    with pytest.raises(ValueError, match="invalid artifact path"):
        service.try_predict(
            "tenant", "model-id", "v1", "aW1hZ2U=", artifact_path="../secret.onnx"
        )

    assert calls == []
    assert service.saved_calls == []
    assert service.s3.anchors == []
