# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for required credential validation and explicit propagation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import boto3
import pytest

from app.clients.k8s import K8sService, download_model_from_s3
from app.clients.kfp import KFPClient, PipelineRunSpec
from app.clients.package_jobs import PackageJobRunner
from app.core.config import (
    DatabaseConfig,
    config,
    load_config,
    require_s3_credentials,
)
from app.services.jobs import JobService
from app.worker import Worker


def _blank_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the shared storage pair without changing the process environment."""
    monkeypatch.setattr(config.storage, "s3_access_key", "")
    monkeypatch.setattr(config.storage, "s3_secret_key", "")


@pytest.mark.parametrize(
    ("access_key", "secret_key", "missing_field"),
    [
        ("", "pytest-secret", "storage.s3_access_key"),
        ("pytest-access", " \t", "storage.s3_secret_key"),
        ("", "", "storage.s3_access_key"),
    ],
)
def test_s3_pair_validation_names_missing_fields_without_leaking_values(
    access_key: str, secret_key: str, missing_field: str,
) -> None:
    """Incomplete pairs fail with field names rather than credential material."""
    with pytest.raises(ValueError) as caught:
        require_s3_credentials(access_key, secret_key, scope="storage.s3")

    message = str(caught.value)
    assert missing_field in message
    assert "NEREIDFORGE_STORAGE_S3_" in message
    assert "pytest-access" not in message
    assert "pytest-secret" not in message


def test_database_password_has_no_default_and_is_required_for_dsn() -> None:
    """DSN construction rejects an absent password instead of using a shared one."""
    database = DatabaseConfig(user="pytest-db-user")

    assert database.password == ""
    with pytest.raises(ValueError, match="NEREIDFORGE_DATABASE_PASSWORD"):
        _ = database.dsn

    database.password = "pytest-database-password"
    assert database.dsn.startswith("postgresql://pytest-db-user:")


def test_load_config_rejects_missing_storage_pair(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration fails before module-level S3 or service clients can exist."""
    path = tmp_path / "missing-storage-credentials.yaml"
    path.write_text("storage:\n  s3_access_key: ''\n  s3_secret_key: ''\n")
    monkeypatch.delenv("NEREIDFORGE_STORAGE_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("NEREIDFORGE_STORAGE_S3_SECRET_KEY", raising=False)

    with pytest.raises(ValueError) as caught:
        load_config(path)

    assert "storage.s3_access_key" in str(caught.value)
    assert "storage.s3_secret_key" in str(caught.value)


def test_worker_rejects_missing_pair_before_constructing_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker startup performs no DB, KFP, or S3 factory calls with a bad pair."""
    _blank_storage(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr("app.worker.SessionLocal", lambda: calls.append("db"))
    monkeypatch.setattr("app.worker.KFPClient", lambda *_args: calls.append("kfp"))
    monkeypatch.setattr("app.worker.s3_client", lambda: calls.append("s3"))

    with pytest.raises(ValueError, match="storage.s3_access_key"):
        Worker()

    assert calls == []


def test_job_service_rejects_missing_pair_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normal dependency-injection path cannot create jobs with a bad pair."""
    _blank_storage(monkeypatch)

    with pytest.raises(ValueError, match="storage.s3_access_key"):
        JobService(SimpleNamespace(), None)  # type: ignore[arg-type]


def test_package_job_requires_and_propagates_only_explicit_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package manifests carry the supplied pair and reject an incomplete pair."""
    _blank_storage(monkeypatch)
    with pytest.raises(ValueError, match="storage.s3_access_key"):
        PackageJobRunner(batch_api=SimpleNamespace(), core_api=SimpleNamespace())

    runner = PackageJobRunner(
        batch_api=SimpleNamespace(),
        core_api=SimpleNamespace(),
        s3_endpoint="http://object-store.test",
        s3_access_key="pytest-package-access-key",
        s3_secret_key="pytest-package-secret-key",
    )
    manifest = runner.job_manifest(
        "s3://bucket/request.json", "onnx-structural", "package-test",
    )
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"]}

    assert env["S3_ACCESS_KEY"] == "pytest-package-access-key"
    assert env["S3_SECRET_KEY"] == "pytest-package-secret-key"


def test_kfp_logs_reject_missing_pair_before_boto3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KFP log retrieval never constructs an S3 client with fallback credentials."""
    monkeypatch.setattr(config.kfp, "object_store_endpoint", "http://object-store.test")
    monkeypatch.setattr(config.kfp, "object_store_access_key", "")
    monkeypatch.setattr(config.kfp, "object_store_secret_key", "")
    calls: list[str] = []
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: calls.append("boto3"))

    with pytest.raises(ValueError, match="kfp.object_store_access_key"):
        KFPClient(lambda: {}).get_run_logs("run-test")

    assert calls == []


def test_model_download_does_not_swallow_missing_credentials(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credential errors remain visible instead of becoming a false download result."""
    _blank_storage(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: calls.append("boto3"))

    with pytest.raises(ValueError, match="storage.s3_access_key"):
        download_model_from_s3("s3://bucket/model.onnx", str(tmp_path / "model.onnx"))

    assert calls == []


def test_onnx_conversion_rejects_missing_pair_before_pipeline_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No conversion pipeline is submitted until its storage pair is complete."""
    _blank_storage(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        "app.clients.k8s.cluster_ops.submit_pipeline",
        lambda *_args, **_kwargs: calls.append("pipeline"),
    )

    with pytest.raises(ValueError, match="storage.s3_access_key"):
        K8sService().convert_to_onnx("s3://bucket/input.pt", "s3://bucket/output.onnx")

    assert calls == []


def _pipeline_service(access_key: str, secret_key: str, monkeypatch: pytest.MonkeyPatch) -> JobService:
    """Build a job service with entity resolution isolated from external stores."""
    monkeypatch.setattr(config.storage, "s3_access_key", access_key)
    monkeypatch.setattr(config.storage, "s3_secret_key", secret_key)
    service = JobService.__new__(JobService)
    service.s3 = SimpleNamespace(
        s3_endpoint="http://object-store.test",
        tenant_uri=lambda uri, _tenant: uri,
    )
    service._resolve_model_artifact_uri = lambda *_args: "s3://bucket/model.onnx"
    service._resolve_eval_dataset_uri = lambda *_args: "s3://bucket/dataset/"
    return service


def test_training_and_evaluation_payloads_use_explicit_dummy_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trainer, optimizer, and evaluator payloads propagate configured values only."""
    service = _pipeline_service(
        "pytest-pipeline-access-key", "pytest-pipeline-secret-key", monkeypatch,
    )
    job = SimpleNamespace(id="job-test", batch_id=None, params={}, config={})

    trainer_env = {
        item["name"]: item["value"] for item in service._trainer_env()
    }
    optimized = service.optimization_pipeline_params("default", {}, job)
    evaluated = service.evaluation_pipeline_params("default", {}, job)

    assert trainer_env["S3_ACCESS_KEY"] == "pytest-pipeline-access-key"
    assert trainer_env["S3_SECRET_KEY"] == "pytest-pipeline-secret-key"
    for payload in (optimized, evaluated):
        assert payload["s3_access_key"] == "pytest-pipeline-access-key"
        assert payload["s3_secret_key"] == "pytest-pipeline-secret-key"


@pytest.mark.parametrize("pipeline_name", ["optimization-pipeline", "evaluation-pipeline"])
def test_kfp_submission_retains_required_s3_parameters(
    pipeline_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final KFP request retains required credentials but drops internal keys."""
    requests: list[dict] = []

    def fake_post(_url, **kwargs):
        """fake_post captures the exact run body without contacting KFP."""
        requests.append(kwargs)
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"run_id": "run-test", "state": "SUCCEEDED"},
        )

    monkeypatch.setattr("app.clients.kfp.httpx.post", fake_post)
    client = KFPClient(lambda: {"kfp_host": "http://kfp.test"})
    result = client.submit_run(PipelineRunSpec(
        pipeline_name=pipeline_name,
        display_name="credential-boundary-test",
        pipeline_id="pipeline-test",
        pipeline_version_id="version-test",
        parameters={
            "model_uri": "s3://bucket/model.onnx",
            "s3_access_key": "pytest-submit-access-key",
            "s3_secret_key": "pytest-submit-secret-key",
            "internal_key": "must-not-be-submitted",
        },
    ))

    parameters = requests[0]["json"]["runtime_config"]["parameters"]
    assert parameters == {
        "model_uri": "s3://bucket/model.onnx",
        "s3_access_key": "pytest-submit-access-key",
        "s3_secret_key": "pytest-submit-secret-key",
    }
    assert result.parameters == parameters


def test_katib_trial_and_pipeline_use_explicit_dummy_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Katib receives identical explicit credentials in its Pod and pipeline input."""
    service = _pipeline_service(
        "pytest-katib-access-key", "pytest-katib-secret-key", monkeypatch,
    )
    submitted: dict = {}

    def submit(*_args, **kwargs):
        submitted.update(kwargs)
        return "run-test"

    monkeypatch.setattr("app.clients.cluster_ops.submit_pipeline", submit)
    job = SimpleNamespace(id="job-katib")

    assert service._launch_katib_search("default", {}, job) == "run-test"

    params = submitted["params"]
    experiment = json.loads(params["experiment_json"])
    container = experiment["spec"]["trialTemplate"]["trialSpec"]["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["S3_ACCESS_KEY"] == "pytest-katib-access-key"
    assert env["S3_SECRET_KEY"] == "pytest-katib-secret-key"
    assert params["s3_access_key"] == "pytest-katib-access-key"
    assert params["s3_secret_key"] == "pytest-katib-secret-key"
