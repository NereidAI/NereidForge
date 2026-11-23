# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for JobService start idempotency and terminal-job restart.

Repeated POST /{id}/start calls must not orphan an active run, while an
explicit restart of a cancelled/failed job must reset stale state and submit
exactly one replacement run.
"""

import types

import pytest

from app.services import jobs as jobs_mod
from app.services.jobs import JobService


@pytest.mark.parametrize("status", ["pending", "running", "completed"])
def test_start_is_idempotent_when_run_already_exists(monkeypatch, status):
    """A non-retryable job with a kfp_run_id is returned without resubmitting."""
    monkeypatch.setattr(jobs_mod, "_job_dict", lambda j: {"kfp_run_id": j.kfp_run_id})
    svc = JobService.__new__(JobService)
    job = types.SimpleNamespace(
        id="job1", type="training", status=status, kfp_run_id="run-existing",
    )
    svc.get_job = lambda jt, tenant, jid: job

    class _FakeKFP:
        def __init__(self) -> None:
            self.submitted = 0

        def submit_run(self, spec):
            self.submitted += 1
            raise AssertionError("must not resubmit")

    svc.kfp = _FakeKFP()
    out = svc.start_job("training", "default", "job1")

    assert out["kfp_run_id"] == "run-existing"
    assert svc.kfp.submitted == 0


@pytest.mark.parametrize("terminal_status", ["cancelled", "failed"])
def test_restart_terminal_job_submits_fresh_run(monkeypatch, terminal_status):
    """Restart clears stale run/result state before submitting exactly one run."""
    monkeypatch.setattr(
        jobs_mod,
        "_job_dict",
        lambda j: {
            "status": j.status,
            "kfp_run_id": j.kfp_run_id,
            "error_message": j.error_message,
            "results": j.results,
            "metrics": j.metrics,
            "progress": j.progress,
        },
    )

    class _FakeDB:
        def __init__(self) -> None:
            self.commits = 0

        def commit(self) -> None:
            self.commits += 1

        def refresh(self, job) -> None:
            pass

    class _FakeKFP:
        def __init__(self) -> None:
            self.specs = []

        def submit_run(self, spec):
            self.specs.append(spec)
            return types.SimpleNamespace(run_id="run-new")

    job = types.SimpleNamespace(
        id="job1",
        type="training",
        name="retry-me",
        status=terminal_status,
        kfp_run_id="run-stale",
        error_message="old failure",
        results={"registered_model": {"id": "stale"}},
        metrics={"current_epoch": 10},
        logs="old logs",
        progress=75,
        started_at="old-start",
        completed_at="old-end",
        params={"custom_image": "trainer:test"},
        config={"epochs": 100},
    )
    svc = JobService.__new__(JobService)
    svc.db = _FakeDB()
    svc.kfp = _FakeKFP()
    svc.get_job = lambda job_type, tenant, job_id: job
    svc.training_pipeline_params = lambda tenant, params, current_job: {
        "trainer_image": "trainer:test",
    }
    svc.check_capability = lambda tenant, params, config: None
    svc.tenant_kfp_userid = lambda tenant: "user"
    svc.tenant_kfp_experiment = lambda tenant: "experiment"
    svc._now = lambda: "new-start"

    out = svc.restart_job("training", "default", "job1")

    assert len(svc.kfp.specs) == 1
    assert svc.kfp.specs[0].pipeline_name == "training-pipeline"
    assert out == {
        "status": "running",
        "kfp_run_id": "run-new",
        "error_message": "",
        "results": {},
        "metrics": {},
        "progress": 0,
    }
    assert job.started_at == "new-start"
    assert job.completed_at is None
    assert job.logs == ""
    assert svc.db.commits == 1
