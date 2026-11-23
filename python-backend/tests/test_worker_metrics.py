# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Tests for the worker's training-metrics fallback (results.csv parsing and
# precedence over the trainer's metrics.json).
from __future__ import annotations

import json
from types import SimpleNamespace

from app.services.jobs import JobService
from app.worker import Worker


HEADER = (
    "epoch,time,metrics/precision(B),metrics/recall(B),"
    "metrics/mAP50(B),metrics/mAP50-95(B)\n"
)


def _csv() -> bytes:
    rows = [
        # epoch 48: mediocre
        "48,100,0.80,0.60,0.60833,0.41665",
        # epoch 49: slightly better
        "49,101,0.81,0.61,0.60977,0.41838",
        # epoch 50 (last): still mediocre, must NOT be selected
        "50,102,0.83,0.58933,0.61047,0.42061",
        # epoch 37: the fitness peak, matches best.pt
        "37,90,0.86657,0.82688,0.86677,0.59895",
    ]
    return (HEADER + "\n".join(rows) + "\n").encode()


class _FakeS3:
    """Minimal S3 stand-in serving one CSV payload or raising."""

    def __init__(self, payload: bytes | None):
        self._payload = payload

    def read_object(self, path: str) -> bytes:
        if self._payload is None:
            raise FileNotFoundError(path)
        return self._payload

    @staticmethod
    def tenant_filer_path(path: str, tenant_id: str) -> str:
        return f"/tenant/{tenant_id}{path}"


def _worker(payload: bytes | None) -> Worker:
    w = Worker.__new__(Worker)
    w.s3 = _FakeS3(payload)
    return w


def _job() -> SimpleNamespace:
    return SimpleNamespace(id="job-1", tenant_id="default")


def test_picks_best_fitness_row_not_last_row() -> None:
    m = _worker(_csv())._read_final_metrics(_job())
    assert m["mAP50"] == 0.86677
    assert m["mAP50-95"] == 0.59895
    assert m["precision"] == 0.86657
    assert m["recall"] == 0.82688


def test_reads_cumulative_training_time_independently_of_best_row() -> None:
    """The final cumulative time is not the timestamp of the best checkpoint."""
    m = _worker(_csv())._read_final_metrics(_job())
    assert m["train_time_s"] == 102.0


def test_missing_csv_returns_empty() -> None:
    assert _worker(None)._read_final_metrics(_job()) == {}


def test_empty_csv_returns_empty() -> None:
    assert _worker(b"")._read_final_metrics(_job()) == {}


def test_header_only_csv_returns_empty() -> None:
    assert _worker(HEADER.encode())._read_final_metrics(_job()) == {}


def test_merge_does_not_overwrite_trainer_metrics() -> None:
    # metrics.json values (final re-validation of best.pt) must win.
    metrics = {
        "mAP50": 0.8644898,
        "mAP50-95": 0.5943136,
        "precision": 0.86636,
        "recall": 0.82574,
        "train_time_s": 88.5,
    }
    _worker(_csv())._merge_csv_fallback(metrics, _job())
    assert metrics["mAP50"] == 0.8644898
    assert metrics["mAP50-95"] == 0.5943136
    assert metrics["train_time_s"] == 88.5


def test_merge_fills_missing_keys_from_csv() -> None:
    metrics: dict = {}
    _worker(_csv())._merge_csv_fallback(metrics, _job())
    assert metrics["mAP50"] == 0.86677
    assert metrics["mAP50-95"] == 0.59895


def test_merge_partial_metrics_only_fills_gaps() -> None:
    metrics = {"mAP50": 0.8644898}
    _worker(_csv())._merge_csv_fallback(metrics, _job())
    assert metrics["mAP50"] == 0.8644898
    assert metrics["mAP50-95"] == 0.59895


def test_training_metrics_recovers_time_for_an_already_finished_job() -> None:
    """Historical BYOT jobs need the same CSV fallback as new completions."""
    svc = JobService.__new__(JobService)
    svc.training_curves = lambda tenant, job_id: [
        {"epoch": "99", "time": "10000.25"},
        {"epoch": "100", "time": "10123.75"},
    ]
    original = {"mAP50": 0.98}

    metrics = svc.training_metrics("default", "job-1", original)

    assert metrics == {"mAP50": 0.98, "train_time_s": 10123.8}
    assert original == {"mAP50": 0.98}


def test_training_metrics_does_not_replace_trainer_reported_time() -> None:
    svc = JobService.__new__(JobService)
    svc.training_curves = lambda *_args: (_ for _ in ()).throw(
        AssertionError("CSV must not be read when metrics.json reports train_time_s")
    )

    assert svc.training_metrics("default", "job-1", {"train_time_s": 88.5}) == {
        "train_time_s": 88.5
    }


class _ProgressDb:
    """Minimal session for exercising terminal timestamp ownership."""

    def __init__(self):
        self.job = SimpleNamespace(
            status="running", progress=50, error_message="", results={}, metrics={},
            completed_at=None,
        )

    def get(self, _model, _id):
        return self.job

    def commit(self):
        pass


def test_terminal_progress_sets_completed_at_once() -> None:
    svc = JobService.__new__(JobService)
    svc.db = _ProgressDb()

    svc.set_progress("job-1", "running", 60)
    assert svc.db.job.completed_at is None
    svc.set_progress("job-1", "completed", 100)
    finished = svc.db.job.completed_at
    assert finished is not None

    svc.set_progress("job-1", "completed", 100)
    assert svc.db.job.completed_at is finished


def test_progress_persistence_replaces_non_finite_numbers_with_null() -> None:
    """PostgreSQL JSONB must never receive Python's non-standard NaN/Infinity."""
    svc = JobService.__new__(JobService)
    svc.db = _ProgressDb()

    svc.set_progress(
        "job-1", "running", 10,
        metrics={"history": [{"epoch": 10, "val_loss": float("nan")}]},
        results={"values": [float("inf"), float("-inf"), 1.25]},
    )

    assert svc.db.job.metrics["history"][0]["val_loss"] is None
    assert svc.db.job.results["values"] == [None, None, 1.25]
    json.dumps(svc.db.job.metrics, allow_nan=False)
    json.dumps(svc.db.job.results, allow_nan=False)


def test_progress_curves_replace_non_finite_numbers_with_null() -> None:
    """A missing validation loss must not invalidate the entire curves response."""
    payload = b'{"history":[{"epoch":10,"train_loss":3.8,"val_loss":NaN}]}'

    def read_object(path: str) -> bytes:
        if path.endswith("/progress.json"):
            return payload
        raise FileNotFoundError(path)

    svc = JobService.__new__(JobService)
    svc.s3 = SimpleNamespace(
        read_object=read_object,
        tenant_filer_path=lambda path, _tenant: path,
    )

    rows = svc.training_curves("default", "job-1")

    assert rows == [{"epoch": 10, "train_loss": 3.8, "val_loss": None}]
    json.dumps(rows, allow_nan=False)


class _PendingDb:
    """Track rollback ordering across two independent job reconciliations."""

    is_active = True

    def __init__(self, jobs: list[SimpleNamespace], events: list[str]):
        self.jobs = jobs
        self.events = events

    def scalars(self, _query):
        return SimpleNamespace(all=lambda: self.jobs)

    def rollback(self) -> None:
        self.events.append("rollback")


def test_reconcile_rolls_back_before_processing_the_next_job() -> None:
    """One malformed job must not poison the next job's database transaction."""
    jobs = [
        SimpleNamespace(id="bad", type="training", kfp_run_id="run-bad"),
        SimpleNamespace(id="good", type="training", kfp_run_id="run-good"),
    ]
    events: list[str] = []
    worker = Worker.__new__(Worker)
    worker.db = _PendingDb(jobs, events)

    def reconcile(job: SimpleNamespace) -> None:
        events.append(job.id)
        if job.id == "bad":
            raise RuntimeError("invalid JSONB value")

    worker.reconcile_job = reconcile
    worker.process_pending_jobs()

    assert events == ["bad", "rollback", "good", "rollback"]


def test_kfp_run_preserves_task_states_for_optimization_progress(monkeypatch) -> None:
    """test_kfp_run_preserves_task_states_for_optimization_progress keeps KFP evidence."""
    from app.clients.kfp import KFPClient

    tasks = [{"display_name": "baseline-step", "state": "SUCCEEDED"}]
    response = SimpleNamespace(
        status_code=200,
        json=lambda: {
            "run_id": "run-1",
            "state": "RUNNING",
            "run_details": {"task_details": tasks},
        },
    )
    monkeypatch.setattr("app.clients.kfp.httpx.get", lambda *_args, **_kwargs: response)
    client = KFPClient(lambda: {"kfp_host": "http://kfp", "kfp_userid": "user"})

    run = client.get_run("run-1")

    assert run.tasks == tasks


def test_running_optimization_reports_gate_after_optimizer_metrics() -> None:
    """test_running_optimization_reports_gate_after_optimizer_metrics reproduces the gate gap."""
    worker = Worker.__new__(Worker)
    worker.read_metrics = lambda _job: {"speedup_ratio": 1.387}
    worker.read_gate_report = lambda _job: {}
    updates = []
    worker.jobs = SimpleNamespace(
        set_progress=lambda *args, **kwargs: updates.append((args, kwargs)),
    )
    job = SimpleNamespace(
        id="job-opt", type="optimization", params={}, kfp_run_id="run-opt",
    )
    run = SimpleNamespace(
        status="Running",
        tasks=[{"display_name": "baseline-step", "state": "SUCCEEDED"}],
    )
    worker.kfp = SimpleNamespace(get_run=lambda _run_id: run)

    worker.reconcile_job(job)

    assert len(updates) == 1
    args, kwargs = updates[0]
    assert args == ("job-opt", "running", 50)
    assert kwargs["metrics"]["pipeline_stage"] == "validating"
    assert kwargs["metrics"]["completed_stages"] == 2
    assert kwargs["metrics"]["total_stages"] == 4
    assert kwargs["metrics"]["speedup_ratio"] == 1.387


def test_running_optimization_distinguishes_baseline_and_finalization() -> None:
    """test_running_optimization_distinguishes_baseline_and_finalization avoids time estimates."""
    worker = Worker.__new__(Worker)
    updates = []
    worker.jobs = SimpleNamespace(
        set_progress=lambda *args, **kwargs: updates.append((args, kwargs)),
    )
    job = SimpleNamespace(id="job-opt", type="optimization", params={})
    run = SimpleNamespace(
        tasks=[{"display_name": "baseline-step(0)", "state": "SUCCEEDED"}],
    )

    worker.read_metrics = lambda _job: {}
    worker.read_gate_report = lambda _job: {}
    worker._sync_optimization_progress(job, run)
    worker.read_metrics = lambda _job: {"mAP50": 0.96}
    worker.read_gate_report = lambda _job: {"gate_result": "PASS"}
    worker._sync_optimization_progress(job, run)

    assert updates[0][0] == ("job-opt", "running", 25)
    assert updates[0][1]["metrics"]["pipeline_stage"] == "optimizing"
    assert updates[1][0] == ("job-opt", "running", 75)
    assert updates[1][1]["metrics"]["pipeline_stage"] == "finalizing"
