# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Worker process: polls KFP run status and processes async tasks.

Mirrors the Go worker (internal/worker/worker.go):
- reconciles pending/running jobs against KFP run state
- reads real metrics/artifacts from SeaweedFS (multiple candidate paths)
- registers trained/optimized models to the hub when jobs complete
- processes async tasks: dataset_import / preannotate / batch_accept
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.clients.cvat import CvatClient
from app.clients.hub import hub_client
from app.clients.kfp import KFPClient
from app.clients.labelstudio import LabelStudioClient
from app.clients.package_jobs import PackageJobRunner
from app.clients.s3 import SeaweedFSClient
from app.core.config import config, require_s3_credentials
from app.core.images import nf_image
from app.db.session import SessionLocal
from app.models.orm import AsyncTask, DatasetVersion, Job, MLBackend
from app.services import kfp_config_fn
from app.services.admin import AdminService
from app.services.dataset import AsyncTaskCancelled, DatasetService
from app.services.jobs import JobService, normalize_target_device
from app.services.model_packages import ModelPackageService
from app.services.models import ModelService
from app.services.project import ProjectService

logger = logging.getLogger("nereidforge.worker")

TRAINING_WINDOW = timedelta(minutes=20)
METRICS_PATHS = {
    "training": ("models", "{id}.metrics.json", "{id}/{id}.metrics.json"),
    "notebook": ("models", "{id}.metrics.json", "{id}/{id}.metrics.json"),
    "optimization": ("optimized", "{id}.metrics.json", "{id}/{id}.metrics.json"),
    "evaluation": ("eval", "{id}.metrics.json", "{id}/{id}.metrics.json"),
}

# job.config accelerator label -> device family recorded in MR metadata.
_ACCELERATOR_HARDWARE = {"intel": "xpu", "nvidia": "cuda", "amd": "rocm"}


def _trained_hardware(metrics: dict[str, Any], job: Job) -> str:
    """Device family the model actually trained on (xpu/cuda/rocm/cpu).

    Prefer the trainer's reported ``device`` (e.g. "xpu:0" -> "xpu"); fall
    back to the job accelerator profile. Older code hardcoded "cpu", which
    mislabeled every GPU/XPU run in the Model Registry.
    """
    device = str((metrics or {}).get("device") or "").split(":", 1)[0].strip()
    if device:
        return device
    accelerator = str(((job.config or {}).get("accelerator") or "")).strip().lower()
    return _ACCELERATOR_HARDWARE.get(accelerator, "cpu")


class Worker:
    """Background worker reconciling KFP runs and processing async tasks."""

    def __init__(self) -> None:
        # This must precede _db/KFP/S3 construction: a worker with incomplete
        # credentials should fail at startup, not after clients or jobs exist.
        access_key, secret_key = require_s3_credentials(
            config.storage.s3_access_key,
            config.storage.s3_secret_key,
            scope="storage.s3",
        )
        self.poll_interval = max(1, config.worker.poll_interval or 3)
        self.db: Optional[Session] = None
        self.kfp = KFPClient(kfp_config_fn(self._db()))
        self.s3 = s3_client()
        self.admin = AdminService(self._db())
        self.dataset = DatasetService(self._db(), self.s3, self.kfp)
        self.models = ModelService(self._db(), self.s3, hub=hub_client)
        self.packages = ModelPackageService(self.s3, self.models, self.dataset)
        self.package_jobs = PackageJobRunner(
            namespace=config.serving.namespace,
            s3_endpoint=config.storage.s3_cluster_endpoint or self.s3.s3_endpoint(),
            s3_access_key=access_key,
            s3_secret_key=secret_key,
        )
        self.jobs = JobService(
            self._db(), self.s3, kfp=self.kfp, dataset=self.dataset,
            trainer_config_fn=self._trainer_cfg(), models=self.models,
        )
        self.ls = LabelStudioClient()
        self.cvat = CvatClient()
        self.project = ProjectService(
            self._db(), self.s3, self.ls, cvat=self.cvat, dataset=self.dataset,
            s3_access_key=access_key,
            s3_secret_key=secret_key,
        )

    # -- session helpers ----------------------------------------------------

    def _db(self) -> Session:
        if self.db is None or not self.db.is_active:
            self.db = SessionLocal()
        return self.db

    def _trainer_cfg(self):
        def _read() -> tuple[str, str, str, str, str, str, str]:
            cfg = self.admin.get_system_config()
            return (
                str(cfg.get("trainer_accelerator_type", "") or ""),
                str(cfg.get("trainer_gpu_request", "0") or "0"),
                str(cfg.get("trainer_cpu_request", "4") or "4"),
                str(cfg.get("trainer_cpu_limit", "8") or "8"),
                str(cfg.get("trainer_memory_request", "8Gi") or "8Gi"),
                str(cfg.get("trainer_memory_limit", "16Gi") or "16Gi"),
                str(cfg.get("trainer_image", "") or nf_image("trainer/torch-cpu")),
            )

        return _read

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        """Run the worker loop until interrupted."""
        logger.info("worker started (poll=%ss)", self.poll_interval)
        ml_backend_sync_counter = 0
        ML_BACKEND_SYNC_INTERVAL = 10  # sync every 10 cycles (30s with default poll=3s)
        while True:
            try:
                self.process_pending_jobs()
                self.process_async_tasks()
                ml_backend_sync_counter += 1
                if ml_backend_sync_counter >= ML_BACKEND_SYNC_INTERVAL:
                    self.sync_ml_backend_status()
                    ml_backend_sync_counter = 0
            except Exception as e:  # noqa: BLE001
                logger.exception("worker cycle failed: %s", e)
                try:
                    if self.db is not None and self.db.is_active:
                        self.db.rollback()
                except Exception:  # noqa: BLE001
                    pass
            self._close_db()
            time.sleep(self.poll_interval)

    def _close_db(self) -> None:
        if self.db is not None:
            try:
                self.db.close()
            except Exception:
                pass
            self.db = None

    # -- jobs reconciliation ------------------------------------------------

    def process_pending_jobs(self) -> None:
        """Reconcile pending/running jobs against KFP run state."""
        db = self._db()
        rows = db.scalars(
            select(Job).where(Job.status.in_(["pending", "running"])).order_by(Job.created_at)
        ).all()
        notebook_rows = [j for j in rows if j.type == "notebook"]
        for job in notebook_rows:
            if job.kfp_run_id or job.status != "pending":
                continue
            try:
                job.kfp_run_id = self._launch_notebook(job)
                job.status = "running"
                job.progress = 5
                self.db.commit()
                logger.info("notebook job %s submitted as run %s", job.id, job.kfp_run_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("notebook job %s submit failed: %s", job.id, e)
                self.jobs.set_progress(job.id, "failed", 0, str(e))
        rows = [j for j in rows if j.type != "notebook" or j.kfp_run_id]

        for job in rows:
            if not job.kfp_run_id:
                continue
            try:
                self.reconcile_job(job)
            except Exception as e:  # noqa: BLE001
                logger.warning("reconcile job %s failed: %s", job.id, e)
                # Roll back before touching the next job. Otherwise one invalid
                # JSON value poisons every subsequent reconciliation in this cycle.
                try:
                    if self.db is not None:
                        self.db.rollback()
                except Exception:  # noqa: BLE001
                    pass
        # A failed flush inside reconcile leaves the session poisoned; rolling
        # back here lets the next cycle start from a clean transaction.
        try:
            if self.db is not None and self.db.is_active:
                self.db.rollback()
        except Exception:  # noqa: BLE001
            pass

    def reconcile_job(self, job: Job) -> None:
        """Advance a job based on its KFP run state."""
        run = self.kfp.get_run(job.kfp_run_id)
        status = run.status
        if status in ("Succeeded", "Failed", "Error", "Cancelled"):
            pass  # terminal; execution node reporting was removed with cluster access
        if status == "Succeeded":
            if job.type == "optimization" and self._is_search_mode(job):
                self._register_katib_result(job)
            self.finish_job(job, "completed", 100, "")
        elif status in ("Failed", "Error"):
            if job.type == "optimization":
                # Gate failure is a first-class outcome: the report exists and
                # records FAIL; the job fails with the gate reason and the
                # artifact is NOT registered (admin/creator may override).
                report = self.read_gate_report(job)
                if report and report.get("gate_result") == "FAIL":
                    gates = report.get("gates") or {}
                    why = "; ".join(
                        f"{k}={v.get('value')} vs thr {v.get('threshold')}"
                        for k, v in gates.items() if v.get("hard") and not v.get("passed")
                    ) or "hard gate failed"
                    self.finish_job(job, "failed", 100, f"GATE_FAIL: {why}", report=report)
                    return
            if self.pipeline_artifacts_exist(job):
                self.finish_job(job, "completed", 100, run.display_name or "")
            elif job.type == "training" and self._training_window_open(job):
                # Training may still be writing live progress; hold off until
                # the window closes. Optimization/evaluation never stream
                # progress.json, so a failed run is terminal immediately.
                self._sync_live_progress(job)
            else:
                # Surface the pipeline's real failure reason (e.g. optimize
                # step stderr) instead of a generic "pipeline failed".
                self.finish_job(job, "failed", 0, self._run_failure_reason(job))
        elif status == "Cancelled":
            self.finish_job(job, "cancelled", 0, "")
        elif status == "Running":
            if job.type == "optimization" and not self._is_search_mode(job):
                self._sync_optimization_progress(job, run)
            else:
                self._sync_live_progress(job)
        else:
            # KFP run exists but is not Running/terminal yet (queued, pod
            # starting): keep progress at 0 — 5 here leaked into the UI for
            # every newly started job before real work began.
            self.jobs.set_progress(job.id, "pending" if job.status == "pending" else job.status, 0)

    def _is_search_mode(self, job: Job) -> bool:
        """True when the optimization job runs a Katib grid search (vs the
        fixed multi-target KFP DAG)."""
        return str((job.params or {}).get("search_mode") or "").lower() in ("1", "true", "yes")

    def _register_katib_result(self, job: Job) -> None:
        """Register the winning trial's optimized artifact as the job result.

        The optimization-search pipeline pod publishes best_trial.json to S3
        (it watched the Katib Experiment in-cluster); the worker only reads S3
        — it never touches the cluster.
        """
        try:
            path = f"/buckets/nereidforge/optimized/{job.id}/best_trial.json"
            body = self.s3.read_object(self.s3.tenant_filer_path(path, job.tenant_id))
            record = json.loads(body)
        except Exception:  # noqa: BLE001
            logger.warning("katib search %s: best_trial.json missing", job.id)
            return
        trial_name = str(record.get("best_trial") or "")
        if not trial_name:
            return
        base = f"s3://nereidforge/optimized/{job.id}/"
        artifact_uri = f"{base}{trial_name}/{trial_name}.optimized.onnx"
        # metrics.json written by evaluate.py inside the trial
        metrics: dict = {}
        try:
            filer = self.s3.tenant_filer_path(
                f"/buckets/nereidforge/optimized/{job.id}/{trial_name}/{trial_name}.metrics.json",
                job.tenant_id,
            )
            metrics = json.loads(self.s3.read_object(filer))
        except Exception:  # noqa: BLE001
            metrics = {}
        # The trial's evaluate run did not know the artifact path; attach it so
        # register_optimized_artifact can resolve the version's file.
        metrics["optimized_uri"] = artifact_uri
        params = dict(job.params or {})
        params["optimized_uri"] = artifact_uri
        params["search_mode"] = "true"
        params["quant_method"] = "graph"
        params["target_device"] = "cpu"
        job.params = params
        self.db.commit()
        # register optimized artifact to PG + hub (same path as gate PASS)
        from app.services.jobs import register_optimized_artifact
        try:
            register_optimized_artifact(self.models, job, metrics, actor="nereidforge-worker")
        except Exception as e:  # noqa: BLE001
            logger.warning("register katib result failed: %s", e)
            try:
                self.db.rollback()
            except Exception:  # noqa: BLE001
                pass

    def _run_failure_reason(self, job: Job) -> str:
        """Extract a concise failure reason from the run's executor logs.

        Falls back to a generic message when logs are unavailable.
        """
        if not job.kfp_run_id:
            return "pipeline failed"
        try:
            logs = self.kfp.get_run_logs(job.kfp_run_id, tail=400) or ""
        except Exception:
            return "pipeline failed"
        # Pull the last meaningful error-ish line(s): SystemExit/RuntimeError
        # messages, quantizer errors, GATE_FAIL, etc.
        lines = [ln.strip() for ln in logs.splitlines() if ln.strip()]
        hits = []
        for ln in reversed(lines):
            low = ln.lower()
            if any(k in low for k in (
                "systemexit", "runtimeerror", "requires", "not found",
                "failed", "error", "exception", "gate_fail", "protobuf",
                "unsupported", "cannot", "unable",
            )):
                hits.append(ln)
                if len(hits) >= 3:
                    break
        if not hits:
            return "pipeline failed"
        # Trim to a readable length.
        msg = " | ".join(reversed(hits))
        return msg[:400]

    def read_gate_report(self, job: Job) -> dict[str, Any]:
        """Read the gate report from evaluation-reports/{job_id}/report.json."""
        path = self.s3.tenant_filer_path(
            f"/buckets/nereidforge/evaluation-reports/{job.id}/report.json", job.tenant_id
        )
        try:
            return json.loads(self.s3.read_object(path))
        except Exception:
            return {}

    @staticmethod
    def _pipeline_step_succeeded(run: Any, step_name: str) -> bool:
        """_pipeline_step_succeeded matches component duplicates, excluding drivers."""
        for task in getattr(run, "tasks", []) or []:
            display_name = str(task.get("display_name") or "")
            if (
                task.get("state") == "SUCCEEDED"
                and (display_name == step_name or display_name.startswith(f"{step_name}("))
            ):
                return True
        return False

    def _sync_optimization_progress(self, job: Job, run: Any) -> None:
        """_sync_optimization_progress persists milestones without estimating duration.

        The shared baseline comes from KFP task state. This job's metrics and
        gate report prove its own optimize/gate completion, including when a
        batch has parallel branches. Four equal milestones describe completed
        workflow stages only; the terminal reconciler alone publishes 100%.
        """
        optimized = self.read_metrics(job)
        report = self.read_gate_report(job) if optimized else {}
        baseline_done = (
            self._pipeline_step_succeeded(run, "baseline-step")
            or bool(optimized)
            or bool(report)
        )
        completed = int(baseline_done) + int(bool(optimized)) + int(bool(report))
        if report:
            stage = "finalizing"
        elif optimized:
            stage = "validating"
        elif baseline_done:
            stage = "optimizing"
        else:
            stage = "baseline"
        live_metrics = dict(optimized)
        live_metrics.update({
            "pipeline_stage": stage,
            "completed_stages": completed,
            "total_stages": 4,
        })
        self.jobs.set_progress(job.id, "running", completed * 25, metrics=live_metrics)

    def _sync_live_progress(self, job: Job) -> None:
        """Push live training progress (epoch/loss) from progress.json into the job record."""
        prog = self.read_progress(job)
        if not prog:
            # No progress file yet (pipeline just started): report 0, not a
            # fake placeholder. train.py seeds progress.json at 0 before the
            # first epoch, so this window is short.
            self.jobs.set_progress(job.id, "running", 0)
            return
        pct = int(prog.get("progress_pct") or prog.get("current_epoch") * 100 / max(
            1, int(prog.get("total_epochs") or 1)
        ))
        metrics = dict(prog)
        metrics.pop("progress_pct", None)
        self.jobs.set_progress(job.id, "running", max(0, min(95, pct)), metrics=metrics)

    def read_progress(self, job: Job) -> dict[str, Any]:
        """Read live progress.json from SeaweedFS for a job (empty dict when absent)."""
        subdir, *_ = METRICS_PATHS.get(job.type, METRICS_PATHS["training"])
        base = f"/buckets/nereidforge/{subdir}"
        candidates = [
            f"{base}/{job.id}/progress.json",
            f"{base}/{job.kfp_run_id}/progress.json",
            f"{base}/{job.id}.progress.json",
        ]
        for p in candidates:
            try:
                p = self.s3.tenant_filer_path(p, job.tenant_id)
                data = self.s3.read_object(p)
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return {}

    def _training_window_open(self, job: Job) -> bool:
        if not job.started_at:
            return False
        return datetime.now(timezone.utc) - job.started_at < TRAINING_WINDOW

    # -- artifact detection -------------------------------------------------

    def pipeline_artifacts_exist(self, job: Job) -> bool:
        """Check whether the job's metrics artifacts landed on SeaweedFS."""
        subdir, *_ = METRICS_PATHS.get(job.type, METRICS_PATHS["training"])
        candidates = self._metric_candidates(job, subdir)
        for p in candidates:
            try:
                self.s3.read_object(p)
                return True
            except Exception:
                continue
        return False

    def _metric_candidates(self, job: Job, subdir: str) -> list[str]:
        """Build candidate filer paths for a job's metrics file."""
        base = f"/buckets/nereidforge/{subdir}"
        paths = [
            f"{base}/{job.kfp_run_id}.metrics.json",
            f"{base}/{job.id}.metrics.json",
            f"{base}/{job.kfp_run_id}/{job.kfp_run_id}.metrics.json",
            f"{base}/{job.id}/{job.id}.metrics.json",
        ]
        if job.type == "optimization":
            # The multi-target DAG sets OUTPUT_DIR=<output_uri>/<artifact_id>
            # and optimize.py writes <artifact_id>.metrics.json inside it.
            # For a single job output_uri already ends in the job id, so the
            # artifact lands one level DEEPER than training:
            # optimized/{id}/{id}/{id}.metrics.json. (Batch jobs use
            # optimized/{batch_id}/{id}/... and are covered below.)
            paths += [
                f"{base}/{job.id}/{job.id}/{job.id}.metrics.json",
                f"{base}/{job.kfp_run_id}/{job.kfp_run_id}/{job.kfp_run_id}.metrics.json",
            ]
        if job.batch_id:
            # Optimization batch layout: optimized/{batch_id}/{job_id}/...
            # batch_id already carries the "batch-" prefix.
            paths += [
                f"{base}/{job.batch_id}/{job.id}/{job.id}.metrics.json",
                f"{base}/{job.batch_id}/{job.kfp_run_id}/{job.kfp_run_id}.metrics.json",
            ]
        # Notebook jobs write to the BARE bucket (ADR-013 D6 model dir
        # s3://<bucket>/models/{job_id}/ -- the injected NF_MODEL_DIR), while
        # trainer pipelines target tenant-scoped paths. Skip tenant injection
        # for notebooks so read_metrics finds the file the notebook was told
        # to produce; job ids are globally unique, so no collision risk.
        if job.type == "notebook":
            return paths
        return [self.s3.tenant_filer_path(p, job.tenant_id) for p in paths]

    def read_metrics(self, job: Job) -> dict[str, Any]:
        """Read real metrics from SeaweedFS for a job (empty dict when absent)."""
        subdir, *_ = METRICS_PATHS.get(job.type, METRICS_PATHS["training"])
        for p in self._metric_candidates(job, subdir):
            try:
                data = self.s3.read_object(p)
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return {}

    def _discover_artifact_uri(self, job: Job) -> str:
        """Find the produced model file in a training job's output directory.

        Custom scripts (I-11) are not required to emit the trainer's metrics
        contract, so when no weights/onnx URI is reported we scan the job's
        output prefix and pick the best artifact: prefer ONNX, then common
        weight formats, otherwise the largest non-metrics/non-log file.
        """
        output_uri = str((job.params or {}).get("output_uri") or "").strip()
        if not output_uri:
            output_uri = self.s3.tenant_uri(
                f"s3://nereidforge/models/{job.id}/", job.tenant_id
            )
        filer_path = self.s3.s3_uri_to_filer_path(output_uri)
        if not filer_path:
            return ""
        try:
            files = self.s3.walk(filer_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("_discover_artifact_uri: walk failed for %s: %s", filer_path, exc)
            return ""
        candidates = [
            f for f in files
            if not f.endswith(".metrics.json")
            and "/tensorboard/" not in f
            and not f.endswith((".csv", ".log", ".txt"))
        ]
        if not candidates:
            return ""
        # Preference order: ONNX > torch/safetensors weights > anything else.
        def _rank(path: str) -> tuple[int, int]:
            score = 0
            if path.endswith(".onnx"):
                score = 3
            elif path.endswith((".pt", ".pth", ".safetensors")):
                score = 2
            elif path.endswith((".bin", ".h5", ".engine")):
                score = 1
            return (score, self.s3.object_size(path))

        best = max(candidates, key=_rank)
        uri = "s3://" + best[len("/buckets/"):]
        return self.s3.tenant_uri(uri, job.tenant_id)

    # -- finish -------------------------------------------------------------

    def _read_final_metrics(self, job: Job) -> dict[str, Any]:
        """Read the best-checkpoint evaluation metrics from results.csv.

        The training container writes per-epoch validation metrics to
        results.csv (Ultralytics format). Ultralytics selects ``best.pt`` by
        fitness (0.1 * mAP50 + 0.9 * mAP50-95), so the row with the highest
        fitness -- not the last row -- matches the shipped checkpoint. Extract
        that row's key metrics with clean names so the job list can display
        mAP50-95 / mAP50 without an extra curves API call. This is a fallback
        for images that do not write a metrics.json; the trainer's own
        metrics.json always takes precedence in ``finish_job``. The independent
        cumulative ``time`` column also recovers duration for BYOT images that
        omit ``train_time_s`` from metrics.json.
        """
        try:
            filer = self.s3.tenant_filer_path(
                f"/buckets/nereidforge/models/{job.id}/tensorboard/results.csv",
                job.tenant_id,
            )
            data = self.s3.read_object(filer)
        except Exception:
            return {}
        text = data.decode("utf-8", errors="ignore")
        try:
            rows = list(csv.DictReader(io.StringIO(text)))
        except Exception:
            return {}
        if not rows:
            return {}

        def _fitness(row: dict[str, str]) -> float:
            # Mirror Ultralytics' checkpoint fitness: 0.1 * mAP50 + 0.9 *
            # mAP50-95; missing values count as zero so incomplete rows lose.
            try:
                map50 = float(row.get("metrics/mAP50(B)")
                              or row.get("metrics/mAP50") or 0)
                map_ = float(row.get("metrics/mAP50-95(B)")
                             or row.get("metrics/mAP50-95") or 0)
            except (ValueError, TypeError):
                return -1.0
            return 0.1 * map50 + 0.9 * map_

        best = max(rows, key=_fitness)
        # Map Ultralytics column names (prefixed, with/without the (B) suffix)
        # to clean metric keys.
        aliases: dict[str, list[str]] = {
            "mAP50-95": ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"],
            "mAP50": ["metrics/mAP50(B)", "metrics/mAP50", "mAP50"],
            "precision": ["metrics/precision(B)", "metrics/precision", "precision"],
            "recall": ["metrics/recall(B)", "metrics/recall", "recall"],
        }
        result: dict[str, Any] = {}
        for clean, keys in aliases.items():
            for k in keys:
                v = best.get(k)
                if v not in (None, ""):
                    try:
                        result[clean] = float(v)
                    except (ValueError, TypeError):
                        pass
                    break
        train_time = JobService.training_time_from_rows(rows)
        if train_time is not None:
            result["train_time_s"] = train_time
        return result

    def _merge_csv_fallback(self, metrics: dict[str, Any], job: Job) -> None:
        """Fill missing training metrics from the results.csv best row.

        The trainer's metrics.json reflects the final re-validation of
        best.pt and must never be overwritten by a per-epoch CSV row; this
        fallback only serves images/custom scripts that omit metrics.json so
        the job list can still show mAP50-95 / mAP50 and actual training time.
        """
        final = self._read_final_metrics(job)
        for k, v in final.items():
            metrics.setdefault(k, v)

    def finish_job(self, job: Job, status: str, progress: int, err_msg: str,
                   report: Optional[dict] = None) -> None:
        """Mark a job complete/failed and register its model to the hub."""
        metrics = self.read_metrics(job)
        # Fill any missing metric keys from the results.csv fallback.
        if status == "completed" and job.type == "training":
            self._merge_csv_fallback(metrics, job)
        # Gate report (optimization/evaluation) rides along in results
        report = report if report is not None else (
            self.read_gate_report(job) if job.type in ("optimization", "evaluation") else None
        )
        results: dict[str, Any] = {"message": "pipeline completed", "metrics": metrics}
        if report:
            results["report"] = report
            results["gate_result"] = report.get("gate_result", "")
        # Refresh the job object to ensure we have the latest params from DB
        fresh_job = self.db.get(Job, job.id)
        if not fresh_job:
            fresh_job = job
        self.jobs.set_progress(fresh_job.id, status, progress, err_msg, results=results, metrics=metrics)

        # Extract source_model_id before calling register_trained_model
        source_model_id = ""
        if fresh_job.params:
            source_model_id = str(fresh_job.params.get("model_id", "") or fresh_job.name)
        elif job.params:
            source_model_id = str(job.params.get("model_id", "") or job.name)
        else:
            source_model_id = job.name

        if fresh_job.type == "training" and status == "completed":
            registered = self.register_trained_model(fresh_job, metrics, source_model_id)
            if registered:
                # Persist the produced-model link on the job so the training
                # list can show/link the artifact without an N+1 model lookup.
                results["registered_model"] = registered
                self.jobs.set_progress(
                    fresh_job.id, status, progress, err_msg,
                    results=results, metrics=metrics,
                )
        elif fresh_job.type == "optimization" and status == "completed":
            registered = self.register_optimized_model(fresh_job, metrics)
            self._maybe_auto_package(fresh_job, metrics, report, registered)
        elif fresh_job.type == "notebook" and status == "completed":
            registered = self.register_notebook_model(fresh_job, metrics)
            if registered:
                results["registered_model"] = registered
                self.jobs.set_progress(
                    fresh_job.id, status, progress, err_msg,
                    results=results, metrics=metrics,
                )

    # File suffixes that identify a weight/checkpoint file rather than a model
    # name; they are stripped when deriving the source model name.
    _WEIGHT_SUFFIXES = (
        ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".bin", ".gguf",
    )

    @classmethod
    def _source_model_name(cls, job: Job) -> str:
        """Derive the Model Registry name of the model a job was trained from.

        Precedence: ``config.base_model`` -> ``params.base_model`` ->
        ``params.model_id`` -> job name. Weight filenames are reduced to the
        model stem (``ppyoloe_crn_l_300e_coco.pdparams`` -> ``ppyoloe``); a base-model URI uses its
        final directory (``s3://.../base-models/qwen3/`` -> ``qwen3``), or its
        parent directory when the URI ends on a weight file
        (``.../qwen3/model.safetensors`` -> ``qwen3``).
        """
        cfg = job.config or {}
        params = job.params or {}
        raw = str(
            cfg.get("base_model") or params.get("base_model")
            or params.get("model_id") or job.name or ""
        ).strip()
        if raw.startswith(("s3://", "http://", "https://")):
            from urllib.parse import urlparse

            path = urlparse(raw).path.rstrip("/")
            parts = [p for p in path.split("/") if p]
            # A registered-version checkpoint (.../models/<name>/<version>/
            # <file>.pt) belongs to the model two levels up, not to its
            # timestamped version directory.
            for i, part in enumerate(parts[:-1]):
                if part == "models" and i + 1 < len(parts):
                    raw = parts[i + 1]
                    break
            else:
                leaf = path.rsplit("/", 1)[-1]
                if leaf.lower().endswith(cls._WEIGHT_SUFFIXES) and "/" in path:
                    leaf = path.rsplit("/", 2)[-2]
                raw = leaf
        if raw.lower().endswith(cls._WEIGHT_SUFFIXES):
            for suffix in cls._WEIGHT_SUFFIXES:
                if raw.lower().endswith(suffix):
                    raw = raw[: -len(suffix)]
                    break
        return raw.strip() or str(job.name or "model")

    def _resolve_source_model_name(self, job: Job) -> tuple[str, str]:
        """Resolve the source model name plus the explicit base model id.

        When the job config/params carry ``base_model_id`` (set by the UI when
        the user picked a registered version), the name is taken from that
        model row directly: the checkpoint S3 path alone cannot identify the
        source model, because several registered models may reference the same
        physical weights. Returns ``(model_name, base_model_id)``.
        """
        cfg = job.config or {}
        params = job.params or {}
        base_model_id = str(
            cfg.get("base_model_id") or params.get("base_model_id") or ""
        ).strip()
        if base_model_id:
            base = self.models.get_model(job.tenant_id, base_model_id)
            if base and base.get("name"):
                return str(base["name"]), base_model_id
        return self._source_model_name(job), ""

    def register_trained_model(self, job: Job, metrics: dict[str, Any], source_model_id: str,
                               task_type_override: Optional[str] = None) -> dict[str, str] | None:
        """Register a trained model to PG + hub after successful training.

        Returns the produced model reference ``{"id", "name", "version"}`` so
        the caller can persist the link back onto the job.
        """
        # A trained model is versioned under the model it was trained from.
        # An explicit base_model_id (UI-picked registered version) wins and is
        # resolved against the model table; otherwise the base model name
        # (config.base_model, e.g. ppyoloe_crn_l_300e_coco.pdparams -> ppyoloe), params.model_id and
        # finally the job name are the fallbacks.
        model_name, resolved_base_id = self._resolve_source_model_name(job)
        if resolved_base_id:
            source_model_id = resolved_base_id
        else:
            source_model_id = source_model_id or model_name
        logger.info("register_trained_model: job.id=%s, job.params=%s, source_model_id=%s", job.id, job.params, source_model_id)
        task_type = task_type_override or str(
            (job.config or {}).get("task_type")
            or (job.params or {}).get("task_type")
            or (job.config or {}).get("model_type")
            or "detection"
        )
        is_llm = task_type in ("llm", "vlm")
        framework = "llm" if is_llm else "pytorch"
        hub_format = "safetensors" if is_llm else "onnx"
        metadata_framework = "llm" if is_llm else "pytorch"
        model = self.models.create_model(
            job.tenant_id, model_name, job.name, framework, task_type, "nereidforge-worker"
        )
        weights_uri = str(metrics.get("weights_uri", ""))
        onnx_uri = str(metrics.get("onnx_uri", ""))
        file_path = onnx_uri or weights_uri
        # Custom training scripts (I-11) may upload an artifact without
        # following the trainer's metrics.json contract. Fall back to scanning
        # the job's output directory so the produced model still registers.
        if not file_path:
            file_path = self._discover_artifact_uri(job)
            if file_path:
                if file_path.endswith(".onnx"):
                    onnx_uri = file_path
                    metrics["onnx_uri"] = file_path
                else:
                    weights_uri = file_path
                    metrics["weights_uri"] = file_path
                logger.info("register_trained_model: discovered artifact %s for job %s", file_path, job.id)
        # Version label equals the source job id: every registered version
        # maps 1:1 to its training job, which beats an opaque timestamp when
        # comparing runs. Re-completing the same job upserts that version in
        # place (create_version replaces by (model_id, version)).
        version = job.id
        import json
        metadata = {
            "weights_uri": weights_uri,
            "onnx_uri": onnx_uri,
            "source_model_id": source_model_id,
            "base_model_id": str(
                (job.config or {}).get("base_model_id")
                or (job.params or {}).get("base_model_id") or ""
            ),
            "base_model": str(
                (job.config or {}).get("base_model")
                or (job.params or {}).get("base_model") or ""
            ),
            # timm architecture id (resnet50, vit_base_patch16_224, ...);
            # needed to rebuild the network when continuing a classification
            # model from its state dict checkpoint.
            "arch": str(metrics.get("architecture") or metrics.get("model") or ""),
            "job_id": job.id,
            "job_name": job.name,
            "kfp_run_id": job.kfp_run_id,
            "dataset_id": str((job.params or {}).get("dataset_id", "")),
            # Dataset version the model was trained from; lets the Test-model
            # dialog preselect the exact image source for a model version.
            "dataset_version": str((job.params or {}).get("dataset_version", "")),
            "hardware": _trained_hardware(metrics, job),
            "framework": metadata_framework,
            "train_metrics": json.dumps(metrics, ensure_ascii=False) if isinstance(metrics, dict) else str(metrics),
        }
        size_bytes = self.s3.object_size(file_path)
        logger.info("register_trained_model: creating version for job %s, model %s, size_bytes=%d, metadata_keys=%s", job.id, model["id"], size_bytes, list(metadata.keys()))
        try:
            result = self.models.create_version(
                job.tenant_id, model["id"], version, job.name, file_path, size_bytes, metadata, "nereidforge-worker"
            )
            logger.info("register_trained_model: created version %s with source_model_id=%s", result.get("version"), result.get("metadata", {}).get("source_model_id"))
        except Exception as e:
            logger.error("create_version failed for job %s: %s", job.id, e)
            raise
        if file_path:
            try:
                # The hub requires string custom properties; serialize any
                # nested dict (e.g. train_metrics) to JSON to satisfy pydantic.
                hub_metadata = dict(metadata)
                if isinstance(hub_metadata.get("train_metrics"), dict):
                    hub_metadata["train_metrics"] = json.dumps(
                        hub_metadata["train_metrics"], ensure_ascii=False
                    )
                self.models.register_to_hub(
                    model_name, file_path, version, hub_format, hub_metadata
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("hub register trained model failed: %s", e)
        logger.info("registered trained model %s/%s", job.tenant_id, model_name)
        return {"id": str(model["id"]), "name": model_name, "version": version}

    def register_optimized_model(self, job: Job, metrics: dict[str, Any]) -> dict[str, Any]:
        """Register an optimized model version to PG + hub."""
        from app.services.jobs import register_optimized_artifact

        return register_optimized_artifact(self.models, job, metrics, actor="nereidforge-worker")

    def _maybe_auto_package(
        self,
        job: Job,
        metrics: dict[str, Any],
        report: Optional[dict[str, Any]],
        registered: dict[str, Any],
    ) -> None:
        """Auto-enqueue the precision edge package after a hard-gate PASS.

        One optimization job produces one precision artifact (fp16 method ->
        FP16 package; int8-static NNCF method -> INT8 package). Device never
        enters the decision: every package is verified on both CPU and GPU by
        the single edge-verified profile. A failed/missing gate, an FP32 graph
        artifact (not shipped) or an unresolved dataset never fails the
        completed optimization (solution D4; packaging stays retryable).
        """
        params = job.params or {}
        quant = str(params.get("quant_method") or metrics.get("quant_method") or "").lower()
        # The UI submits the VENDOR value ("intel"); only pipeline params get
        # normalized via normalize_target_device(). Normalize here too or every
        # vendor-submitted Intel job silently skips packaging (pit 182).
        raw_target = str(params.get("target_device") or metrics.get("target_device") or "")
        target_device = normalize_target_device(raw_target)
        if target_device != "openvino":
            logger.info(
                "optimization %s: no auto package for target_device=%s", job.id, raw_target
            )
            return
        # Precision packaging map: only the two shipped precisions auto-build.
        precision = {"fp16": "fp16", "int8-static": "int8", "int8": "int8"}.get(quant)
        if not precision:
            logger.info(
                "optimization %s: no auto package for quant_method=%s", job.id, quant
            )
            return
        # An explicit auto_package=false always opts out (default: auto).
        auto_raw = params.get("auto_package")
        if auto_raw is None:
            auto = True
        elif isinstance(auto_raw, str):
            auto = auto_raw.strip().lower() not in ("0", "false", "no")
        else:
            auto = bool(auto_raw)
        if not auto:
            return
        gate_result = str((report or {}).get("gate_result") or "").upper()
        if gate_result not in ("PASS", "YELLOW"):
            logger.info("optimization %s: skip auto package (gate=%s)", job.id, gate_result or "none")
            return
        try:
            out = self.packages.enqueue(
                job.tenant_id,
                # Registration returns both the mv-* version row id and the
                # parent m-* model_id; enqueue resolves the parent model.
                str(registered.get("model_id") or ""),
                str(registered.get("version") or ""),
                precision,
                "nereidforge-worker-auto",
            )
            logger.info(
                "optimization %s: auto-enqueued %s package task %s",
                job.id, precision, out["task_id"],
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("optimization %s: auto package skipped: %s", job.id, e)

    @staticmethod
    def _safe(s: str) -> str:
        """Sanitize a string for use in a version suffix."""
        return "".join(c if c.isalnum() else "-" for c in s.lower())

    # -- notebook (Kale) jobs --------------------------------------------------

    def _launch_notebook(self, job: Job) -> str:
        """Compile + submit a Kale notebook run; returns the KFP run id.

        The backend image embeds the kale CLI + the KFP SDK, so the worker
        stages the notebook from S3 and compiles/submits in-process through
        app.services.notebook_runtime (ADR-013 D3). Coordinates for the
        registered user are persisted on the job so the UI can link them.
        """
        from app.core.config import config as _cfg
        from app.services import notebook_runtime

        params = job.params or {}
        nb_uri = str(params.get("notebook_uri") or "")
        if not nb_uri.startswith("s3://"):
            raise ValueError("notebook_uri (s3://...) required")
        access_key, secret_key = require_s3_credentials(
            _cfg.storage.s3_access_key,
            _cfg.storage.s3_secret_key,
            scope="storage.s3",
        )
        filer = self.s3.s3_uri_to_filer_path(nb_uri)
        data = self.s3.read_object(filer)

        cfg = kfp_config_fn(self._db())()
        kfp_host = cfg.get("kfp_host") or _cfg.kfp.host
        if not kfp_host:
            raise ValueError("kfp host not configured")
        step_image = str(params.get("notebook_image") or (job.config or {}).get("notebook_image") or "")
        if not step_image:
            step_image = notebook_runtime.DEFAULT_STEP_IMAGE

        bucket = _cfg.storage.bucket or "nereidforge"
        s3_env = {
            "S3_ENDPOINT": (
                _cfg.storage.s3_cluster_endpoint
                or "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"
            ),
            "S3_ACCESS_KEY": access_key,
            "S3_SECRET_KEY": secret_key,
            "S3_BUCKET": bucket,
            "NF_JOB_ID": job.id,
            "NF_TENANT_ID": str(job.tenant_id),
            "NF_MODEL_DIR": f"s3://{bucket}/models/{job.id}",
        }
        result = notebook_runtime.compile_and_submit(
            job_id=job.id,
            tenant_id=str(job.tenant_id),
            nb_bytes=data,
            step_image=step_image,
            kfp_host=kfp_host,
            kfp_namespace=cfg.get("kfp_namespace") or "nereidforge",
            kfp_userid=cfg.get("kfp_userid") or "admin@nereidforge.local",
            s3_env=s3_env,
            kfp_experiment_id=cfg.get("kfp_experiment_id") or "",
        )
        # Persist the typed-artifact manifest so registration on success uses
        # the declared outputs instead of a largest-blob scan (ADR-013 D6).
        job.params = {**dict(job.params or {}), "_kale_outputs": result.get("outputs", [])}
        self.db.commit()
        return str(result["run_id"])

    def register_notebook_model(self, job: Job, metrics: dict[str, Any]) -> Optional[dict[str, str]]:
        """Register a model produced by a Kale notebook job (ADR-013 D6).

        The notebook uploads its produced artifact to the job's S3 model dir
        (``s3://<bucket>/models/{job.id}/``) and may write a ``metrics.json``
        there too. The typed-artifact manifest recorded at submit time tells
        us what the notebook declared; the bytes come from the model dir (its
        name/type metadata survives even though the content never leaves NF
        storage). No model file -> nothing is registered.
        """
        bucket = config.storage.bucket or "nereidforge"
        filer_path = f"/buckets/{bucket}/models/{job.id}"
        try:
            files = self.s3.walk(filer_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("register_notebook_model: walk failed for %s: %s", filer_path, exc)
            return None
        candidates = [
            f for f in files
            if not f.endswith((".metrics.json", ".csv", ".log", ".txt", ".json"))
            and "/tensorboard/" not in f
        ]
        if not candidates:
            logger.info("register_notebook_model: job %s produced no model artifact; skipped", job.id)
            return None
        declared_model = any(str(o.get("type", "")).lower().endswith("model") for o in (job.params or {}).get("_kale_outputs") or [])
        if declared_model:
            best = max(candidates, key=self.s3.object_size)
        else:
            # No typed Model output: rank like the training heuristic (scoped
            # to the notebook's own model dir only, never the whole bucket).
            def _rank(path: str) -> tuple[int, int]:
                score = 0
                if path.endswith(".onnx"):
                    score = 3
                elif path.endswith((".pt", ".pth", ".safetensors")):
                    score = 2
                elif path.endswith((".bin", ".h5", ".engine")):
                    score = 1
                return (score, self.s3.object_size(path))

            best = max(candidates, key=_rank)
        uri = "s3://" + best[len("/buckets/"):]
        metrics = dict(metrics or {})
        if not (str(metrics.get("weights_uri") or "") or str(metrics.get("onnx_uri") or "")):
            if uri.endswith(".onnx"):
                metrics["onnx_uri"] = uri
            else:
                metrics["weights_uri"] = uri
        source_model_id = str((job.params or {}).get("model_id", "") or job.name)
        task_type = str((job.params or {}).get("task_type", "") or "custom")
        try:
            return self.register_trained_model(
                job, metrics, source_model_id, task_type_override=task_type
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("register_notebook_model failed for job %s: %s", job.id, exc)
            return None

    # -- async tasks --------------------------------------------------------

    def process_async_tasks(self) -> None:
        """Process pending async tasks (import / preannotate / batch_accept)."""
        db = self._db()
        rows = db.scalars(
            select(AsyncTask).where(AsyncTask.status == "pending").order_by(AsyncTask.created_at).limit(5)
        ).all()
        for task in rows:
            try:
                self.process_async_task(task)
            except AsyncTaskCancelled:
                # User pressed stop: control_async_task already marked it
                # "cancelled"; leave it as-is (do not record as failed).
                logger.info("async task %s stopped by user", task.id)
            except Exception as e:  # noqa: BLE001
                logger.warning("async task %s failed: %s", task.id, e)
                if task.type == "model_deploy":
                    # Surface the reason on the model version too (red tag in
                    # the UI); fail_async_task records it on the task itself.
                    self._deploy_model_failure(task, task.params or {}, e)
                elif task.type == "model_undeploy":
                    self._undeploy_model_failure(task, task.params or {}, e)
                elif task.type == "model_package":
                    self.packages.mark_failed(task, str(e))
                self.dataset.fail_async_task(task.id, str(e))

    def sync_ml_backend_status(self) -> None:
        """Poll ml-backend health endpoints and update DB status."""
        db = self._db()
        backends = db.scalars(select(MLBackend)).all()
        for b in backends:
            if not b.url:
                continue
            try:
                import requests
                r = requests.get(f"{b.url}/health", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    status = "connected" if data.get("status") == "UP" else "error"
                else:
                    status = "error"
            except Exception:
                status = "unreachable"
            
            from datetime import datetime, timezone
            if b.status != status or b.last_health_check is None:
                logger.info("MLBackend %s status changed: %s -> %s", b.name, b.status, status)
            b.status = status
            b.last_health_check = datetime.now(timezone.utc)
            try:
                db.commit()
            except Exception:
                db.rollback()

    def process_async_task(self, task: AsyncTask) -> None:
        """Dispatch an async task by type."""
        self.dataset.set_async_task_progress(task.id, 5, "processing")
        params = task.params or {}
        if task.type == "dataset_import":
            self._import_dataset(task, params)
        elif task.type == "ls_import":
            self._run_ls_import(task, params)
        elif task.type == "preannotate":
            self._run_preannotate(task, params)
        elif task.type == "batch_accept":
            self._run_batch_accept(task, params)
        elif task.type == "model_deploy":
            self._deploy_model(task, params)
        elif task.type == "model_undeploy":
            self._undeploy_model(task, params)
        elif task.type == "model_package":
            self._package_model(task, params)
        else:
            self.dataset.fail_async_task(task.id, f"unknown task type: {task.type}")

    def _package_model(self, task: AsyncTask, params: dict) -> None:
        """Submit and watch the single-container package Batch Job.

        The API writes an immutable package-request JSON. The worker only
        selects the single edge-verified hardware profile and consumes the
        container's result JSON; OpenVINO, dataset traversal, decoder and OVMS
        logic never enter core.
        """
        spec_uri = str(params.get("request_uri") or "")
        profile = str(params.get("hardware_profile") or "edge-verified")
        if not spec_uri:
            raise RuntimeError("model package task is missing request URI")

        self.dataset.set_async_task_progress(task.id, 10, "processing")
        job_name = str(params.get("job_name") or "")
        if not job_name:
            job_name = f"pkg-{task.id}"
            job_name = self.package_jobs.submit(spec_uri, profile, job_name=job_name)
            self.packages.set_job_name(task.id, job_name)
        self.dataset.set_async_task_progress(task.id, 30, "processing")

        deadline = time.time() + 2400
        last_progress = 30
        while time.time() < deadline:
            if self._task_cancelled(task.id):
                self.package_jobs.delete(job_name)
                raise AsyncTaskCancelled("cancelled during package verification")
            state = self.package_jobs.get_state(job_name)
            if state.status == "succeeded":
                self.dataset.set_async_task_progress(task.id, 90, "processing")
                record = self.packages.record_result(task)
                if record.get("status") == "failed":
                    raise RuntimeError(record.get("error") or "package verification failed")
                logger.info(
                    "model_package %s ready: %s (%s detections, devices=%s)",
                    task.id, record.get("package_uri", params.get("package_uri")),
                    record.get("detection_count"),
                    [d.get("device") + ":" + d.get("status") for d in record.get("devices", [])],
                )
                self.dataset.complete_async_task(task.id)
                return
            if state.status == "failed":
                # A domain failure (numeric mismatch, bad IR, bad request) is
                # recorded in result JSON even though the container exits 2.
                try:
                    record = self.packages.record_result(task)
                except Exception:  # noqa: BLE001
                    detail = state.message or state.reason or "package job failed"
                    if state.log_tail:
                        detail = f"{detail}\n{state.log_tail}"
                    raise RuntimeError(detail) from None
                if record.get("status") == "failed":
                    raise RuntimeError(record.get("error") or "package verification failed")
                raise RuntimeError("package job exited without a successful result")
            last_progress = min(85, last_progress + 1)
            self.dataset.set_async_task_progress(task.id, last_progress, "processing")
            time.sleep(10)
        raise TimeoutError(f"package job {job_name} timed out after 2400s")

    def _task_cancelled(self, task_id: str) -> bool:
        """Re-read an async task state to detect a user cancel signal."""
        row = self._db().get(AsyncTask, task_id)
        return bool(row and row.status == "cancelled")

    def _deploy_model(self, task: AsyncTask, params: dict) -> None:
        """Run one async model deployment: ONNX conversion then KServe rollout.

        Both stages are long (a big .pt converts in 5-10 minutes; KServe can
        take several minutes to roll out), which is exactly why deploy runs as
        an async task instead of blocking the HTTP request. Cancellation is
        checked at every stage boundary so undeploy can stop a queued deploy.
        """
        from app.clients.k8s import InferenceServiceSpec, K8sService
        from app.services.model_format import detect_family, is_convertible

        tenant = str(params.get("tenant", "default"))
        model_id = str(params["model_id"])
        version = str(params["version"])
        ns = str(params["ns"])
        is_name = str(params["is_name"])
        runtime_type = str(params.get("runtime_type", "kserve-tritonserver"))
        runtime_image = str(params.get("runtime_image", ""))
        domain = str(params.get("domain", ""))
        ingress_class = str(params.get("ingress_class", "traefik"))
        model_uri = str(params["model_uri"])

        k8s = K8sService(ns)

        # Stage 1: transparent conversion to ONNX for convertible artifacts.
        if is_convertible(model_uri.rstrip("/").lower()):
            self.dataset.set_async_task_progress(task.id, 10, "processing")
            if self._task_cancelled(task.id):
                raise AsyncTaskCancelled("cancelled before conversion")
            onnx_uri = model_uri.rsplit(".", 1)[0] + ".onnx"
            source = str(params.get("source", "auto") or "auto").strip().lower()
            architecture = str(params.get("architecture", "") or "").strip()
            if not model_uri.startswith("s3://") or "/" not in model_uri[5:]:
                raise ValueError(
                    f"convertible artifact must be an s3:// URI, got {model_uri}"
                )
            bucket, key = model_uri[5:].split("/", 1)
            logger.info("model_deploy %s: converting %s (%s)", task.id, model_uri, source)
            family = detect_family(self.s3.client, bucket, key, source)
            k8s.convert_to_onnx(
                model_uri, onnx_uri, ns,
                family=family, source=source, architecture=architecture,
            )
            model_uri = onnx_uri
            # Persist the ONNX so trials and later redeploys reuse it instead
            # of re-running the long conversion.
            try:
                self.models.update_version_metadata(
                    tenant, model_id, version, {"onnx_uri": onnx_uri}
                )
            except Exception as e:  # noqa: BLE001 - metadata is non-fatal
                logger.warning("model_deploy %s: record onnx_uri failed: %s", task.id, e)
            if self._task_cancelled(task.id):
                raise AsyncTaskCancelled("cancelled after conversion")

        # Stage 2: create InferenceService + Traefik IngressRoute via KFP and
        # wait for KServe to report the predictor ready.
        self.dataset.set_async_task_progress(task.id, 50, "processing")
        spec = InferenceServiceSpec(
            name=is_name,
            namespace=ns,
            model_uri=model_uri,
            runtime_image=runtime_image,
            runtime_type=runtime_type,
            gpu=str(params.get("gpu", "cpu")),
            cpu_request="1",
            memory_request="2Gi",
            replicas=1,
            domain=domain,
            ingress_class=ingress_class,
        )
        logger.info("model_deploy %s: rolling out %s in %s", task.id, is_name, ns)
        from app.clients.k8s import (
            K8sServingUnavailable,
            K8sServingVerifyError,
            serving_reader,
        )

        try:
            out = k8s.create_inference_service(spec)
        except Exception:
            # The deploy pipeline can report FAILED after the component itself
            # succeeded — argo/containerd occasionally loses the main container
            # exit code ("main: Unknown (exit code 255)", pit 82/170). Before
            # failing the task, reconcile against the live InferenceService:
            # if KServe actually rolled it out and it is Ready, treat the
            # deploy as successful instead of stranding a good revision.
            if serving_reader.available:
                try:
                    serving_reader.wait_until_ready(is_name, ns, timeout=90, interval=5)
                except K8sServingUnavailable as e:
                    logger.warning("model_deploy %s: %s", task.id, e)
                    raise
                except K8sServingVerifyError as e:
                    logger.warning(
                        "model_deploy %s: pipeline failed and live check failed: %s",
                        task.id, e,
                    )
                    raise
                else:
                    logger.warning(
                        "model_deploy %s: pipeline reported failure but %s is Ready "
                        "(likely exit-code-255 flake); reconciling as deployed",
                        task.id, is_name,
                    )
                    self.models.mark_deployed(
                        tenant, model_id, version, is_name, runtime_type, domain
                    )
                    self.dataset.complete_async_task(task.id)
                    return
            raise
        logger.info("model_deploy %s: rollout done: %s", task.id, out[:200])

        # Stage 3: independent post-deploy verification. The KFP run saying
        # "succeeded" is not proof — a cached/skipped apply or a silently
        # deleted object leaves no ISVC behind (pit 170). Confirm the object
        # exists and is Ready from the cluster itself before flipping the DB.
        if serving_reader.available:
            try:
                serving_reader.wait_until_ready(is_name, ns, timeout=120, interval=5)
            except K8sServingUnavailable as e:
                # Reader dropped mid-rollout: keep the old trust-the-run
                # behavior rather than failing a deployment we could verify.
                logger.warning("model_deploy %s: %s", task.id, e)
            except K8sServingVerifyError:
                raise
        else:
            logger.warning(
                "model_deploy %s: kubernetes reader unavailable; skipped "
                "post-deploy verification for %s", task.id, is_name,
            )

        self.models.mark_deployed(tenant, model_id, version, is_name, runtime_type, domain)
        self.dataset.complete_async_task(task.id)

    def _undeploy_model(self, task: AsyncTask, params: dict) -> None:
        """Verify an undeploy submitted by the API actually removed the ISVC.

        The API keeps the click fast (submit KFP run wait=False, mark
        "terminating"); this task then polls the live object until a confirmed
        404 before clearing the DB row. A still-existing object after the
        deadline flips the version back to deployed with a visible error,
        instead of leaving DB and cluster silently divergent.
        """
        tenant = str((params or {}).get("tenant", "default"))
        model_id = str((params or {}).get("model_id", ""))
        version = str((params or {}).get("version", ""))
        is_name = str((params or {}).get("service", ""))
        ns = str((params or {}).get("namespace", "nereidforge"))
        run_id = str((params or {}).get("run_id", ""))

        from app.clients import cluster_ops
        from app.clients.k8s import serving_reader

        self.dataset.set_async_task_progress(task.id, 30, "processing")
        if not serving_reader.available:
            # Nothing can verify the deletion; trust the KFP run like the old
            # fire-and-forget path so environments without a reader keep working.
            logger.warning(
                "model_undeploy %s: kubernetes reader unavailable; trusting run %s",
                task.id, run_id,
            )
            self.models.mark_undeployed(tenant, model_id, version)
            self.dataset.complete_async_task(task.id)
            return

        deadline_s = 300
        import time

        deadline = time.time() + deadline_s
        last_presence: bool | None = None
        while time.time() < deadline:
            if self._task_cancelled(task.id):
                raise AsyncTaskCancelled("cancelled during undeploy verification")
            last_presence = serving_reader.isvc_presence(is_name, ns)
            if last_presence is False:
                logger.info(
                    "model_undeploy %s: confirmed %s/%s removed", task.id, ns, is_name
                )
                self.models.mark_undeployed(tenant, model_id, version)
                self.dataset.complete_async_task(task.id)
                return
            # Fail fast when the deletion pipeline itself failed.
            if run_id:
                state, err = cluster_ops.get_run_state(run_id)
                if state in ("FAILED", "CANCELED", "CANCELLED", "ERROR"):
                    raise RuntimeError(
                        f"undeploy pipeline run {state}: {err or 'unknown error'}"
                    )
            time.sleep(5)
        if last_presence is None:
            raise RuntimeError(
                f"could not verify undeploy of {ns}/{is_name}: kubernetes API "
                f"unreachable for {deadline_s}s"
            )
        raise RuntimeError(
            f"undeploy pipeline reported success but InferenceService "
            f"{ns}/{is_name} still exists after {deadline_s}s"
        )

    def _undeploy_model_failure(
        self, task: AsyncTask, params: dict, err: Exception
    ) -> None:
        """Record a failed undeploy on the version (service still present)."""
        tenant = str((params or {}).get("tenant", "default"))
        model_id = str((params or {}).get("model_id", ""))
        version = str((params or {}).get("version", ""))
        if model_id and version:
            try:
                self.models.mark_undeploy_failed(tenant, model_id, version, str(err))
            except Exception:  # noqa: BLE001 - never mask the original error
                logger.exception(
                    "model_undeploy %s: failed to record failure state", task.id
                )

    def _deploy_model_failure(self, task: AsyncTask, params: dict, err: Exception) -> None:
        """Record a failed deployment on the version (called from the catch)."""
        tenant = str((params or {}).get("tenant", "default"))
        model_id = str((params or {}).get("model_id", ""))
        version = str((params or {}).get("version", ""))
        if model_id and version:
            try:
                self.models.mark_deploy_failed(tenant, model_id, version, str(err))
            except Exception:  # noqa: BLE001 - never mask the original error
                logger.exception("model_deploy %s: failed to record failure state", task.id)

    def _import_dataset(self, task: AsyncTask, params: dict) -> None:
        """Import uploaded files into a dataset draft on SeaweedFS.

        Files are staged in S3 (prefix s3://{bucket}/.uploads/{id}/{name})
        by the upload API; `file_paths` entries are therefore S3 URIs (plain
        local paths are still supported for hosts sharing a filesystem, e.g.
        a dev box running API + worker on the same machine).
        """
        dataset_id = str(params.get("dataset_id", ""))
        dest_uri = str(params.get("dest_uri", ""))
        file_paths = params.get("file_paths") or []
        imported = 0
        errors: list[str] = []
        warnings: list[str] = []
        filer = self.s3.s3_uri_to_filer_path(dest_uri) if dest_uri else f"/buckets/{self.s3.s3_bucket()}/{dataset_id}"
        total = max(1, len(file_paths))
        for i, fp in enumerate(file_paths):
            try:
                if str(fp).startswith("s3://"):
                    data = self.s3.read_object(self.s3.s3_uri_to_filer_path(str(fp)))
                else:
                    with open(fp, "rb") as f:
                        data = f.read()
                name = str(fp).rsplit("/", 1)[-1]
                lower = name.lower()
                if lower.endswith(".zip"):
                    # External corpora: detect PP-YOLOE/COCO/VOC/raw inside the
                    # archive, normalize into the flat draft layout, and keep
                    # the original zip in a separate prefix for traceability.
                    from app.services.dataset_zip import normalize_dataset_zip

                    result = normalize_dataset_zip(data)
                    bucket = self.s3.s3_bucket()
                    archive_rel = f"dataset-archives/{dataset_id}/{name}"
                    if task.tenant_id != "default":
                        archive_rel = f"{task.tenant_id}/{archive_rel}"
                    self.s3.put_object(f"/buckets/{bucket}/{archive_rel}", data)
                    for obj_key, payload in result["files"].items():
                        self.s3.put_object(f"{filer}/{obj_key}", payload)
                    if result.get("warning"):
                        warnings.append(f"{name}: {result['warning']}")
                    imported += 1
                    continue
                if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    sub = "/images"
                elif lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v")):
                    sub = "/videos"
                elif lower.endswith((".txt", ".json", ".jsonl", ".csv", ".md")):
                    # text datasets (NLP): raw docs, sharegpt/JSONL, CSV
                    sub = "/texts"
                else:
                    sub = ""
                self.s3.put_object(f"{filer}{sub}/{name}", data)
                imported += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("import file %s failed: %s", fp, e)
                errors.append(f"{str(fp).rsplit('/', 1)[-1]}: {e}")
            finally:
                self.dataset.set_async_task_progress(task.id, 5 + int(i * 90 / total), "processing")
        # Best-effort cleanup of the S3 staging prefix written by the upload API.
        try:
            staging_uri = file_paths[0].rsplit("/", 1)[0] if file_paths else ""
            if str(staging_uri).startswith("s3://"):
                for obj in self.s3.walk(self.s3.s3_uri_to_filer_path(staging_uri)):
                    self.s3.delete_object(obj)
        except Exception:
            logger.warning("staging cleanup failed", exc_info=True)
        if imported == 0:
            detail = f" ({errors[0]})" if errors else ""
            self.dataset.fail_async_task(task.id, f"no files imported{detail}")
            return
        # Update draft or dataset file_count and total size from real object
        # sizes (a single paginated LIST) instead of a per-file placeholder.
        db = self._db()
        draft = db.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.tenant_id == task.tenant_id,
                DatasetVersion.is_draft.is_(True),
            )
        )
        if draft:
            # Re-run content detection so the draft is immediately shown as a
            # trainable ppyoloe/coco/llm-sft version once the import finishes.
            self.dataset._apply_content_detection(draft)
            db.commit()
        self.dataset.complete_async_task(task.id)
        if warnings:
            logger.info("dataset import %s warnings: %s", task.id, "; ".join(warnings))
            self.dataset.merge_async_task_params(task.id, {"warnings": warnings})

    def _run_ls_import(self, task: AsyncTask, params: dict) -> None:
        """Build and import LS tasks for a large version, in batches.

        Scheduled by ``_provision_ls_project`` when a version has at least
        ``ls_dataset.ASYNC_THRESHOLD`` images. Task conversion (labels ->
        saved annotations) runs once, then tasks are POSTed in fixed batches;
        progress and the pause/cancel gate are honored between batches.
        """
        ls_id = int(params.get("ls_id", 0))
        dataset_id = str(params.get("dataset_id", ""))
        dataset_version = int(params.get("dataset_version", 1))
        schema = params.get("schema") or {}
        task_type = str(params.get("task_type", ""))
        if not ls_id or not dataset_id:
            self.dataset.fail_async_task(task.id, "ls_import task missing ls_id/dataset_id")
            return
        self.dataset.set_async_task_progress(task.id, 5, "processing")
        tasks = self.project.build_tasks_from_dataset(
            task.tenant_id, dataset_id, dataset_version, dict(schema), task_type)
        if not tasks:
            self.dataset.fail_async_task(task.id, "no tasks built from dataset version")
            return
        # Start from a clean LS project; the in-request path never imported.
        self.ls.delete_all_tasks(ls_id)
        batch_size = 100
        total = len(tasks)
        for start in range(0, total, batch_size):
            self.dataset.async_task_control_gate(task.id)
            self.ls.import_tasks(ls_id, tasks[start:start + batch_size])
            done = min(start + batch_size, total)
            self.dataset.set_async_task_progress(
                task.id, 10 + int(done * 85 / total), "processing")
        # Refresh the NF project's item counts after import.
        try:
            from app.models.orm import AnnotationProject

            row = self._db().scalar(select(AnnotationProject).where(
                AnnotationProject.external_id == str(ls_id)))
            if row:
                self.project.sync_progress(task.tenant_id, row.id)
        except Exception:
            logger.warning("ls_import %s: progress sync failed", task.id, exc_info=True)
        self.dataset.complete_async_task(task.id)
        logger.info("ls_import %s: imported %s tasks into LS project %s",
                    task.id, total, ls_id)

    def _run_preannotate(self, task: AsyncTask, params: dict) -> None:
        """Run ML backend pre-annotation for a project."""
        def set_progress(tid: str, p: int, st: str) -> None:
            self.dataset.set_async_task_progress(tid, p, st)

        def gate() -> None:
            # Polls params["_control"]; blocks while paused, raises on stop.
            self.dataset.async_task_control_gate(task.id)

        processed, predicted, total = self.project.run_preannotate_task(
            task.tenant_id, task.id, params, set_progress, gate
        )
        self.dataset.complete_async_task(task.id)
        self.dataset.merge_async_task_params(task.id, {
            "result_total": total,
            "result_processed": processed,
            "result_predicted": predicted,
        })
        logger.info("preannotate done: processed=%s predicted=%s total=%s", processed, predicted, total)

    def _run_batch_accept(self, task: AsyncTask, params: dict) -> None:
        """Convert predictions to annotations for a project."""
        def set_progress(tid: str, p: int, st: str) -> None:
            self.dataset.set_async_task_progress(tid, p, st)

        def gate() -> None:
            self.dataset.async_task_control_gate(task.id)

        processed, accepted, empty = self.project.run_batch_accept_task(
            task.tenant_id, task.id, params, set_progress, gate
        )
        self.dataset.complete_async_task(task.id)
        self.dataset.merge_async_task_params(task.id, {
            "result_processed": processed,
            "result_accepted": accepted,
            "result_empty": empty,
        })
        logger.info("batch accept done: processed=%s accepted=%s empty=%s", processed, accepted, empty)


def s3_client() -> SeaweedFSClient:
    """Return a fresh S3 client (import guard for module-level singletons)."""
    from app.clients.s3 import SeaweedFSClient as _S3

    return _S3()


def main() -> None:
    """Entry point: run the worker until interrupted."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    Worker().run()


if __name__ == "__main__":
    main()
