# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Job service: training / optimization / evaluation jobs with KFP submission.

Mirrors the Go JobService (internal/service/jobs.go):
- job CRUD with status/progress tracking
- StartJob builds pipeline params and submits a KFP v2 run
- training writes a structured train_config.json to SeaweedFS for the TrainJob
- resolves training data URIs and generates data.yaml when needed
- reads training curves from SeaweedFS results.csv
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
from typing import Any, Optional

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.clients.kfp import KFPClient, PipelineRunSpec
from app.clients.s3 import SeaweedFSClient
from app.core.config import config, require_s3_credentials
from app.core.images import nf_image
from app.core.util import iso, new_id, page_bounds
from app.models.orm import Job, ModelVersion
from app.services.accelerators import ACCELERATOR_PROFILES, get_profile, resolve_vendor
from app.services.base_models import infer_stock_arch
from app.services.images import canonical_trainer_ref
from app.services.model_format import artifact_capability_permitted

logger = logging.getLogger(__name__)

JOB_TYPES = ("training", "optimization", "evaluation", "notebook")

# Terminal progress owns completion timestamps; retries clear them in start_job.
_TERMINAL_STATUSES = frozenset(("completed", "failed", "cancelled"))

# Only terminal unsuccessful jobs may be restarted. Completed jobs keep their
# registered artifact versions immutable; active jobs retain the start guard.
_RESTARTABLE_STATUSES = frozenset(("cancelled", "failed"))


def _json_safe(value: Any) -> Any:
    """_json_safe converts non-finite floats to null at JSON integration boundaries.

    Python's JSON encoder emits NaN and Infinity by default, but PostgreSQL JSONB,
    browsers and strict JSON consumers reject those tokens. Keeping the containing
    row preserves epoch alignment while marking the individual metric unavailable.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


# Suffix convention for optimized artifact versions: {source_version}-{suffix}.
# ONNX-family targets keep the historic "-onnx-*" names; OpenVINO produces IR
# (.xml/.bin) so its versions are named "-ir-*".
OPT_VERSION_SUFFIX = {
    "fp16": "fp16",
    "int8-dynamic": "int8-dyn",
    "int8-static": "int8",
    "graph": "graph",
}
OPENVINO_IR_SUFFIX = {
    "fp16": "ir-fp16",
    "int8-dynamic": "ir-int8-dyn",
    "int8-static": "ir-int8",
    "graph": "ir-fp32",
}

# Vendor-first hardware model: a target vendor maps to the runtime/export path
# the optimizer image uses. Values are passed through to optimize.py as
# `target_device` and drive format + benchmark decisions there.
VENDOR_RUNTIME = {
    "intel": "openvino",
    "nvidia": "tensorrt",
    "amd": "onnx-rocm",
    "huawei": "ascend",
}


def normalize_target_device(v: Any) -> str:
    """Normalize a user-supplied target (vendor or legacy runtime) to the
    optimize.py device string. Legacy values (cpu/openvino/tensorrt) pass
    through unchanged; vendor values are mapped to their runtime."""
    s = str(v or "").strip().lower()
    if not s:
        return "cpu"
    if s in VENDOR_RUNTIME:
        return VENDOR_RUNTIME[s]
    return s


def register_optimized_artifact(models_svc: Any, job: Job, metrics: dict,
                                report: Optional[dict] = None,
                                actor: str = "nereidforge-worker") -> dict:
    """Register an optimization artifact as a new model version.

    Shared by the worker (gate PASS/YELLOW) and the gate-override endpoint.
    Version naming: {source_version}-onnx-{suffix} (ONNX-family targets) or
    {source_version}-ir-{fp32|fp16|int8} (OpenVINO IR); falls back to a
    timestamped name when the source version is unknown.
    """
    from datetime import datetime, timezone

    params = job.params or {}
    target_device = normalize_target_device(params.get("target_device"))
    is_ir = target_device == "openvino"
    hub_format = "openvino" if is_ir else "onnx"
    source_model_id = str(params.get("source_model_id") or params.get("model_id") or "")
    model = models_svc.get_model(job.tenant_id, source_model_id) if source_model_id else None
    if not model:
        model = models_svc.create_model(
            job.tenant_id, source_model_id or job.name, job.name,
            hub_format, "detection", actor,
        )
    opt_uri = str(metrics.get("optimized_uri", ""))
    if opt_uri.startswith("s3://"):
        # Pipeline param trailing slashes used to yield "s3://b/opt//id/id.xml".
        # S3 keys have no meaningful empty segments; collapse them.
        opt_uri = "s3://" + opt_uri[len("s3://"):].replace("//", "/")
    quant_method = str(params.get("quant_method") or "")
    if is_ir:
        # Suffix already carries the ir- prefix, e.g. ir-fp32 / ir-int8.
        suffix = OPENVINO_IR_SUFFIX.get(quant_method, f"ir-{quant_method or 'fp32'}")
        version_tail = suffix
    else:
        suffix = OPT_VERSION_SUFFIX.get(quant_method, quant_method or target_device)
        version_tail = f"onnx-{suffix}"
    src_ver = str(params.get("source_version") or "")
    version = (
        f"{src_ver}-{version_tail}" if src_ver
        else datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-opt-{target_device}"
    )
    # Multiple jobs optimizing the same source version would collide on the
    # same version string; append a timestamp suffix when the name is taken.
    if source_model_id:
        try:
            existing = models_svc.db.scalar(
                select(ModelVersion).where(and_(
                    ModelVersion.model_id == source_model_id,
                    ModelVersion.version == version,
                ))
            )
            if existing:
                version = f"{version}-{datetime.now(timezone.utc).strftime('%H%M%S')}"
        except Exception:
            pass
    metadata = {
        "optimized_uri": opt_uri,
        "weights_uri": opt_uri,
        "source_version": src_ver,
        "source_model_id": source_model_id,
        "target_device": target_device,
        "quant_method": quant_method,
        "dtype": str(params.get("dtype", "") or ""),
        "job_id": job.id,
        "kfp_run_id": job.kfp_run_id,
        "optimizer": str(metrics.get("optimizer", "")),
        "speedup_ratio": metrics.get("speedup_ratio"),
        "framework": hub_format,
        "gate_result": (report or {}).get("gate_result", ""),
        "opt_metrics": metrics,
    }
    # IR ships as an .xml graph plus a same-basename .bin weights file; surface
    # the companion explicitly so downloads/deploys never lose the weights.
    def _norm(u: str) -> str:
        return ("s3://" + u[len("s3://"):].replace("//", "/")) if u.startswith("s3://") else u

    raw_extra = metrics.get("extra_artifacts") or []
    if isinstance(raw_extra, str):
        raw_extra = [raw_extra]
    extra = [_norm(u) for u in raw_extra if str(u or "").startswith("s3://")]
    if is_ir:
        from app.services.model_format import ir_companion_bin

        companion = ir_companion_bin(opt_uri)
        metadata["ir_bin_uri"] = companion
        # Build the set from a list of complete URIs. Never star-unpack a URI
        # string: that iterates characters and records '-', '.', '/', ... as
        # bogus artifact paths.
        artifact_uris = [*extra, companion] if companion else extra
        metadata["extra_artifacts"] = sorted({u for u in artifact_uris if u})
    else:
        metadata["extra_artifacts"] = sorted(set(extra))
    size = 0
    try:
        sizes = [models_svc.s3.object_size(opt_uri)] if hasattr(models_svc, "s3") else []
        if is_ir and metadata.get("ir_bin_uri"):
            try:
                sizes.append(models_svc.s3.object_size(metadata["ir_bin_uri"]))
            except Exception:
                pass
        size = sum(sizes)
    except Exception:
        size = 0
    out = models_svc.create_version(
        job.tenant_id, model["id"], version, job.name, opt_uri, size, metadata, actor
    )
    if opt_uri:
        try:
            models_svc.register_to_hub(model["name"], opt_uri, version, hub_format, metadata)
        except Exception:
            pass
    return {"model_id": model["id"], "version": version, **out}


def _job_dict(j: Job) -> dict[str, Any]:
    """Serialize a Job to the API response shape (flattened view)."""
    params = j.params or {}
    config = j.config or {}
    results = j.results if isinstance(j.results, dict) else {}
    total_epochs = int(config.get("epochs") or 0) or None
    metrics = j.metrics if isinstance(j.metrics, dict) else {}
    current_epoch = int(metrics.get("current_epoch") or metrics.get("final_epoch") or 0) or None
    if current_epoch is None and j.status == "running" and total_epochs:
        current_epoch = 0
    # The produced model is recorded by the worker after auto-registration
    # (results.registered_model); model_id in params is the *source/base* model.
    registered = results.get("registered_model")
    registered = registered if isinstance(registered, dict) else {}
    return {
        "id": j.id,
        "tenant_id": j.tenant_id,
        "type": j.type,
        "name": j.name,
        "description": j.description,
        "batch_id": j.batch_id,
        "status": j.status,
        "progress": j.progress,
        "params": params,
        "config": config,
        "results": results,
        "metrics": j.metrics,
        "logs": j.logs,
        "kfp_run_id": j.kfp_run_id,
        "error_message": j.error_message,
        "created_by": j.created_by,
        "created_at": iso(j.created_at),
        "updated_at": iso(j.updated_at),
        "started_at": iso(j.started_at),
        "completed_at": iso(j.completed_at),
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "trainer_image": str((config or {}).get("trainer_image") or ""),
        # Flattened refs used by the UI list columns.
        "model_id": str(params.get("model_id") or params.get("source_model_id") or ""),
        # Produced artifact (auto-registered on completion); empty until then.
        "output_model_id": str(registered.get("id") or ""),
        "output_model_name": str(registered.get("name") or ""),
        "output_model_version": str(registered.get("version") or ""),
        "dataset_id": str(params.get("dataset_id") or ""),
        "dataset_version": int(params.get("dataset_version") or 0) or None,
    }


class OptimizationSourceError(ValueError):
    """Raised when an optimization job's source version cannot be consumed (IR-as-source)."""


class JobService:
    """Manages training/optimization/evaluation jobs and their KFP runs."""

    def __init__(
        self,
        db: Session,
        s3: SeaweedFSClient,
        kfp: Optional[KFPClient] = None,
        dataset: Any = None,
        trainer_config_fn: Any = None,
        models: Any = None,
    ) -> None:
        # Validate at service construction so no database Job can be created through
        # the normal DI path with credentials that would fail only inside a Pod.
        require_s3_credentials(
            config.storage.s3_access_key,
            config.storage.s3_secret_key,
            scope="storage.s3",
        )
        self.db = db
        self.s3 = s3
        self.kfp = kfp
        self.dataset = dataset
        self.trainer_config_fn = trainer_config_fn
        self.models = models
        # Set by callers that know the tenant (profile lookups are tenant-scoped).
        self.tenant_id = "default"

    # -- CRUD ---------------------------------------------------------------

    def create_job(
        self, job_type: str, tenant_id: str, name: str, description: str,
        created_by: str, params: dict, cfg: dict,
    ) -> dict:
        """Create a job (pending status)."""
        job = Job(
            id=new_id("job"),
            tenant_id=tenant_id,
            type=job_type,
            name=name,
            description=description or "",
            status="pending",
            progress=0,
            params=params or {},
            config=cfg or {},
            created_by=created_by or "anonymous",
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return _job_dict(job)

    def create_job_with_batch(
        self, job_type: str, tenant_id: str, name: str, description: str,
        batch_id: str, created_by: str, params: dict, cfg: dict,
    ) -> dict:
        """Create a job with a shared batch_id (optimization multi-target)."""
        job = Job(
            id=new_id("job"),
            tenant_id=tenant_id,
            type=job_type,
            name=name,
            description=description or "",
            batch_id=batch_id,
            status="pending",
            progress=0,
            params=params or {},
            config=cfg or {},
            created_by=created_by or "anonymous",
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return _job_dict(job)

    def list_jobs(
        self, job_type: str, tenant_id: str, page: int, page_size: int,
        status: str = "", model_id: str = "", search: str = "",
    ) -> tuple[list[dict], int]:
        """List jobs of a type with optional filters and pagination."""
        from sqlalchemy import func

        query = select(Job).where(and_(Job.type == job_type, Job.tenant_id == tenant_id))
        count_q = select(func.count(Job.id)).where(and_(Job.type == job_type, Job.tenant_id == tenant_id))
        if status:
            query = query.where(Job.status == status)
            count_q = count_q.where(Job.status == status)
        if model_id:
            query = query.where(Job.params["model_id"].astext == model_id)
            count_q = count_q.where(Job.params["model_id"].astext == model_id)
        if search:
            query = query.where(Job.name.ilike(f"%{search.strip()}%"))
            count_q = count_q.where(Job.name.ilike(f"%{search.strip()}%"))
        total = self.db.scalar(count_q) or 0
        limit, offset = page_bounds(page, page_size)
        rows = self.db.scalars(query.order_by(Job.created_at.desc()).limit(limit).offset(offset)).all()
        return [_job_dict(j) for j in rows], total

    def get_job(self, job_type: str, tenant_id: str, id: str) -> Optional[Job]:
        """Fetch a Job row by type/tenant/id."""
        return self.db.scalar(
            select(Job).where(and_(Job.id == id, Job.tenant_id == tenant_id, Job.type == job_type))
        )

    def get_job_dict(self, job_type: str, tenant_id: str, id: str) -> Optional[dict]:
        """Get a job as an API dict."""
        job = self.get_job(job_type, tenant_id, id)
        return _job_dict(job) if job else None

    def update_job(self, job_type: str, tenant_id: str, id: str, patch: dict[str, Any]) -> Optional[dict]:
        """Update job fields from a patch dict."""
        job = self.get_job(job_type, tenant_id, id)
        if not job:
            return None
        for k, v in patch.items():
            # Custom training selectors live in job.params, not as Job columns.
            if k in ("custom_image", "custom_script_uri"):
                p = dict(job.params or {})
                if v:
                    p[k] = v
                else:
                    p.pop(k, None)
                job.params = p
                continue
            if hasattr(job, k) and k not in ("id", "created_at"):
                setattr(job, k, v)
        self.db.commit()
        self.db.refresh(job)
        return _job_dict(job)

    def delete_job(self, job_type: str, tenant_id: str, id: str) -> bool:
        """Delete a job by id.

        For training jobs this also removes the cluster-side workload
        (Kubeflow TrainJob CR -> JobSet -> trainer pods), which is a separate
        object graph that survives both the KFP run and this DB row. Cluster
        cleanup runs first: on failure the row is kept so the UI action can be
        retried instead of orphaning the TrainJob.
        """
        job = self.get_job(job_type, tenant_id, id)
        if not job:
            return False
        if job_type == "training":
            self._delete_cluster_training(job)
        self.db.delete(job)
        self.db.commit()
        return True

    def _delete_cluster_training(self, job: Job) -> list[str]:
        """Terminate the KFP run (if active) and delete the job's TrainJob CRs.

        TrainJobs are labeled nereidforge/artifact-id=<job id> by train_step.
        Deleting the CR garbage-collects the JobSet, per-node Jobs and pods.
        """
        client = self._train_log_client()
        if not client.available:
            logger.warning("k8s client unavailable; skipping TrainJob cleanup for %s", job.id)
            return []
        if self.kfp and job.kfp_run_id and job.status in ("running", "pending", ""):
            try:
                self.kfp.cancel_run(job.kfp_run_id)
            except Exception as e:  # noqa: BLE001 - best-effort, CR deletion stops the pods
                logger.warning("terminate run %s failed: %s", job.kfp_run_id, e)
        deleted = client.delete_trainjobs(job.id)
        logger.info("deleted cluster TrainJobs for job %s: %s", job.id, deleted or "none")
        return deleted

    def training_runtime(self, tenant_id: str, job_id: str) -> dict | None:
        """Return cluster-side runtime objects (TrainJob CRs, trainer pods).

        Lets the UI map an NF job id to the objects visible in kubectl:
        TrainJob `train-<kfp run id>` and its JobSet pods.
        """
        job = self.get_job("training", tenant_id, job_id)
        if not job:
            return None
        client = self._train_log_client()
        trainjobs: list[dict] = []
        pods: list[dict] = []
        if client.available:
            try:
                trainjobs = client.find_trainjobs(job.id)
            except Exception as e:  # noqa: BLE001
                logger.warning("find trainjobs for %s failed: %s", job.id, e)
            try:
                pods = client.find_trainer_pods(job.id)
            except Exception as e:  # noqa: BLE001
                logger.warning("find trainer pods for %s failed: %s", job.id, e)
        return {
            "kfp_run_id": job.kfp_run_id or "",
            "trainjobs": trainjobs,
            "pods": pods,
        }

    # -- lifecycle ----------------------------------------------------------

    def start_job(self, job_type: str, tenant_id: str, id: str) -> dict:
        """Start a job, or retry a cancelled/failed job with a fresh KFP run."""
        job = self.get_job(job_type, tenant_id, id)
        if not job:
            raise ValueError("job not found")
        # Idempotency guard: repeated /start calls (double-click, stale UI
        # state) must not submit a second KFP run and orphan the first one.
        # Terminal unsuccessful jobs are the exception: restart_job delegates
        # here, so this owner must clear the stale run state before resubmitting
        # (pit 195). Completed jobs remain immutable to protect registered
        # artifact versions, and active jobs keep their existing run.
        is_retry = job.status in _RESTARTABLE_STATUSES
        if job.kfp_run_id and not is_retry:
            return _job_dict(job)
        if is_retry:
            job.kfp_run_id = ""
            job.error_message = ""
            job.results = {}
            job.metrics = {}
            job.logs = ""
            job.progress = 0
            job.started_at = None
            job.completed_at = None
        job.status = "pending"
        if job_type == "notebook":
            # Kale notebooks compile into their own pipeline at run time;
            # the worker's _launch_notebook handles submission.
            job.status = "pending"
            self.db.commit()
            return self._job_dict(job_type, job) if hasattr(self, "_job_dict") else {"id": job.id, "status": job.status}

        if self.kfp:
            params = dict(job.params or {})
            batch_jobs: list[Job] = [job]
            if job_type == "optimization":
                # Revalidate every start/retry before any launch branch, including
                # Katib, so stale jobs cannot bypass artifact capabilities.
                self.validate_optimization_source(tenant_id, params)
                # Katib search mode: run the grid search as a KFP pipeline
                # (the pipeline pod creates/watches the Katib Experiment);
                # the worker polls the KFP run and registers the best trial.
                if str(params.get("search_mode") or "").lower() in ("1", "true", "yes"):
                    run_id = self._launch_katib_search(tenant_id, params, job)
                    job.status = "running"
                    job.progress = 0
                    job.kfp_run_id = run_id
                    job.error_message = ""
                    job.started_at = self._now()
                    self.db.commit()
                    self.db.refresh(job)
                    return _job_dict(job)
                # Batch semantics: one run for every pending sibling sharing
                # the batch_id (single baseline + one branch per target).
                if job.batch_id:
                    rows = self.db.scalars(select(Job).where(and_(
                        Job.tenant_id == tenant_id,
                        Job.batch_id == job.batch_id,
                        Job.type == "optimization",
                        Job.status.in_(("pending", "")),
                    ))).all()
                    batch_jobs = list(rows) or [job]
                params = self.optimization_pipeline_params(tenant_id, params, job, batch_jobs)
            if job_type == "training":
                params = self.training_pipeline_params(tenant_id, params, job)
                # Pre-execution recheck (defense in depth): the entitlement can
                # have been revoked between job creation and worker pickup.
                self.check_capability(tenant_id, params, dict(job.config or {}))
            elif job_type == "evaluation":
                params = self.evaluation_pipeline_params(tenant_id, params, job)
            # Persist the resolved trainer image onto the job so the UI list can
            # show which image is/was used for this job. training_pipeline_params
            # already resolved the right engine image (llm/timm/yolo/custom)
            # into params["trainer_image"]; only fall back to the system
            # default when it did not.
            if job_type == "training":
                cfg = dict(job.config or {})
                cfg["trainer_image"] = (
                    str(params.get("trainer_image") or "")
                    or str(params.get("custom_image") or "")
                    or self._default_trainer_image()
                )
                # Persist the custom script URI (I-11) so the UI can restore it
                # when cloning/retraining the job.
                if params.get("custom_script_uri"):
                    cfg["custom_script_uri"] = str(params.get("custom_script_uri"))
                job.config = cfg
            result = self.kfp.submit_run(PipelineRunSpec(
                pipeline_name=f"{job_type}-pipeline",
                display_name=job.name,
                parameters={k: str(v) for k, v in params.items()},
                user_id=self.tenant_kfp_userid(tenant_id),
                experiment_id=self.tenant_kfp_experiment(tenant_id),
            ))
            job.status = "running"
            # The run is only submitted; progress must stay 0 until the worker
            # observes real run/pipeline progress (a fixed 5 here made every
            # freshly started job show 5% before any work happened).
            job.progress = 0
            job.kfp_run_id = result.run_id
            job.started_at = self._now()
            # Batch siblings ride the same run; the worker finishes each one
            # from its own per-target artifacts.
            if len(batch_jobs) > 1:
                for sib in batch_jobs:
                    if sib.id == job.id:
                        continue
                    sib.status = "running"
                    sib.progress = 0
                    sib.kfp_run_id = result.run_id
                    sib.started_at = job.started_at
        else:
            job.status = "running"
            job.progress = 0
            job.started_at = self._now()
        self.db.commit()
        self.db.refresh(job)
        return _job_dict(job)

    def stop_job(self, job_type: str, tenant_id: str, id: str) -> dict:
        """Stop a running job (cancel the KFP run).

        For training jobs the TrainJob CR is deleted as well: terminating the
        KFP run only kills the watcher step, while the JobSet trainer pods
        keep running until their owning CR is removed.
        """
        job = self.get_job(job_type, tenant_id, id)
        if not job:
            raise ValueError("job not found")
        if self.kfp and job.kfp_run_id:
            try:
                self.kfp.cancel_run(job.kfp_run_id)
            except Exception:
                pass
        if job_type == "training":
            try:
                self._delete_cluster_training(job)
            except Exception as e:  # noqa: BLE001 - status flip still applies
                logger.warning("TrainJob cleanup during stop for %s failed: %s", job.id, e)
        self.set_progress(id, "cancelled", job.progress)
        self.db.refresh(job)
        return _job_dict(job)

    def restart_job(self, job_type: str, tenant_id: str, id: str) -> dict:
        """Restart a job (cancelled/failed -> start again)."""
        return self.start_job(job_type, tenant_id, id)

    # -- gate override ---------------------------------------------------------

    def override_gate(self, tenant_id: str, job_id: str, user: str,
                      is_admin: bool, reason: str) -> dict:
        """Force-register an artifact whose hard gate failed (creator/admin only).

        Writes an audit log entry and marks the job results with the override
        provenance; the registered version carries the FAIL gate result.
        """
        from datetime import datetime, timezone

        from app.models.orm import AuditLog

        job = self.get_job("optimization", tenant_id, job_id)
        if not job:
            raise ValueError("job not found")
        if job.created_by != user and not is_admin:
            raise PermissionError("only the job creator or an admin can override a gate")
        if job.status != "failed":
            raise ValueError("only failed (gate-rejected) jobs can be overridden")
        results = job.results if isinstance(job.results, dict) else {}
        report = results.get("report") or {}
        if report.get("gate_result") != "FAIL":
            # not a gate failure (infra error) — no artifact to force-register
            raise ValueError("job did not fail a hard gate (no gate-rejected artifact)")
        metrics = results.get("metrics") or job.metrics or {}
        if not metrics.get("optimized_uri"):
            raise ValueError("artifact URI missing; cannot register")
        if not reason or not reason.strip():
            raise ValueError("an override reason is required")
        reg = register_optimized_artifact(
            self.models, job, metrics, report=report, actor=user or "unknown")
        results["gate_override"] = {
            "by": user, "reason": reason.strip(),
            "at": datetime.now(timezone.utc).isoformat(),
            "registered_version": reg.get("version", ""),
        }
        job.results = results
        job.error_message = f"gate overridden by {user}: {reason.strip()}"
        self.db.add(AuditLog(
            id=new_id("audit"), tenant_id=tenant_id, user_id=user or "",
            action="optimization.gate.override", resource_type="job",
            resource_id=job.id,
            details={"reason": reason.strip(), "version": reg.get("version", ""),
                     "gates": report.get("gates", {})},
        ))
        self.db.commit()
        self.db.refresh(job)
        return _job_dict(job)

    @staticmethod
    def _image_repository(image: str) -> str:
        """Return an image reference without its tag, preserving registry ports."""
        image = str(image or "").strip()
        last_slash = image.rfind("/")
        last_colon = image.rfind(":")
        if last_colon > last_slash:
            return image[:last_colon]
        return image

    @classmethod
    def _resolve_builtin_trainer_image(cls, image: str, expected_image: str) -> str:
        """Keep explicit user images, but correct stale built-in image choices.

        A cloned job can retain a generic CPU CV image while its accelerator is
        changed to NVIDIA/AMD/Intel. Such a built-in image is known and cannot
        provide the requested backend, so use the vendor profile image instead.
        Truly custom repositories are returned untouched.
        """
        selected = str(image or "").strip()
        if not selected:
            return expected_image
        selected_repo = cls._image_repository(selected)
        expected_repo = cls._image_repository(expected_image)
        known_repos = {
            cls._image_repository(p["cv_image"])
            for p in ACCELERATOR_PROFILES.values()
            if p.get("cv_image")
        }
        known_repos.add(cls._image_repository(nf_image("trainer/torch-cpu")))
        if selected_repo in known_repos and selected_repo != expected_repo:
            logger.warning(
                "replacing incompatible built-in trainer image %s with %s",
                selected, expected_image,
            )
            return expected_image
        return selected

    def _default_trainer_image(self) -> str:
        """Resolve the default trainer image from config (job custom or system)."""
        try:
            acc, gpu, cpu_req, cpu_lim, mem_req, mem_lim, img = self._trainer_cfg()
            return img or nf_image("trainer/torch-cpu")
        except Exception:
            return nf_image("trainer/torch-cpu")

    def set_progress(
        self, id: str, status: str, progress: int, error_msg: str = "",
        results: Optional[dict] = None, metrics: Optional[dict] = None,
    ) -> None:
        """Update job status/progress/results/metrics (used by the worker)."""
        job = self.db.get(Job, id)
        if not job:
            return
        job.status = status
        job.progress = progress
        if status in _TERMINAL_STATUSES and job.completed_at is None:
            job.completed_at = self._now()
        if error_msg:
            job.error_message = error_msg
        if results is not None:
            job.results = _json_safe(results)
        if metrics is not None:
            job.metrics = _json_safe(metrics)
        self.db.commit()

    # -- pipeline param builders --------------------------------------------

    @staticmethod
    def _is_classification_job(params: dict, config: dict) -> bool:
        """Detect an image-classification training job (timm family).

        Markers: task_type/model_type/engine in (classification, cls, timm),
        or a known timm base_model name (e.g. resnet50, vit_base_patch16_224).
        """
        timm_names = ("resnet", "vit_", "convnext", "efficientnet", "swin", "deit", "dinov", "mobilenet", "densenet")
        for src in (config, params):
            for key in ("task_type", "model_type", "engine"):
                v = str(src.get(key) or "").lower()
                if v in ("classification", "cls", "timm"):
                    return True
            bm = str(src.get("base_model") or src.get("model") or "").lower()
            if bm.endswith(".pt") is False and bm.startswith(timm_names):
                return True
        return False

    # Offline fallback only (ADR-015): the authoritative default comes from the
    # selected image's own announcement (`trainer-profile.json` ->
    # container_images.profile). This table survives for jobs whose image is
    # unknown/unannounced (external trainers, freshly registered images) and the
    # UI overrides the value per job anyway.
    DEFAULT_ARCHS = {
        "classification": "resnet50",
        "multilabel": "resnet50",
        "detection": "fasterrcnn_mobilenet_v3_large_fpn",
        "instance_seg": "maskrcnn_resnet50_fpn_v2",
        "semantic_seg": "deeplabv3_mobilenet_v3_large",
        "pose": "keypointrcnn_resnet50_fpn",
    }

    def check_capability(self, tenant_id: str, params: dict, config: dict) -> None:
        """Raise `CapabilityLocked` when the job needs an ungranted entitlement.

        Mirrors the platform-wide rule (ADR-015 D-04): the entitlement is the
        only gate, it applies to NF's own extended capabilities, and it is
        checked before a job is created so the UI can explain the upgrade path.
        """
        from app.services import entitlements

        self.tenant_id = tenant_id
        image = str(params.get("custom_image") or params.get("trainer_image") or "")
        if not image:
            try:
                _acc, _gpu, _cr, _cl, _mr, _ml, sys_img, *_rest = (*self._trainer_cfg(),)
            except Exception:  # noqa: BLE001
                sys_img = ""
            if not sys_img:
                vendor = resolve_vendor(config or {}, params, system_accelerator="", gpu_count=0)
                sys_img = get_profile(vendor).get("cv_image") or self._default_trainer_image()
            image = str(sys_img or "")
        profile = self._image_profile(image)
        if not profile:
            return
        task = self._job_task(params or {}, config or {})
        # The split training form carries architecture independently from the
        # checkpoint URI. Prefer it so capability checks inspect the network
        # being built rather than treating an S3 URI as an architecture id.
        arch = str(
            (config or {}).get("architecture")
            or (config or {}).get("base_model")
            or (config or {}).get("model")
            or ""
        )
        entitlements.check(
            profile, entitlements.granted(self.db),
            task=task, arch=arch, image=image,
        )

    def _image_profile(self, image: str) -> dict:
        """Announcement of the image a job will run on ({} when unknown).

        Best effort: capability gating and default architecture must never turn
        a scheduleable job into an error because a catalog row is missing.
        """
        try:
            from sqlalchemy import select

            from app.models.orm import ContainerImage

            tenant = str(getattr(self, "tenant_id", "") or "default")
            repo, _, tag = str(image or "").rpartition(":")
            repo = repo or str(image or "")
            row = self.db.scalar(
                select(ContainerImage).where(
                    ContainerImage.repository == repo,
                    ContainerImage.tenant_id == tenant,
                )
            )
            if not row and tag:
                row = self.db.scalar(
                    select(ContainerImage).where(
                        ContainerImage.repository == repo,
                        ContainerImage.tag == tag,
                        ContainerImage.tenant_id == tenant,
                    )
                )
            if not row:
                # Any tenant: the catalog is per-tenant but announcements match.
                row = self.db.scalar(
                    select(ContainerImage).where(ContainerImage.repository == repo)
                )
            return dict(row.profile or {}) if row else {}
        except Exception:  # noqa: BLE001 - never block scheduling on a lookup
            return {}

    def _default_arch(self, task: str, image: str) -> str:
        """Default architecture: the image's announcement, else the fallback table."""
        from app.services import trainer_profiles

        announced = trainer_profiles.default_arch(self._image_profile(image), task)
        if announced:
            return announced
        return self.DEFAULT_ARCHS.get(task, "fasterrcnn_mobilenet_v3_large_fpn")

    @classmethod
    def _job_task(cls, params: dict, config: dict) -> str:
        """Normalise the UI/API task markers to the trainer task vocabulary."""
        raw = ""
        for src in (config, params):
            for key in ("task_type", "model_type", "task", "engine"):
                value = str(src.get(key) or "").strip().lower()
                if value:
                    raw = value
                    break
            if raw:
                break
        mapping = {
            "cls": "classification",
            "classify": "classification",
            "classification": "classification",
            "timm": "classification",
            "class": "classification",
            "multi_label": "multilabel",
            "multilabel": "multilabel",
            "det": "detection",
            "detection": "detection",
            "object_detection": "detection",
            "obb": "detection",
            "rotation": "detection",
            "seg": "instance_seg",
            "segmentation": "instance_seg",
            "instance_seg": "instance_seg",
            "instance_segmentation": "instance_seg",
            "mask": "instance_seg",
            "semantic_seg": "semantic_seg",
            "semantic_segmentation": "semantic_seg",
            "keypoint": "pose",
            "keypoints": "pose",
            "pose": "pose",
        }
        return mapping.get(raw, "detection")

    def _trainer_env(self, nproc: int = 1, extra: Optional[list[dict]] = None) -> list[dict]:
        """Build the env injected into every training (TrainJob trainer) pod.

        Trainer containers download the base model / dataset and upload
        artifacts over S3, so they need the real cluster endpoint and
        credentials (the in-image defaults are stale on purpose replaced here).
        NPROC is required for multi-process / multi-GPU entrypoints (torchrun).
        Values are sourced from the runtime config single-source-of-truth.
        Vendor-specific env (e.g. HSA_OVERRIDE_GFX_VERSION) is appended via
        `extra`, de-duplicated by name with the built-in keys winning.
        """
        from app.core.config import config as _cfg

        access_key, secret_key = require_s3_credentials(
            _cfg.storage.s3_access_key,
            _cfg.storage.s3_secret_key,
            scope="storage.s3",
        )
        endpoint = (
            _cfg.storage.s3_cluster_endpoint
            or "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"
        )
        base = [
            {"name": "S3_ENDPOINT", "value": endpoint},
            {"name": "S3_ACCESS_KEY", "value": access_key},
            {"name": "S3_SECRET_KEY", "value": secret_key},
            {"name": "S3_BUCKET", "value": _cfg.storage.bucket or "nereidforge"},
            {"name": "NPROC", "value": str(max(1, int(nproc or 1)))},
        ]
        seen = {e["name"] for e in base}
        for e in extra or []:
            if isinstance(e, dict) and e.get("name") and e["name"] not in seen:
                base.append({"name": str(e["name"]), "value": str(e.get("value", ""))})
                seen.add(e["name"])
        return base

    @staticmethod
    def _profile_resources(profile: dict, count: str, cpu_req: str, cpu_lim: str,
                           mem_req: str, mem_lim: str, multiplier: int = 1):
        """Build requests/limits maps with the vendor's extended resource key.

        GPU vendors register different resource names (nvidia.com/gpu,
        amd.com/gpu, gpu.intel.com/xe); the profile owns that key so this code
        stays vendor-neutral. Returns (requests, limits, effective_count).
        """
        res_req = {"cpu": cpu_req, "memory": mem_req}
        res_lim = {"cpu": cpu_lim, "memory": mem_lim}
        rkey = profile.get("resource")
        if rkey:
            try:
                n = int(str(count or "0"))
            except (TypeError, ValueError):
                n = 0
            if n <= 0:
                # A GPU vendor was explicitly selected; request one card.
                n = 1
            n = max(1, n * max(1, multiplier))
            res_req[rkey] = str(n)
            res_lim[rkey] = str(n)
        return res_req, res_lim

    @staticmethod
    def _requested_gpu_count(cfg: dict, params: dict, profile: dict, default_count: Any = 0) -> int:
        """Resolve the per-node GPU/shared-device count requested by a job.

        No hardcoded upper bound: the value is validated against cluster
        capacity in the UI and ultimately by the Kubernetes scheduler. A
        minimum of 1 applies whenever the vendor profile uses an extended
        resource, including shared vGPU devices advertised per physical card.
        """
        if not profile.get("resource"):
            return 0
        raw = cfg.get("gpu_count", params.get("gpu_count", cfg.get("num_proc_per_node", default_count)))
        try:
            count = int(str(raw or "0"))
        except (TypeError, ValueError):
            count = int(str(default_count or "0"))
        return max(1, count)

    @staticmethod
    def _trainer_device(vendor: str, profile: dict, gpu_count: int) -> str:
        """Build the device string expected by the selected trainer."""
        if vendor == "cpu" or gpu_count <= 1:
            return str(profile.get("device") or "cpu")
        if vendor == "intel":
            return "xpu:0"
        return ",".join(str(i) for i in range(gpu_count))

    def training_pipeline_params(self, tenant_id: str, params: dict, job: Job) -> dict[str, str]:
        """Build training pipeline params and write train_config.json to S3.

        train_config is written under train-input/<job_id>/ (per-job key) so
        concurrent CV jobs never read each other's config — mirrors the LLM
        path in llm_pipeline_params().
        """
        if self._is_llm_job((job.params or {}), (job.config or {})):
            return self.llm_pipeline_params(tenant_id, params, job)
        out = {k: str(v) for k, v in params.items()}
        ds_id = out.get("dataset_id", "") or out.get("model_id", "")
        out["dataset_name"] = ds_id
        out.setdefault("dataset_version", "1")
        out["model_name"] = out.get("model_id") or "model"
        out["model_version"] = "1.0.0"
        out["artifact_id"] = job.id
        # Same explicit-URI convention as the optimization/evaluation pipelines:
        # an advanced caller (custom script, I-11) may pin where artifacts land;
        # otherwise derive from job_id like today.
        out["train_output_dir"] = str(params.get("output_uri") or "").strip() or self.s3.tenant_uri(
            f"s3://nereidforge/models/{job.id}/", tenant_id
        )
        out["dataset_output_dir"] = str(params.get("dataset_uri") or "").strip() or self.resolve_training_data_uri(tenant_id, out, job)

        acc, gpu, cpu_req, cpu_lim, mem_req, mem_lim, sys_img = self._trainer_cfg()
        # Resolve the accelerator vendor (per-job selector -> system default ->
        # nvidia for legacy GPU counts -> cpu) and its vendor-specific image /
        # extended resource / device string (ADR-012, services/accelerators.py).
        vendor = resolve_vendor(job.config or {}, params, system_accelerator=acc, gpu_count=gpu)
        profile = get_profile(vendor)
        trainer_img = profile.get("cv_image") or sys_img or \
            nf_image("trainer/torch-cpu")
        # Custom training support: a job may point to a user-owned image/script.
        # When provided, the KFP train step pulls that image and runs the given
        # entrypoint/script instead of the built-in trainer template.
        custom_image = canonical_trainer_ref(str(params.get("custom_image", "") or ""))
        custom_script = str(params.get("custom_script_uri", "") or "")
        # Every vision task (classification, detection, segmentation, pose) runs
        # on the vendor's torch image: one image serves the whole task matrix
        # (ADR-014 D1). Unknown custom images always win; known built-in images
        # are corrected when a cloned job pairs them with an incompatible
        # accelerator.
        is_cls_job = self._is_classification_job((job.params or {}), (job.config or {}))
        job_task = self._job_task((job.params or {}), (job.config or {}))
        expected_image = trainer_img
        trainer_img = self._resolve_builtin_trainer_image(custom_image, expected_image)
        out["accelerator"] = vendor
        out["trainer_image"] = trainer_img
        gpu_eff = self._requested_gpu_count(job.config or {}, params, profile, gpu)
        out["gpu_request"] = str(gpu_eff)
        out["gpu_limit"] = str(gpu_eff)
        out["cpu_request"] = cpu_req
        out["cpu_limit"] = cpu_lim
        out["memory_request"] = mem_req
        out["memory_limit"] = mem_lim

        epochs = str(job.config.get("epochs", "2"))
        imgsz = str(job.config.get("imgsz", "640"))
        batch = str(job.config.get("batch", "16"))
        default_base = self._default_arch(job_task, trainer_img)
        base_model = str(job.config.get("base_model") or job.config.get("model") or default_base)

        res_req, res_lim = self._profile_resources(
            profile, gpu_eff, cpu_req, cpu_lim, mem_req, mem_lim
        )

        train_env = self._trainer_env(gpu_eff, extra=profile.get("env"))
        device = self._trainer_device(vendor, profile, gpu_eff)
        # torchrun (mlPolicy.torch in the ClusterTrainingRuntime) spawns one
        # process per proc count; a CPU/non-GPU job still runs a single process.
        nproc = gpu_eff if profile.get("resource") and gpu_eff > 0 else 1
        train_config = {
            "image": trainer_img,
            "numNodes": 1,
            "numProcPerNode": nproc,
            "resourcesPerNode": {"requests": res_req, "limits": res_lim},
            "env": train_env,
            "accelerator": vendor,
            "device": device,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "model": base_model,
            "split_ratio": job.config.get("split_ratio", 0.8),
        }
        # DataLoader defaults belong to each trainer (Torch=2, custom YOLO=0).
        # Emit the field only for an explicit request, preserving zero as a real
        # user choice instead of collapsing it into the trainer's default.
        workers = job.config.get("workers")
        if workers is not None:
            train_config["workers"] = max(0, int(workers))
        if profile.get("tolerations"):
            train_config["tolerations"] = profile["tolerations"]
        # Augmentation/regularization block for image detection jobs. Passed
        # through verbatim; the trainer whitelists + clamps the keys.
        # Classification and LLM/VLM jobs ignore it.
        aug = job.config.get("augmentation")
        if isinstance(aug, dict) and not is_cls_job:
            train_config["augmentation"] = aug
        # Node placement: merge the vendor profile selector with an optional
        # user-requested hostname pin (empty = scheduler decides).
        node_selector = dict(profile.get("node_selector") or {})
        node = str((job.config or {}).get("node") or params.get("node") or "").strip()
        if node:
            node_selector["kubernetes.io/hostname"] = node
        if node_selector:
            train_config["node_selector"] = node_selector
        # The trainer reads the task from the config (ADR-014): one entrypoint
        # serves classification, detection, instance/semantic segmentation and
        # pose. `model` is the vision architecture id (resnet50,
        # fasterrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn_v2, ...).
        train_config["task"] = job_task
        # Current clients separate architecture, initialization and checkpoint
        # URI. Legacy clients may still overload `base_model` with a bare
        # architecture id, a weights URI, or a legacy stock file name; keep
        # that wire compatibility at this edge instead of leaking it into the
        # trainer contract (ADR-014 D-07).
        explicit_arch = str(job.config.get("architecture") or "").strip()
        explicit_weights = str(job.config.get("base_weights_uri") or "").strip()
        chosen = explicit_weights or str(job.config.get("base_model") or "").strip()
        weights_exts = (".pt", ".pth", ".ckpt", ".safetensors")
        weights_ref = bool(explicit_weights) or (
            bool(chosen) and ("/" in chosen or chosen.startswith("s3:"))
        )
        legacy_stock = (
            not explicit_weights and bool(chosen)
            and not weights_ref and chosen.endswith(weights_exts)
        )
        arch = explicit_arch or (
            "" if (weights_ref or legacy_stock) else (chosen or base_model)
        )
        if weights_ref:
            train_config["base_weights_uri"] = chosen
        elif legacy_stock:
            # Legacy bare checkpoint name: kept for trainers that resolve it
            # themselves (ADR-014 D5: NF ships no weights in the image).
            train_config["legacy_base_model"] = chosen
        # Stock ultralytics checkpoints encode their architecture in the file
        # name (yolov8n.pt, yolo11x-seg.pt, ...). Customer-owned ultralytics
        # trainer images (ADR-016) read CONFIG_JSON["model"] and ignore
        # base_weights_uri, so without this derivation a pool selection
        # silently kept the torchvision task default (pit 184). Explicit arch
        # wins; otherwise registered-version metadata below takes precedence
        # over stock inference, and arbitrary names stay on the task default.
        stock_arch = (
            infer_stock_arch(chosen) if (weights_ref or legacy_stock) else ""
        )
        if stock_arch:
            logger.info(
                "training %s: base_model %s carries stock arch %s",
                job.id, chosen, stock_arch,
            )
        if "pretrained" in job.config:
            # JSON boolean supplied by the explicit initialization selector.
            # Legacy jobs omit it and retain the trainer's pretrained default.
            train_config["pretrained"] = bool(job.config.get("pretrained"))
        base_id = str(job.config.get("base_model_id") or "")
        if base_id:
            source_arch = self._source_version_arch(job.tenant_id, base_id, base_model)
            if source_arch:
                # Preserve the source architecture as lineage while allowing
                # an explicit UI selection to override it for transfer learning.
                if not explicit_arch:
                    arch = source_arch
                train_config["base_arch"] = source_arch
        train_config["model"] = arch or stock_arch or default_base
        if custom_script:
            train_config["custom_script_uri"] = custom_script
        train_input_path = self.s3.tenant_filer_path(
            f"/buckets/nereidforge/train-input/{job.id}/train_config.json", tenant_id
        )
        try:
            self.s3.put_object(train_input_path, json.dumps(train_config).encode())
        except Exception:
            pass
        out["train_config"] = self.s3.tenant_uri(
            f"s3://nereidforge/train-input/{job.id}/train_config.json", tenant_id
        )
        out["namespace"] = "nereidforge"
        out["seaweedfs_url"] = "http://seaweedfs-filer.nereidforge.svc.cluster.local:8888"
        out["runtime_name"] = "torch-distributed"
        out["dataset_registry_url"] = ""
        out["hub_api_url"] = ""
        out["event_bus_url"] = ""
        return out

    # -- source version helpers ----------------------------------------------

    def _source_version_arch(self, tenant_id: str, model_id: str, checkpoint_uri: str) -> str:
        """Resolve the timm arch recorded on a registered source version.

        Matches the version by its metadata ``weights_uri`` / ``file_path``
        against the checkpoint the job starts from; falls back to the newest
        version of the model. Returns an empty string when no arch metadata is
        available.
        """
        rows = self.db.scalars(
            select(ModelVersion).where(
                and_(
                    ModelVersion.model_id == model_id,
                    ModelVersion.tenant_id == tenant_id,
                )
            )
        ).all()
        rows = sorted(rows, key=lambda r: str(r.version or ""))
        matched = None
        for row in rows:
            meta = row.metadata_ or {}
            if str(meta.get("weights_uri") or "") == checkpoint_uri \
                    or str(row.file_path or "") == checkpoint_uri:
                matched = row
                break
        if matched is None and rows:
            matched = rows[-1]
        if matched is None:
            return ""
        return str((matched.metadata_ or {}).get("arch") or "")

    # -- LLM / VLM (LLaMA-Factory) training -----------------------------------
    @staticmethod
    def _is_llm_job(params: dict, config: dict) -> bool:
        """Detect an LLM/VLM training job from the task type markers sent by the UI."""
        for src in (config, params):
            if str(src.get("task_type") or "") in ("llm", "vlm"):
                return True
            if str(src.get("model_type") or "") in ("llm", "vlm"):
                return True
            if str(src.get("engine") or "") == "llm":
                return True
        return False

    def llm_pipeline_params(self, tenant_id: str, params: dict, job: Job) -> dict[str, str]:
        """Build KFP pipeline params and train_config for LLM/VLM jobs.

        CUDA and ROCm images use the LLaMA-Factory entrypoint, while Intel XPU VLM
        jobs use the native transformers/peft Idefics3 entrypoint.
        """
        out = {k: str(v) for k, v in params.items()}
        cfg = dict(job.config or {})
        ds_id = out.get("dataset_id", "") or out.get("model_id", "")
        out["dataset_name"] = ds_id
        out.setdefault("dataset_version", "1")
        out["model_name"] = out.get("model_id") or "llm"
        out["model_version"] = "1.0.0"
        out["artifact_id"] = job.id
        # Explicit output_uri/dataset_uri override (same convention as the
        # optimization/evaluation pipelines); fall back to job-derived paths.
        out["train_output_dir"] = str(params.get("output_uri") or "").strip() or self.s3.tenant_uri(
            f"s3://nereidforge/models/{job.id}/", tenant_id
        )
        out["dataset_output_dir"] = str(params.get("dataset_uri") or "").strip() or self.resolve_llm_data_uri(tenant_id, out, job)
        out["namespace"] = "nereidforge"
        out["seaweedfs_url"] = "http://seaweedfs-filer.nereidforge.svc.cluster.local:8888"
        out["trainjob_name"] = f"train-{job.id}"

        acc, gpu, cpu_req, cpu_lim, mem_req, mem_lim, _sys_img = self._trainer_cfg()
        # LLM/VLM always needs a GPU vendor; resolve the per-vendor image and
        # resource key (cu/rocm run LLaMA-Factory, xpu runs the native torch.xpu
        # transformers/peft entrypoint — separate images, ADR-012).
        vendor = resolve_vendor(cfg, params, system_accelerator=acc, gpu_count=gpu)
        profile = get_profile(vendor)
        trainer_img = profile.get("llm_image") or nf_image("trainer/llm-cu")
        num_nodes = int(cfg.get("num_nodes") or 1)
        gpu_count = self._requested_gpu_count(cfg, params, profile, gpu or 1)
        nproc = max(1, gpu_count)
        if not profile.get("distributed", True):
            num_nodes, nproc = 1, 1
        task_type = str(
            cfg.get("task_type") or params.get("task_type")
            or cfg.get("model_type") or params.get("model_type") or "llm"
        ).lower()
        entrypoint = "/app/vlm_train.py" if vendor == "intel" and task_type == "vlm" else "/app/llm_train.py"

        # resourcesPerNode = cards per node (nproc); vendor key comes from profile.
        res_req, res_lim = self._profile_resources(
            profile, 1, cpu_req, cpu_lim, mem_req, mem_lim, multiplier=nproc
        )

        train_config = {
            "image": trainer_img,
            "command": ["python"],
            "args": [entrypoint],
            "numNodes": num_nodes,
            "numProcPerNode": nproc,
            "resourcesPerNode": {"requests": res_req, "limits": res_lim},
            "dataset_uri": out["dataset_output_dir"],
            "output_dir": out["train_output_dir"],
            "llm": self._llm_args(cfg, params),
            "env": self._trainer_env(nproc, extra=profile.get("env")),
            "accelerator": vendor,
            "task_type": task_type,
        }
        if profile.get("tolerations"):
            train_config["tolerations"] = profile["tolerations"]
        node_selector = dict(profile.get("node_selector") or {})
        node = str(cfg.get("node") or params.get("node") or "").strip()
        if node:
            node_selector["kubernetes.io/hostname"] = node
        if node_selector:
            train_config["node_selector"] = node_selector
        out["accelerator"] = vendor
        # Unique per-run config path so concurrent LLM runs never clobber the fixed CV slot.
        train_input_path = self.s3.tenant_filer_path(
            f"/buckets/nereidforge/train-input/{job.id}/train_config.json", tenant_id
        )
        try:
            self.s3.put_object(train_input_path, json.dumps(train_config).encode())
        except Exception:
            pass
        out["train_config"] = self.s3.tenant_uri(
            f"s3://nereidforge/train-input/{job.id}/train_config.json", tenant_id
        )
        out["trainer_image"] = trainer_img
        return out

    def resolve_llm_data_uri(self, tenant_id: str, params: dict, job: Job) -> str:
        """Resolve the S3 prefix of an `llm-sft` dataset version (contains .jsonl).

        Falls back to s3://nereidforge/llm-train/ when no annotated LLM dataset is found.
        """
        fallback = self.s3.tenant_uri("s3://nereidforge/llm-train/", tenant_id)
        ds_id = params.get("dataset_id", "")
        if not ds_id or not self.dataset:
            return fallback
        try:
            ver = int(params.get("dataset_version", "1") or "1")
        except ValueError:
            ver = 1
        ds = self.dataset.get_dataset(tenant_id, ds_id)
        if not ds:
            return fallback
        if ver <= 0:
            ver = ds["version"]
        ver_obj = self.dataset.get_dataset_version(tenant_id, ds_id, ver)
        if not ver_obj or not ver_obj.uri:
            return fallback
        base_uri = ver_obj.uri.rstrip("/")
        filer_base = self.s3.s3_uri_to_filer_path(base_uri)
        if not filer_base:
            return fallback
        try:
            keys = self.s3.walk(filer_base)
        except Exception:
            return fallback
        if any(str(k).endswith(".jsonl") for k in keys):
            return base_uri + "/"
        return fallback

    @staticmethod
    def _llm_args(cfg: dict, params: dict) -> dict:
        """Map the NF job config to a LLaMA-Factory-compatible argument block.

        A default set is provided for every key so a bare request still trains;
        the UI can override any value through the nested `llm` config map.
        """
        llm = dict(cfg.get("llm") or {})
        args = {
            "finetuning_type": llm.get("finetuning_type", "lora"),
            "stage": llm.get("stage", "sft"),
            "do_train": True,
            "base_model_uri": str(cfg.get("base_model") or llm.get("base_model_uri") or ""),
            "template": str(llm.get("template") or "qwen"),
            "dataset": ["train"],
            "cutoff_len": int(llm.get("cutoff_len") or 1024),
            "num_train_epochs": float(llm.get("num_train_epochs") or cfg.get("epochs") or 3),
            "learning_rate": float(llm.get("learning_rate") or 2e-4),
            "per_device_train_batch_size": int(llm.get("per_device_train_batch_size") or cfg.get("batch") or 4),
            "gradient_accumulation_steps": int(llm.get("gradient_accumulation_steps") or 4),
            "lora_rank": int(llm.get("lora_rank") or 16),
            "lora_alpha": int(llm.get("lora_alpha") or 32),
            "lora_dropout": float(llm.get("lora_dropout") or 0.05),
            "lora_target": str(llm.get("lora_target") or "all"),
            "logging_steps": int(llm.get("logging_steps") or 1),
            "save_steps": int(llm.get("save_steps") or 500),
            "save_only_model": bool(llm.get("save_only_model", True)),
            "plot_loss": bool(llm.get("plot_loss", False)),
            "report_to": "none",
            "trust_remote_code": True,
        }
        # quantization needs CUDA (bitsandbytes); only set when requested AND
        # explicitly non-zero, otherwise CPU runs crash at startup.
        qb = llm.get("quantization_bit") or cfg.get("quantization_bit") or 0
        if int(qb) > 0:
            args["quantization_bit"] = int(qb)
        for k, v in llm.items():
            if k not in args and k not in ("quantization_bit",):
                args[k] = v
        return args

    # -- entity resolution (optimization / evaluation) ------------------------

    def validate_optimization_source(self, tenant_id: str, params: dict) -> None:
        """Reject versions whose recorded capabilities exclude optimization.

        New uploads advertise only implemented workflows; legacy metadata keeps
        the historical ONNX/IR checks. The optimizer quantizes the ONNX graph
        (ONNX Runtime) and converts to
        the vendor-native format only afterwards (ovc for OpenVINO IR, trtexec
        later for TensorRT). An OpenVINO IR (.xml/.bin) version cannot be
        re-ingested: ``onnx.load`` on an .xml graph raises
        ``google.protobuf.message.DecodeError`` deep in the pipeline.
        INT8/FP16 IR is produced from the ONNX source in a single job, not by
        re-optimizing an ir-fp32 version.
        """
        model_id = str(params.get("source_model_id") or params.get("model_id") or "")
        if not model_id:
            return
        version = str(params.get("source_version") or "")
        from app.models.orm import ModelVersion

        q = select(ModelVersion).where(and_(
            ModelVersion.tenant_id == tenant_id, ModelVersion.model_id == model_id))
        if version:
            q = q.where(ModelVersion.version == version)
        else:
            q = q.order_by(ModelVersion.created_at.desc())
        row = self.db.scalars(q).first()
        if not row:
            # Unknown source; the resolution error surfaces with context at start.
            return
        uri = (row.file_path or "").lower()
        meta = row.metadata_ or {}
        if not artifact_capability_permitted(meta, "optimize"):
            artifact_kind = str(meta.get("artifact_format") or "this artifact")
            raise OptimizationSourceError(
                f"source version '{row.version}' ({artifact_kind}) does not support "
                "optimization in this release; select an ONNX source version"
            )
        if uri.endswith(".xml") or str(meta.get("framework") or "") == "openvino":
            raise OptimizationSourceError(
                f"source version '{row.version}' is an OpenVINO IR (.xml/.bin); "
                "the optimizer consumes ONNX. Select the original ONNX version "
                "as the source — INT8/FP16 IR is produced directly from ONNX in "
                "one optimization job."
            )

    def _resolve_model_artifact_uri(self, tenant_id: str, model_id: str, version: str) -> str:
        """Resolve a model version to its deployable artifact URI.

        Prefers the version file_path; when it is a .pt export, falls back to
        the onnx_uri recorded in version metadata (training exports both).
        """
        from app.models.orm import Model, ModelVersion

        if not model_id:
            raise ValueError("model_id is required")
        q = select(ModelVersion).where(
            and_(ModelVersion.tenant_id == tenant_id, ModelVersion.model_id == model_id)
        )
        if version:
            q = q.where(ModelVersion.version == str(version))
        else:
            q = q.order_by(ModelVersion.created_at.desc())
        row = self.db.scalars(q).first()
        if not row:
            m = self.db.scalar(select(Model).where(
                and_(Model.tenant_id == tenant_id, Model.id == model_id)))
            hint = f" '{model_id}'" if not m else ""
            raise ValueError(f"model version not found{hint}")
        uri = (row.file_path or "").strip()
        meta = row.metadata_ or {}
        if uri.lower().endswith(".pt"):
            onnx_uri = str(meta.get("onnx_uri") or "")
            if onnx_uri:
                return onnx_uri
            raise ValueError(
                f"model version {row.version} is a .pt export without an onnx_uri; "
                "deploy or optimize an exported version instead"
            )
        if not uri:
            raise ValueError(f"model version {row.version} has no artifact URI")
        return uri

    def _resolve_eval_dataset_uri(self, tenant_id: str, params: dict) -> str:
        """Resolve the evaluation dataset prefix (COCO export preferred).

        Uses the dataset version URI; when the version only holds raw images,
        falls back to its most recent annotated/*/ export; the legacy
        yolo-train prefix is the last resort.
        """
        fallback = self.s3.tenant_uri("s3://nereidforge/yolo-train/", tenant_id)
        ds_id = params.get("dataset_id", "")
        if not ds_id:
            return fallback
        try:
            ver = int(params.get("dataset_version", "1") or 1)
        except ValueError:
            ver = 1
        from app.models.orm import DatasetVersion

        row = self.db.scalar(select(DatasetVersion).where(and_(
            DatasetVersion.tenant_id == tenant_id,
            DatasetVersion.dataset_id == ds_id,
            DatasetVersion.version == ver,
        )))
        if not row or not row.uri:
            return fallback
        base = row.uri.rstrip("/")
        filer_base = self.s3.s3_uri_to_filer_path(base)
        if filer_base and self._object_exists(filer_base + "/data.yaml") and (
            self._object_exists(filer_base + "/train/images/")
            or self._object_exists(filer_base + "/train/")
            or self._object_exists(filer_base + "/labels/")
        ):
            # Canonical YOLO split or flat layout: evaluate the version root.
            return base + "/"
        if filer_base and self._object_exists(filer_base + "/class_names.json") and (
            self._object_exists(filer_base + "/train/")
            or self._object_exists(filer_base + "/images/")
        ):
            # Canonical ImageFolder classification layout.
            return base + "/"
        if filer_base and self._object_exists(filer_base + "/coco/annotations.json"):
            return base + "/"
        if filer_base and self._object_exists(filer_base + "/annotations.json"):
            return base + "/"
        if filer_base and self._object_exists(filer_base + "/labels/"):
            return base + "/"
        # Raw version: pick the latest annotated export (timestamped dirs)
        try:
            if filer_base:
                entries = self.s3.list_entries(f"{filer_base}/annotated/")
                names = sorted(e.name for e in entries if e.is_dir)
                for name in reversed(names):
                    if self._object_exists(f"{filer_base}/annotated/{name}/coco/annotations.json"):
                        return base + f"/annotated/{name}/"
        except Exception:
            pass
        return base + "/"

    def _launch_katib_search(self, tenant_id: str, params: dict, job: Job) -> str:
        """Run a Katib grid-search Experiment via the KFP optimization-search
        pipeline and return the KFP run id.

        The experiment runs `search_trial.py` in the optimizer image for every
        parameter combination (grid), minimizing latency with an mAP-drop hard
        constraint. The pipeline pod creates/watches the Experiment (no NF
        cluster access) and writes best_trial.json; the worker polls the KFP
        run and registers the winning trial's artifact.
        """
        import json

        access_key, secret_key = require_s3_credentials(
            config.storage.s3_access_key,
            config.storage.s3_secret_key,
            scope="storage.s3",
        )
        model_id = str(params.get("source_model_id") or params.get("model_id") or "")
        version = str(params.get("source_version") or "")
        model_uri = self._resolve_model_artifact_uri(tenant_id, model_id, version)
        dataset_uri = self._resolve_eval_dataset_uri(tenant_id, params)
        exp_name = f"opt-{job.id}"
        ns = "nereidforge"
        max_drop = float(params.get("max_map_drop_pct", 5) or 5)
        calib_n = int(params.get("calib_n", 200) or 200)

        container = {
            "name": "trial",
            "image": nf_image("train/optimizer"),
            "command": ["python", "/app/search_trial.py"],
            "env": [
                {"name": "S3_ENDPOINT", "value": "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"},
                {"name": "S3_ACCESS_KEY", "value": access_key},
                {"name": "S3_SECRET_KEY", "value": secret_key},
                {"name": "INPUT_URI", "value": model_uri},
                {"name": "DATASET_URI", "value": dataset_uri},
                {"name": "OUTPUT_URI", "value": self.s3.tenant_uri(f"s3://nereidforge/optimized/{job.id}/", tenant_id)},
                {"name": "ARTIFACT_ID", "value": "${trialParameters.trialName}"},
                {"name": "QUANT_METHOD", "value": "${trialParameters.quant_method}"},
                {"name": "IMGSZ", "value": "${trialParameters.imgsz}"},
                {"name": "CONF_THR", "value": "${trialParameters.conf_threshold}"},
                {"name": "CALIB_N", "value": str(calib_n)},
                {"name": "TARGET_DEVICE", "value": normalize_target_device(params.get("target_device"))},
                {"name": "OUTPUT_DIR", "value": "/output"},
            ],
            "resources": {"requests": {"cpu": "2", "memory": "4Gi"}, "limits": {"cpu": "4", "memory": "8Gi"}},
        }

        experiment = {
            "apiVersion": "kubeflow.org/v1beta1",
            "kind": "Experiment",
            "metadata": {"name": exp_name, "namespace": ns},
            "spec": {
                "algorithm": {"algorithmName": "grid"},
                "parameters": [
                    {"name": "quant_method", "parameterType": "categorical",
                     "feasibleSpace": {"list": ["graph", "int8-dynamic", "fp16"]}},
                    {"name": "imgsz", "parameterType": "int",
                     "feasibleSpace": {"min": "320", "max": "640", "step": "160"}},
                    {"name": "conf_threshold", "parameterType": "double",
                     "feasibleSpace": {"min": "0.2", "max": "0.4", "step": "0.1"}},
                ],
                "objective": {
                    "type": "minimize",
                    "objectiveMetricName": "latency_ms",
                    "additionalMetricNames": ["mAP50", "map_drop_pct", "size_reduction_pct"],
                    "metricStrategies": [
                        {"name": "latency_ms", "value": "min"},
                        {"name": "mAP50", "value": "max"},
                        {"name": "map_drop_pct", "value": "min"},
                        {"name": "size_reduction_pct", "value": "max"},
                    ],
                    "constraints": [{
                        "metric": "map_drop_pct",
                        "constraint": {"type": "max", "value": str(max_drop)},
                    }],
                    "goal": 0.0,
                },
                "parallelTrialCount": 2,
                "maxTrialCount": 12,
                "metricsCollector": {
                    "collector": {
                        "kind": "FileCollector",
                        "fileCollector": {"kind": "FileKind"},
                    },
                },
                "trialTemplate": {
                    "primaryContainerName": "trial",
                    "trialParameters": [
                        {"name": "quant_method", "description": "quant method", "reference": "quant_method"},
                        {"name": "imgsz", "description": "image size", "reference": "imgsz"},
                        {"name": "conf_threshold", "description": "confidence", "reference": "conf_threshold"},
                        {"name": "trialName", "description": "trial job name", "reference": "${trialSpec.Name}"},
                    ],
                    "trialSpec": {
                        "apiVersion": "batch/v1",
                        "kind": "Job",
                        "spec": {
                            "template": {"spec": {"containers": [container], "restartPolicy": "Never"}},
                            "backoffLimit": 1,
                        },
                    },
                },
            },
        }
        from app.clients import cluster_ops

        import uuid

        run_id = cluster_ops.submit_pipeline(
            "optimization_search_pipeline",
            params={
                "experiment_json": json.dumps(experiment),
                "job_id": job.id,
                "best_trial_s3_uri": f"s3://nereidforge/optimized/{job.id}/best_trial.json",
                "s3_endpoint": self.s3.s3_endpoint or "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333",
                "s3_access_key": access_key,
                "s3_secret_key": secret_key,
                "timeout_minutes": 360,
            },
            run_name=f"katib-search-{job.id}-{uuid.uuid4().hex[:6]}",
            namespace=ns,
            wait=False,
        )
        return run_id

    def optimization_pipeline_params(
        self, tenant_id: str, params: dict, job: Job, siblings: Optional[list[Job]] = None,
    ) -> dict:
        """Build single-run multi-target DAG params for optimization.

        siblings: all pending jobs sharing the batch (each becomes one DAG
        branch); a single job gets a one-branch DAG.
        """
        access_key, secret_key = require_s3_credentials(
            config.storage.s3_access_key,
            config.storage.s3_secret_key,
            scope="storage.s3",
        )
        source_model_id = str(params.get("source_model_id") or params.get("model_id") or "")
        source_version = str(params.get("source_version") or "")
        model_uri = self._resolve_model_artifact_uri(tenant_id, source_model_id, source_version)
        dataset_uri = self._resolve_eval_dataset_uri(tenant_id, params)
        eval_cfg = {
            "task_type": str(params.get("task_type", "detection")),
            "conf_threshold": float(params.get("conf_threshold", 0.25) or 0.25),
            "max_images": int(params.get("max_images", 0) or 0),
            "imgsz": int(params.get("imgsz", 640) or 640),
            # Score the exporter's val split by default (canonical layout);
            # callers may pass eval_split=test|train|all.
            "eval_split": str(params.get("eval_split", "val") or "val"),
        }
        gates: dict[str, Any] = {}
        for k in ("max_map_drop_pct", "p99_latency_ms", "min_throughput_qps"):
            if params.get(k) not in (None, ""):
                gates[k] = params[k]
        # Edge contract (solution D3 gate-2): OpenVINO edge artifacts must not
        # lose more than 0.01 absolute mAP50 vs the FP32 baseline. Explicit
        # caller value wins; absent config defaults it only for OV batches.
        all_targets = siblings or [job]
        has_openvino = any(
            normalize_target_device((j.params or {}).get("target_device")) == "openvino"
            for j in all_targets
        )
        if params.get("max_map_drop_abs") not in (None, ""):
            gates["max_map_drop_abs"] = float(params["max_map_drop_abs"])
        elif has_openvino and "max_map_drop_abs" not in gates:
            gates["max_map_drop_abs"] = 0.01
        targets = []
        for j in all_targets:
            p = j.params or {}
            method = str(p.get("quant_method") or ("graph" if not p.get("dtype") else p.get("dtype")))
            targets.append({
                "job_id": j.id,
                "target_device": normalize_target_device(p.get("target_device")),
                "quant_method": method,
                "dtype": str(p.get("dtype", "") or ""),
                "calib_n": int(p.get("calib_n", 200) or 200),
                "val_n": int(p.get("val_n", 100) or 100),
                "max_map_drop_abs": (
                    float(p["max_map_drop_abs"]) if p.get("max_map_drop_abs") not in (None, "")
                    else gates.get("max_map_drop_abs")
                ),
            })
        return {
            "model_uri": model_uri,
            "dataset_uri": dataset_uri,
            "output_uri": self.s3.tenant_uri(
                f"s3://nereidforge/optimized/{job.batch_id or job.id}/", tenant_id),
            "targets_json": json.dumps(targets),
            "gates_json": json.dumps(gates),
            "eval_config_json": json.dumps(eval_cfg),
            # S3 credentials for the in-cluster optimizer/evaluator pods
            # (mirrors the training pipeline env injection).
            "s3_access_key": access_key,
            "s3_secret_key": secret_key,
        }

    def evaluation_pipeline_params(self, tenant_id: str, params: dict, job: Job) -> dict[str, str]:
        """Build evaluation pipeline params with entity resolution."""
        access_key, secret_key = require_s3_credentials(
            config.storage.s3_access_key,
            config.storage.s3_secret_key,
            scope="storage.s3",
        )
        model_id = str(params.get("model_id") or "")
        version = str(params.get("model_version") or params.get("version") or "")
        model_uri = self._resolve_model_artifact_uri(tenant_id, model_id, version)
        baseline_uri = ""
        b_id = str(params.get("baseline_model_id") or "")
        if b_id:
            baseline_uri = self._resolve_model_artifact_uri(
                tenant_id, b_id, str(params.get("baseline_model_version") or ""))
        dataset_uri = self._resolve_eval_dataset_uri(tenant_id, params)
        cfg = {
            "task_type": str(params.get("task_type", "detection")),
            "conf_threshold": float(params.get("conf_threshold", 0.25) or 0.25),
            "max_images": int(params.get("max_images", 0) or 0),
            "imgsz": int(params.get("imgsz", 640) or 640),
            # Score the exporter's val split by default (canonical layout);
            # callers may pass eval_split=test|train|all.
            "eval_split": str(params.get("eval_split", "val") or "val"),
        }
        return {
            "model_uri": model_uri,
            "baseline_model_uri": baseline_uri,
            "dataset_uri": dataset_uri,
            "output_uri": self.s3.tenant_uri(f"s3://nereidforge/eval/{job.id}/", tenant_id),
            "report_uri": self.s3.tenant_uri(
                f"s3://nereidforge/evaluation-reports/{job.id}/report.json", tenant_id),
            "artifact_id": job.id,
            "config_json": json.dumps(cfg),
            "cpu_request": str(params.get("cpu_request", "4")),
            "cpu_limit": str(params.get("cpu_limit", "4")),
            "memory_request": str(params.get("memory_request", "8Gi")),
            "memory_limit": str(params.get("memory_limit", "8Gi")),
            "s3_access_key": access_key,
            "s3_secret_key": secret_key,
        }

    def resolve_training_data_uri(self, tenant_id: str, params: dict, job: Job) -> str:
        """Resolve the training data source (version root) from the dataset version.

        Recognizes both canonical YOLO layouts (flat ``images/+labels/+data.yaml``
        or split ``train/val|test + data.yaml``) and both ImageFolder layouts
        (flat ``images/<class>/`` or split ``train/<class>/``, marked by
        ``class_names.json``). Legacy COCO versions (annotations.json only) get
        a generated flat data.yaml; anything unrecognized falls back to yolo-train.
        """
        fallback = self.s3.tenant_uri("s3://nereidforge/yolo-train/", tenant_id)
        ds_id = params.get("dataset_id", "")
        if not ds_id or not self.dataset:
            return fallback
        ver_str = params.get("dataset_version", "1")
        try:
            ver = int(ver_str)
        except ValueError:
            ver = 1
        ds = self.dataset.get_dataset(tenant_id, ds_id)
        if not ds:
            return fallback
        if ver <= 0:
            ver = ds["version"]
        ver_obj = self.dataset.get_dataset_version(tenant_id, ds_id, ver)
        if not ver_obj or not ver_obj.uri:
            return fallback
        base_uri = ver_obj.uri.rstrip("/")
        filer_base = self.s3.s3_uri_to_filer_path(base_uri)
        if not filer_base:
            return fallback
        has_yaml = self._object_exists(filer_base + "/data.yaml")
        has_class_names = self._object_exists(filer_base + "/class_names.json")
        has_flat_images = self._object_exists(filer_base + "/images/")
        has_flat_labels = self._object_exists(filer_base + "/labels/")
        has_split_train = self._object_exists(filer_base + "/train/images/") or \
            self._object_exists(filer_base + "/train/")
        # Canonical detection/segmentation: flat or preserved splits.
        if has_yaml and ((has_flat_images and has_flat_labels) or has_split_train):
            return base_uri + "/"
        # Classification ImageFolder with preserved train/val splits: the
        # train/<class>/ tree alone identifies it (YOLO split versions always
        # carry data.yaml and were returned by the branch above).
        if has_split_train:
            return base_uri + "/"
        # Canonical ImageFolder classification: class_names.json plus a flat
        # images/<class>/ tree.
        if has_class_names and has_flat_images:
            return base_uri + "/"
        categories = self.categories_from_dataset(filer_base)
        if categories is None:
            return fallback
        yaml_content = self.build_data_yaml(categories)
        try:
            self.s3.put_object(filer_base + "/data.yaml", yaml_content.encode())
        except Exception:
            return fallback
        return base_uri + "/"

    def categories_from_dataset(self, filer_base: str) -> Optional[list[str]]:
        """Extract category names from a dataset version's annotations.json."""
        try:
            data = self.s3.read_object(filer_base + "/annotations.json")
        except Exception:
            return None
        try:
            ann = json.loads(data)
        except ValueError:
            return None
        cats = [c.get("name", "") for c in ann.get("categories", []) if c.get("name")]
        return cats or None

    @staticmethod
    def build_data_yaml(categories: list[str]) -> str:
        """Generate a YOLO data.yaml string from category names."""
        lines = ["train: /data/images", "val: /data/images", "names:"]
        for i, c in enumerate(categories):
            lines.append(f"  {i}: {c}")
        lines.append(f"nc: {len(categories)}")
        return "\n".join(lines) + "\n"

    # -- curves -------------------------------------------------------------

    def _progress_history(self, tenant_id: str, job_id: str) -> list[dict]:
        """Per-epoch rows the torch/paddle trainers publish to progress.json.

        Used when the run wrote neither an Ultralytics results.csv nor an
        history.jsonl (the license-neutral in-tree trainers log epochs into
        the ``history`` array of progress.json instead).
        """
        filer = self.s3.tenant_filer_path(
            f"/buckets/nereidforge/models/{job_id}/progress.json", tenant_id
        )
        try:
            prog = json.loads(self.s3.read_object(filer))
        except Exception:
            return []
        rows = prog.get("history") if isinstance(prog, dict) else None
        if not isinstance(rows, list):
            return []
        # Keep scalar columns only (per_class is a nested dict and would break
        # the numeric curves table); epoch stays as the implicit x-axis.
        return _json_safe([
            {k: v for k, v in row.items()
             if isinstance(v, (int, float, str)) and not isinstance(v, bool)}
            for row in rows
            if isinstance(row, dict)
        ])

    @staticmethod
    def training_time_from_rows(rows: list[dict]) -> Optional[float]:
        """training_time_from_rows returns the greatest finite cumulative trainer time.

        Ultralytics records cumulative seconds in ``time``; native trainers use
        ``train_time_s``. The maximum is independent of CSV row order and of
        which epoch produced the best checkpoint. Invalid and non-positive
        values do not become a fabricated duration.
        """
        values: list[float] = []
        for row in rows:
            for key in ("train_time_s", "time"):
                raw = row.get(key)
                if raw in (None, ""):
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0:
                    values.append(value)
        return round(max(values), 1) if values else None

    def training_metrics(self, tenant_id: str, job_id: str, metrics: dict) -> dict:
        """training_metrics restores missing BYOT duration from its training curves.

        The trainer's own metrics.json remains authoritative. This read-only
        fallback also covers jobs completed before duration recovery existed,
        so no database rewrite is needed merely to render their detail page.
        """
        result = dict(metrics or {})
        if self.training_time_from_rows([result]) is not None:
            return result
        train_time = self.training_time_from_rows(self.training_curves(tenant_id, job_id))
        if train_time is not None:
            result["train_time_s"] = train_time
        return result

    def training_curves(self, tenant_id: str, job_id: str) -> list[dict]:
        """Per-epoch curves: Ultralytics results.csv, else progress.json."""
        filer = self.s3.tenant_filer_path(
            f"/buckets/nereidforge/models/{job_id}/tensorboard/results.csv", tenant_id
        )
        try:
            data = self.s3.read_object(filer)
        except Exception:
            data = b""
        if data:
            text = data.decode("utf-8", errors="ignore")
            try:
                reader = csv.DictReader(io.StringIO(text))
                rows = [dict(row) for row in reader]
                if rows:
                    return rows
            except Exception:
                pass
        return self._progress_history(tenant_id, job_id)

    def training_history(self, tenant_id: str, job_id: str) -> list[dict]:
        """Read live per-step history.jsonl (HF/timm trainers).

        Each line is one logged step: {step, epoch, loss, eval_loss?,
        learning_rate?}. LLM/VLM/timm trainers upload (throttled) while the
        run is in progress, giving the UI a live loss curve without
        Ultralytics results.csv.
        """
        filer = self.s3.tenant_filer_path(
            f"/buckets/nereidforge/models/{job_id}/history.jsonl", tenant_id
        )
        try:
            data = self.s3.read_object(filer)
        except Exception:
            data = b""
        out: list[dict] = []
        for line in data.decode("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict):
                out.append(_json_safe(row))
        if out:
            return out
        # Fallback: torch/paddle trainers log epochs into progress.json.
        # Project onto the HF-style keys the live charts render.
        return [
            {
                "epoch": row.get("epoch"),
                "loss": row.get("train_loss"),
                "learning_rate": row.get("lr"),
            }
            for row in self._progress_history(tenant_id, job_id)
            if row.get("train_loss") is not None
        ]

    # -- live trainer pod logs ---------------------------------------------

    def _train_log_client(self):
        """Lazily build the read-only trainer-pod log client."""
        existing = getattr(self, "_train_log_client_cached", None)
        if existing is None:
            from app.clients.k8s import K8sTrainLogClient

            existing = K8sTrainLogClient()
            self._train_log_client_cached = existing
        return existing

    def trainer_pods(self, tenant_id: str, job_id: str) -> list[dict]:
        """Return trainer pod summaries (name/phase) for one job."""
        client = self._train_log_client()
        if not client.available:
            return []
        try:
            return client.find_trainer_pods(job_id)
        except Exception:
            return []

    def trainer_logs(self, tenant_id: str, job_id: str, tail: int = 400) -> str:
        """Return concatenated trainer-pod stdout/stderr for one job.

        Empty string means no trainer pod was found yet (caller falls back to
        the KFP executor/orchestration logs).
        """
        client = self._train_log_client()
        if not client.available:
            return ""
        try:
            pods = client.find_trainer_pods(job_id)
        except Exception:
            return ""
        chunks: list[str] = []
        for pod in pods:
            name = pod.get("name", "")
            if not name:
                continue
            text = client.read_pod_logs(
                name, container=pod.get("container", ""), tail_lines=tail
            )
            if not text:
                continue
            if len(pods) > 1:
                chunks.append(f"===== pod: {name} ({pod.get('phase', '')}) =====\n{text}")
            else:
                chunks.append(text)
        return "\n".join(chunks)

    # -- tenant kfp helpers -------------------------------------------------

    def tenant_kfp_userid(self, tenant_id: str) -> str:
        """Look up the tenant's KFP user id."""
        from app.models.orm import Tenant

        t = self.db.get(Tenant, tenant_id)
        if t and t.kfp_userid:
            return t.kfp_userid
        # The platform super-admin owns the default tenant's KF profile
        # (nereidforge); KFP v2 derives the run namespace from this
        # userid's profile.
        return "admin@nereidai.local"

    def tenant_kfp_experiment(self, tenant_id: str) -> str:
        """Look up the tenant's KFP experiment id."""
        from app.models.orm import Tenant

        t = self.db.get(Tenant, tenant_id)
        if t and t.kfp_experiment_id:
            return t.kfp_experiment_id
        return ""

    # -- helpers ------------------------------------------------------------

    def _object_exists(self, filer_path: str) -> bool:
        try:
            self.s3.read_object(filer_path)
            return True
        except Exception:
            return False

    def _trainer_cfg(self) -> tuple[str, str, str, str, str, str, str]:
        """Read trainer resource config from system_config (defaults when unset)."""
        if self.trainer_config_fn:
            return self.trainer_config_fn()
        return "", "0", "4", "8", "8Gi", "16Gi", nf_image("trainer/torch-cpu")

    @staticmethod
    def _now():
        from datetime import datetime, timezone

        return datetime.now(timezone.utc)
