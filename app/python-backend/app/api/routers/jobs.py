# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Job routes: training / optimization / evaluation job management."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from app.api.helpers import error, get_svc, not_found
from app.core.util import paginate
from app.services import Services
from app.services.jobs import JOB_TYPES

router = APIRouter()


class JobMetricsResponse(BaseModel):
    """JobMetricsResponse preserves trainer-specific metrics and types duration."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)
    schema_: Optional[str] = Field(
        default=None, alias="schema", serialization_alias="schema",
    )
    train_time_s: Optional[float] = None


class TrainJobRuntime(BaseModel):
    """TrainJobRuntime identifies one cluster TrainJob and its current state."""

    name: str
    state: str
    created: Optional[str] = ""


class TrainerPodRuntime(BaseModel):
    """TrainerPodRuntime carries exact container bounds for duration fallback."""

    name: str
    phase: str
    container: str
    job: str
    node: str
    started: Optional[str] = ""
    finished: Optional[str] = ""


class TrainingRuntimeResponse(BaseModel):
    """TrainingRuntimeResponse maps one NF job to its KFP and trainer resources."""

    kfp_run_id: str
    trainjobs: list[TrainJobRuntime]
    pods: list[TrainerPodRuntime]


class JobCreate(BaseModel):
    """Create job request body (loose params/config)."""
    name: str
    description: Optional[str] = ""
    model_id: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_version: Optional[int] = None
    source_model_id: Optional[str] = None
    source_version: Optional[str] = None
    targets: Optional[list] = None
    config: Optional[dict] = None
    params: Optional[dict] = None
    custom_image: Optional[str] = None
    custom_script_uri: Optional[str] = None


class JobUpdate(BaseModel):
    """Update job request body."""
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[dict] = None
    params: Optional[dict] = None
    # Custom training selectors (I-11) are params, not Job columns; declared
    # here so an edit round-trips them (the service folds them into params).
    custom_image: Optional[str] = None
    custom_script_uri: Optional[str] = None


def _build_params(req: JobCreate) -> dict:
    """Assemble job params from the create body (loose, mirrors Go bodyMap+build)."""
    params: dict = {}
    if req.model_id:
        params["model_id"] = req.model_id
    if req.source_model_id:
        params["source_model_id"] = req.source_model_id
    if req.source_version:
        params["source_version"] = req.source_version
    if req.dataset_id:
        params["dataset_id"] = req.dataset_id
    if req.dataset_version is not None:
        params["dataset_version"] = str(req.dataset_version)
    if req.params:
        params.update(req.params)
    if req.custom_image:
        params["custom_image"] = req.custom_image
    if req.custom_script_uri:
        params["custom_script_uri"] = req.custom_script_uri
    return params


@router.post("/{job_type}/jobs")
async def create_job(job_type: str, req: JobCreate, request: Request, svc: Services = Depends(get_svc)):
    """Create a job of the given type."""
    if job_type not in JOB_TYPES:
        raise error(400, "invalid job type")
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    params = _build_params(req)
    if job_type == "optimization":
        if not params.get("dataset_id"):
            raise error(400, "optimization requires a dataset for evaluation and calibration")
        try:
            svc.job.validate_optimization_source(tenant, params)
        except Exception as e:
            raise error(400, str(e))
    cfg = req.config or {}
    # Extended capabilities are registered but not granted by default (ADR-015):
    # fail fast here with a clear message instead of letting the pipeline start
    # and fail at run time. Third-party images that announce nothing pass.
    try:
        svc.job.check_capability(tenant, params, cfg)
    except Exception as e:
        from app.services.entitlements import CapabilityLocked

        if isinstance(e, CapabilityLocked):
            raise error(402, str(e))
        raise
    if req.targets:
        # Optimization multi-target: create one job per target sharing a batch_id.

        from app.core.util import new_id

        batch_id = new_id("batch")
        items = []
        for t in req.targets:
            target = t.get("target_device") if isinstance(t, dict) else t
            p = dict(params)
            p["target_device"] = str(target or "cpu")
            if isinstance(t, dict):
                p.update({k: v for k, v in t.items() if k != "target_device"})
            job = svc.job.create_job_with_batch(
                job_type, tenant, req.name, req.description or "", batch_id, user, p, cfg
            )
            items.append(job)
        return {"items": items, "batch_id": batch_id}
    job = svc.job.create_job(job_type, tenant, req.name, req.description or "", user, params, cfg)
    return job


@router.get("/{job_type}/jobs")
async def list_jobs(
    job_type: str, request: Request, page: int = 1, page_size: int = 20,
    status: str = "", model_id: str = "", search: str = "", svc: Services = Depends(get_svc),
):
    """List jobs of a type with filters/search."""
    if job_type not in JOB_TYPES:
        raise error(400, "invalid job type")
    tenant = request.headers.get("X-Tenant-Id", "default")
    items, total = svc.job.list_jobs(job_type, tenant, page, page_size, status, model_id, search)
    return paginate(items, total, page, page_size)


@router.get("/{job_type}/jobs/{job_id}")
async def get_job(job_type: str, job_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Get a job by id."""
    if job_type not in JOB_TYPES:
        raise error(400, "invalid job type")
    tenant = request.headers.get("X-Tenant-Id", "default")
    job = svc.job.get_job_dict(job_type, tenant, job_id)
    if not job:
        raise not_found()
    return job


@router.patch("/{job_type}/jobs/{job_id}")
async def update_job(job_type: str, job_id: str, req: JobUpdate, request: Request, svc: Services = Depends(get_svc)):
    """Update a job."""
    if job_type not in JOB_TYPES:
        raise error(400, "invalid job type")
    tenant = request.headers.get("X-Tenant-Id", "default")
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    job = svc.job.update_job(job_type, tenant, job_id, patch)
    if not job:
        raise not_found()
    return job


@router.delete("/{job_type}/jobs/{job_id}")
async def delete_job(job_type: str, job_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Delete a job."""
    if job_type not in JOB_TYPES:
        raise error(400, "invalid job type")
    tenant = request.headers.get("X-Tenant-Id", "default")
    if not svc.job.delete_job(job_type, tenant, job_id):
        raise not_found()
    return {"deleted": True}


class GateOverrideRequest(BaseModel):
    """Force-register request body for a gate-rejected artifact."""
    reason: str


@router.post("/optimization/jobs/{job_id}/override")
async def override_gate(job_id: str, req: GateOverrideRequest, request: Request, svc: Services = Depends(get_svc)):
    """Override a hard-gate failure: force-register the rejected artifact.

    Permission: the job creator or an admin; reason is mandatory and is
    recorded in the audit log along with the gate values. Registered BEFORE
    the generic /{action} route so it takes precedence.
    """
    from app.models.orm import User

    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    db_user = svc.db.query(User).filter(User.username == user).first()
    is_admin = bool(db_user and "admin" in (db_user.roles or []))
    try:
        return svc.job.override_gate(tenant, job_id, user, is_admin, req.reason or "")
    except PermissionError as e:
        raise error(403, str(e))
    except ValueError as e:
        raise error(400, str(e))


@router.post("/{job_type}/jobs/{job_id}/{action}")
async def job_action(job_type: str, job_id: str, action: str, request: Request, svc: Services = Depends(get_svc)):
    """Perform a lifecycle action on a job (start/stop/cancel/restart)."""
    if job_type not in JOB_TYPES:
        raise error(400, "invalid job type")
    tenant = request.headers.get("X-Tenant-Id", "default")
    try:
        if action in ("start", "restart"):
            job = svc.job.start_job(job_type, tenant, job_id)
        elif action in ("stop", "cancel"):
            job = svc.job.stop_job(job_type, tenant, job_id)
        else:
            raise error(400, "invalid action")
    except ValueError as e:
        from app.services.jobs import OptimizationSourceError

        if isinstance(e, OptimizationSourceError):
            raise error(400, str(e))
        raise not_found() if "not found" in str(e) else error(500, str(e))
    except Exception as e:
        from app.services.entitlements import CapabilityLocked

        if isinstance(e, CapabilityLocked):
            raise error(402, str(e))
        raise
    return job


@router.get("/training/jobs/{job_id}/runtime", response_model=TrainingRuntimeResponse)
async def training_runtime(job_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Return cluster-side runtime objects for a training job.

    Maps the NF job to its KFP run id, Kubeflow TrainJob CR(s) and JobSet
    trainer pods (name/phase/node) so the UI can show the exact objects
    visible in kubectl.
    """
    tenant = request.headers.get("X-Tenant-Id", "default")
    runtime = svc.job.training_runtime(tenant, job_id)
    if runtime is None:
        raise not_found()
    return runtime


@router.get("/{job_type}/jobs/{job_id}/logs")
async def job_logs(job_type: str, job_id: str, request: Request, tail: int = 400, svc: Services = Depends(get_svc)):
    """Return job logs (DB logs, or the KFP run executor logs when available)."""
    if job_type not in JOB_TYPES:
        raise error(400, "invalid job type")
    tenant = request.headers.get("X-Tenant-Id", "default")
    job = svc.job.get_job_dict(job_type, tenant, job_id)
    if not job:
        raise not_found()
    logs = job.get("logs") or ""
    kfp_run_id = job.get("kfp_run_id") or ""
    # Prefer the real trainer-pod stdout (per-step loss, validation,
    # tracebacks). Fall back to KFP executor logs before the trainer pod
    # exists or when kubernetes log access is unavailable.
    trainer_logs = ""
    try:
        trainer_logs = svc.job.trainer_logs(tenant, job_id, tail=max(tail, 2000))
    except Exception:
        trainer_logs = ""
    if trainer_logs:
        return {"logs": trainer_logs}
    if kfp_run_id:
        try:
            kfp_logs = svc.kfp.get_run_logs(kfp_run_id, tail=tail)
        except Exception:
            kfp_logs = ""
        if kfp_logs:
            logs = kfp_logs
    return {"logs": logs}


@router.get(
    "/{job_type}/jobs/{job_id}/metrics",
    response_model=JobMetricsResponse,
    response_model_exclude_none=True,
)
async def job_metrics(job_type: str, job_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Return job metrics, recovering missing Ultralytics training duration."""
    if job_type not in JOB_TYPES:
        raise error(400, "invalid job type")
    tenant = request.headers.get("X-Tenant-Id", "default")
    job = svc.job.get_job_dict(job_type, tenant, job_id)
    if not job:
        raise not_found()
    metrics = job.get("metrics") or {}
    if job_type == "training":
        return svc.job.training_metrics(tenant, job_id, metrics)
    return metrics


@router.get("/{job_type}/jobs/{job_id}/curves")
async def job_curves(job_type: str, job_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Return training curves from SeaweedFS results.csv."""
    if job_type != "training":
        raise error(400, "curves only for training jobs")
    tenant = request.headers.get("X-Tenant-Id", "default")
    return {
        "curves": svc.job.training_curves(tenant, job_id),
        "history": svc.job.training_history(tenant, job_id),
    }


@router.get("/{job_type}/jobs/{job_id}/results")
async def job_results(job_type: str, job_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Return job results."""
    if job_type not in JOB_TYPES:
        raise error(400, "invalid job type")
    tenant = request.headers.get("X-Tenant-Id", "default")
    job = svc.job.get_job_dict(job_type, tenant, job_id)
    if not job:
        raise not_found()
    return job.get("results") or {}


@router.get("/{job_type}/jobs/{job_id}/checkpoints")
async def job_checkpoints(job_type: str, job_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Return job checkpoints (placeholder)."""
    return {"checkpoints": []}


@router.get("/{job_type}/jobs/{job_id}/download")
async def job_download(job_type: str, job_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Stream the job's downloadable artifact from storage.

    The browser never talks to seaweedfs directly (platform rule): the API
    proxies the bytes. For optimization jobs that is the optimized model
    file; for evaluation jobs the report JSON.
    """
    from fastapi.responses import StreamingResponse

    if job_type not in JOB_TYPES:
        raise error(400, "invalid job type")
    tenant = request.headers.get("X-Tenant-Id", "default")
    job = svc.job.get_job_dict(job_type, tenant, job_id)
    if not job:
        raise not_found()
    uri = ""
    metrics = job.get("metrics") or {}
    if job_type == "optimization":
        uri = str(metrics.get("optimized_uri") or "")
        if not uri.startswith("s3://"):
            # Deterministic artifact location even when the job failed a gate:
            # optimized/{batch_id or job_id}/{job_id}/{job_id}.optimized.onnx
            batch = job.get("batch_id") or job_id
            uri = svc.s3.tenant_uri(
                f"s3://nereidforge/optimized/{batch}/{job_id}/{job_id}.optimized.onnx",
                tenant,
            )
    elif job_type == "evaluation":
        # Report location is deterministic: evaluation-reports/{job}/report.json
        uri = svc.s3.tenant_uri(
            f"s3://nereidforge/evaluation-reports/{job_id}/report.json", tenant)
    if not uri.startswith("s3://"):
        raise error(404, "no downloadable artifact for this job")

    # OpenVINO IR is a (xml, bin) pair; when extra artifacts exist, bundle the
    # main file + extras into a zip so the model is fully usable after download.
    extra_uris: list[str] = []
    if job_type == "optimization":
        extra_uris = [
            u for u in (metrics.get("extra_artifacts") or [])
            if isinstance(u, str) and u.startswith("s3://")
        ]

    def _read(uri: str) -> bytes:
        return svc.s3.read_object(svc.s3.s3_uri_to_filer_path(uri))

    if extra_uris:
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(uri.rsplit("/", 1)[-1] or "model.xml", _read(uri))
            for u in extra_uris:
                zf.writestr(u.rsplit("/", 1)[-1] or "artifact.bin", _read(u))
        fname = f"{job_id}.zip"
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    filer_path = svc.s3.s3_uri_to_filer_path(uri)
    try:
        data = svc.s3.read_object(filer_path)
    except Exception as e:
        raise error(404, f"artifact not found: {e}")
    fname = uri.rsplit("/", 1)[-1] or "artifact"
    return StreamingResponse(
        iter([data]), media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
