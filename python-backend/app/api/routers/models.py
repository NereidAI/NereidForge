# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Model routes: model/version management."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api.helpers import error, get_svc, not_found, require_platform_super_admin
from app.core.config import config
from app.core.images import serving_image
from app.core.util import paginate
from app.services import Services
from app.services.model_format import (
    FORMAT_OPENVINO_IR,
    FORMAT_TENSORRT_PLAN,
    artifact_capability_permitted,
    artifact_format_catalog,
    serving_format,
)
from app.services.model_upload import ArtifactUploadError, prepare_artifact_upload
from app.services.models import ModelDeleteConflict, ModelTrialError

router = APIRouter()

log = logging.getLogger("app.routers.models")


def _tenant_namespace(tenant: str, explicit: str = "") -> str:
    """Resolve the target cluster namespace for a tenant (default -> admin ns)."""
    if explicit:
        return explicit
    if tenant and tenant != "default":
        return f"nereidforge-{tenant}"
    return config.serving.namespace or "nereidforge"


class ModelCreate(BaseModel):
    """Create model request body."""
    name: str
    description: Optional[str] = ""
    framework: Optional[str] = "pytorch"
    task_type: Optional[str] = "detection"


class ModelUpdate(BaseModel):
    """Update model request body."""
    name: Optional[str] = None
    description: Optional[str] = None
    framework: Optional[str] = None
    task_type: Optional[str] = None


class VersionCreate(BaseModel):
    """Create model version request body (uploaded file path)."""
    version: str
    description: Optional[str] = ""
    file_path: Optional[str] = ""
    size_bytes: Optional[int] = 0
    metadata: Optional[dict] = None


class ArtifactFormatDescriptor(BaseModel):
    """ArtifactFormatDescriptor exposes accepted suffixes and implemented workflows."""

    id: str
    label: str
    extensions: list[str]
    primary_extensions: list[str]
    capabilities: list[str]
    compiled: bool


class ArtifactFormatCatalogResponse(BaseModel):
    """ArtifactFormatCatalogResponse is the frontend's upload allowlist and capability source."""

    formats: list[ArtifactFormatDescriptor]
    archive_suffixes: list[str]


class EdgePackageDeviceResponse(BaseModel):
    """EdgePackageDeviceResponse summarizes one executed or unavailable device check."""

    device: str
    status: str


class EdgePackageSummaryResponse(BaseModel):
    """EdgePackageSummaryResponse exposes one ready package without requiring tar inspection."""

    precision: str
    status: str
    revision: Optional[str] = None
    package_uri: Optional[str] = None
    package_sha256: Optional[str] = None
    package_bytes: Optional[int] = None
    detection_count: Optional[int] = None
    devices: list[EdgePackageDeviceResponse] = Field(default_factory=list)
    quality_gate_passed: Optional[bool] = None
    map50_baseline: Optional[float] = None
    map50_candidate: Optional[float] = None


class EdgePackageListResponse(BaseModel):
    """EdgePackageListResponse identifies a model version and all downloadable packages."""

    model_id: str
    version: str
    packages: list[EdgePackageSummaryResponse] = Field(default_factory=list)


class ModelVersionResponse(BaseModel):
    """ModelVersionResponse describes a registered version and its artifact metadata."""

    model_config = ConfigDict(extra="allow")

    id: str
    model_id: str
    version: str
    description: str = ""
    file_path: str = ""
    size_bytes: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: str
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    artifacts_count: int = 0
    created_by: str = ""
    created_at: str = ""


class ModelUploadErrorResponse(BaseModel):
    """ModelUploadErrorResponse reports validation failures before version registration."""

    detail: str


class ModelDeleteResponse(BaseModel):
    """ModelDeleteResponse confirms that registry records were removed or archived."""

    deleted: bool


class ModelDeleteErrorResponse(BaseModel):
    """ModelDeleteErrorResponse describes authorization, lookup, or liveness failures."""

    detail: str


_MODEL_DELETE_RESPONSES = {
    401: {"model": ModelDeleteErrorResponse, "description": "Authentication required"},
    403: {"model": ModelDeleteErrorResponse, "description": "Platform super administrator required"},
    404: {"model": ModelDeleteErrorResponse, "description": "Model or version not found"},
    409: {"model": ModelDeleteErrorResponse, "description": "A version is deployed or transitioning"},
}


class DeployRequest(BaseModel):
    """Deploy model version request body."""
    target: Optional[str] = "cpu"
    runtime: Optional[str] = None
    runtime_type: Optional[str] = None  # frontend sends runtime_type (openvino/triton)
    runtime_image: Optional[str] = None
    gpu: Optional[str] = None
    domain: Optional[str] = None
    # Optional: override the model URI (e.g., for deploying a specific artifact URI directly)
    model_uri: Optional[str] = None


class TrialDecoderSelection(BaseModel):
    """TrialDecoderSelection stores declarative decoder choices without executable code."""

    model_config = ConfigDict(extra="allow")

    mode: str
    decoder_id: Optional[str] = None
    task: Optional[str] = None
    input: Optional[dict[str, Any]] = None
    output: Optional[dict[str, Any]] = None
    labels: Optional[list[str]] = None


class TrialTensor(BaseModel):
    """TrialTensor describes one named model tensor and its static or symbolic shape."""

    model_config = ConfigDict(extra="allow")

    name: str
    shape: list[int | str | None]
    dtype: str


class TrialDecoderDescriptor(BaseModel):
    """TrialDecoderDescriptor describes one decoder and its graph-match state."""

    model_config = ConfigDict(extra="allow")

    id: str
    label: str
    task: str
    matched: bool


class TrialAutoDecoder(BaseModel):
    """TrialAutoDecoder identifies the decoder selected by ML Backend auto-discovery."""

    model_config = ConfigDict(extra="allow")

    id: str
    task: str


class TrialSchemaResponse(BaseModel):
    """TrialSchemaResponse combines ML inspection with NF's per-artifact saved selection."""

    model_config = ConfigDict(extra="allow")

    artifact: str
    inputs: list[TrialTensor]
    outputs: list[TrialTensor]
    decoders: list[TrialDecoderDescriptor]
    auto_decoder: Optional[TrialAutoDecoder]
    mapping_required: bool
    names: dict[str, str]
    saved_decoder: Optional[TrialDecoderSelection]
    input_shape: Optional[list[int]] = None


class ModelTrialResponse(BaseModel):
    """ModelTrialResponse normalizes common fields while retaining task-specific extras."""

    model_config = ConfigDict(extra="allow")

    task: str
    kind: str
    width: int
    height: int
    names: dict[str, str]
    boxes: list[dict[str, Any]] = Field(default_factory=list)
    polygons: list[dict[str, Any]] = Field(default_factory=list)
    mask_overlay: Optional[str] = None
    mask_legend: list[dict[str, Any]] = Field(default_factory=list)
    probs: Optional[dict[str, Any]] = None
    decoder: dict[str, Any]
    heatmap: Optional[str] = None
    time_ms: float


class TrialErrorResponse(BaseModel):
    """TrialErrorResponse keeps readable detail alongside the structured ML error code."""

    detail: str
    error: Optional[str] = None
    code: Optional[str] = None


_TRIAL_ERROR_RESPONSES = {
    400: {"model": TrialErrorResponse, "description": "Invalid trial request or artifact"},
    422: {"model": TrialErrorResponse, "description": "Decoder validation or mapping required"},
    500: {"model": TrialErrorResponse, "description": "ML Backend inference failed"},
}


def _trial_error_response(exc: ModelTrialError) -> JSONResponse:
    """_trial_error_response preserves the service error code alongside readable detail."""
    content: dict[str, str] = {"detail": exc.detail, "error": exc.detail}
    if exc.code:
        content["code"] = exc.code
    return JSONResponse(status_code=exc.status_code, content=content)


@router.get("/models")
async def list_models(
    request: Request, page: int = 1, page_size: int = 20,
    framework: str = "", task_type: str = "", search: str = "", svc: Services = Depends(get_svc),
):
    """List models with filters/search and pagination."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    items, total = svc.model.list_models(tenant, page, page_size, framework, task_type, search)
    return paginate(items, total, page, page_size)


@router.post("/models")
async def create_model(req: ModelCreate, request: Request, svc: Services = Depends(get_svc)):
    """Create a model."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    return svc.model.create_model(tenant, req.name, req.description or "", req.framework, req.task_type, user)


@router.get("/models/trainable-versions")
async def list_trainable_versions(
    request: Request, task_type: str = "", svc: Services = Depends(get_svc)
):
    """List model versions usable as a base for continued training."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    items = svc.model.list_trainable_versions(tenant, task_type)
    return {"items": items, "total": len(items)}


@router.get("/models/serving-runtimes")
async def list_serving_runtimes(request: Request, svc: Services = Depends(get_svc)):
    """List serving runtimes from the KServe ClusterServingRuntime CRDs."""
    items = await asyncio.to_thread(svc.k8s.list_serving_runtimes)
    return {"items": items}


@router.get("/models/deployments")
async def list_deployments(request: Request, svc: Services = Depends(get_svc)):
    """List deployed/deploying/failed inference services (DB rows + live KServe status)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    services = await asyncio.to_thread(svc.model.list_deployed, tenant)
    return {"services": services}


@router.post("/models/deployments/{name}/undeploy")
async def undeploy_service(name: str, request: Request, svc: Services = Depends(get_svc)):
    """Undeploy an inference service by name (deletes ISVC + ingress via KFP).

    Async: submits the KFP undeploy run (wait=False, ~1-3s), marks the version
    "terminating" and enqueues a model_undeploy verification task. The worker
    only clears the DB row after confirming the live InferenceService is gone,
    so a failed/skipped deletion can never leave DB and cluster divergent.
    """
    from app.core.util import new_id
    from app.models.orm import AsyncTask

    tenant = request.headers.get("X-Tenant-Id", "default")
    ns = _tenant_namespace(tenant, request.query_params.get("namespace", ""))
    row = next(
        (s for s in svc.model.list_deployed(tenant) if s.get("name") == name), None
    )
    if not row:
        raise not_found()
    if str(row.get("state", "")) == "terminating":
        raise error(409, "undeploy already in progress for this service")
    svc.model.cancel_deploy_task(tenant, row["model_id"], row["model_version"])
    try:
        run_id = await asyncio.to_thread(svc.k8s.delete_inference_service, name, ns, False)
    except Exception as e:
        log.error("undeploy submit failed for %s: %s", name, e)
        raise error(500, f"undeploy failed: {e}")
    task_id = _enqueue_undeploy_verification(
        svc, AsyncTask, new_id, tenant, row["model_id"], row["model_version"],
        name, ns, run_id,
    )
    log.info("undeploy submitted: name=%s ns=%s run=%s task=%s", name, ns, run_id, task_id)
    return {"undeployed": True, "name": name, "run_id": run_id, "task_id": task_id}


def _enqueue_undeploy_verification(
    svc: Services, async_task_cls, new_id_fn,
    tenant: str, model_id: str, version: str, service: str, ns: str, run_id: str,
) -> str:
    """Mark a version terminating and enqueue its live-deletion verification."""
    task_id = new_id_fn("task")
    svc.model.mark_terminating(tenant, model_id, version, task_id)
    task = async_task_cls(
        id=task_id,
        tenant_id=tenant,
        type="model_undeploy",
        status="pending",
        progress=0,
        params={
            "tenant": tenant,
            "model_id": model_id,
            "version": version,
            "service": service,
            "namespace": ns,
            "run_id": run_id,
        },
        created_by="",
    )
    svc.db.add(task)
    svc.db.commit()
    return task_id


@router.get(
    "/models/artifact-formats",
    response_model=ArtifactFormatCatalogResponse,
    summary="List supported model artifact formats and capabilities",
)
async def list_artifact_formats() -> ArtifactFormatCatalogResponse:
    """Return the authoritative upload registry without requiring cluster access."""
    return ArtifactFormatCatalogResponse.model_validate(artifact_format_catalog())


@router.get("/models/{model_name}")
async def get_model(model_name: str, request: Request, svc: Services = Depends(get_svc)):
    """Get a model by id or name."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    return m


@router.patch("/models/{model_name}")
async def update_model(model_name: str, req: ModelUpdate, request: Request, svc: Services = Depends(get_svc)):
    """Update a model."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    m = svc.model.update_model(tenant, model_name, patch)
    if not m:
        raise not_found()
    return m


@router.delete(
    "/models/{model_name}",
    response_model=ModelDeleteResponse,
    responses=_MODEL_DELETE_RESPONSES,
    summary="Delete a model and all of its registry versions",
    dependencies=[Depends(require_platform_super_admin)],
)
async def delete_model(model_name: str, request: Request, svc: Services = Depends(get_svc)):
    """Archive the Hub model and delete local records while retaining artifact bytes."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    try:
        svc.model.delete_model(tenant, m["id"])
    except ModelDeleteConflict as exc:
        raise error(409, str(exc)) from exc
    return {"deleted": True}


# -- versions ---------------------------------------------------------------


@router.post(
    "/models/{model_name}/versions",
    response_model=ModelVersionResponse,
    responses={
        400: {"model": ModelUploadErrorResponse, "description": "Invalid or incomplete artifact"},
        404: {"model": ModelUploadErrorResponse, "description": "Model not found"},
    },
    summary="Upload and register a model version artifact",
)
async def create_model_version(model_name: str, request: Request, svc: Services = Depends(get_svc)):
    """Validate and stream one file, a file group, or a safe archive into S3.

    Direct checkpoints retain Starlette's spooled streaming behavior. Archives
    are completely extracted and validated before the first S3 write. Blocking
    storage, database, and hub calls run in the worker threadpool so long
    uploads do not stall readiness probes or other requests.
    """
    from fastapi.concurrency import run_in_threadpool

    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    m = await run_in_threadpool(svc.model.get_model, tenant, model_name)
    if not m:
        raise not_found()
    form = await request.form()
    version = str(form.get("version") or "")
    description = str(form.get("description") or "")
    if not version:
        raise error(400, "version is required")
    metadata: dict[str, Any] = {}
    meta_raw = form.get("metadata")
    if meta_raw:
        try:
            parsed = json.loads(str(meta_raw))
        except (TypeError, ValueError) as exc:
            raise error(400, "metadata must be a valid JSON object") from exc
        if not isinstance(parsed, dict):
            raise error(400, "metadata must be a valid JSON object")
        metadata = parsed
    metadata.setdefault("framework", m["framework"])
    metadata.setdefault("uploaded_by", user)
    uploads = [
        item for item in form.getlist("file")
        if getattr(item, "filename", None) is not None
    ]

    def _persist():
        """Commit only a fully validated artifact set and its canonical metadata."""
        if not uploads:
            # Preserve the legacy metadata-only registration path, but discard
            # reserved fields whose values are proved only by artifact validation.
            for reserved in (
                "artifact_format", "artifact_layout", "artifact_capabilities",
                "bundle_uri", "primary_file",
            ):
                metadata.pop(reserved, None)
            return svc.model.create_version(
                tenant, m["id"], version, description, "", 0, metadata, user,
            )

        with prepare_artifact_upload(uploads, metadata) as prepared:
            bucket = config.storage.bucket or "nereidforge"
            bundle_key = f"models/{m['name']}/{version}"
            bundle_uri = f"s3://{bucket}/{bundle_key}/"
            for artifact in prepared.files:
                filer = f"/buckets/{bucket}/{bundle_key}/{artifact.relative_path}"
                with artifact.open() as stream:
                    svc.s3.upload_fileobj(filer, stream)
            file_path = bundle_uri + prepared.primary_path
            metadata.update({
                "artifact_format": prepared.format_id,
                "artifact_layout": prepared.layout,
                "artifact_capabilities": prepared.capabilities,
                "bundle_uri": bundle_uri,
                "primary_file": prepared.primary_path,
            })
            if prepared.runtime_requirements:
                metadata["runtime_requirements"] = prepared.runtime_requirements
            v = svc.model.create_version(
                tenant, m["id"], version, description, file_path,
                prepared.size_bytes, metadata, user,
            )
            # Hub registration is best-effort: the local version remains the
            # source of truth and capabilities still come from validation.
            try:
                svc.model.register_to_hub(
                    m["name"], file_path, version,
                    str(metadata.get("model_format_version", "1")), metadata,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("hub register failed for %s/%s: %s", m["name"], version, e)
            return v

    try:
        return await run_in_threadpool(_persist)
    except ArtifactUploadError as exc:
        raise error(400, str(exc)) from exc


@router.get("/models/{model_name}/versions")
async def list_model_versions(
    model_name: str, request: Request,
    page: int = 1, page_size: int = 20,
    svc: Services = Depends(get_svc),
):
    """List a model's versions (paginated, newest first)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    items, total = svc.model.list_versions_paged(tenant, m["id"], page, page_size)
    return paginate(items, total, page, page_size)


@router.get(
    "/models/{model_name}/versions/{version}",
    response_model=ModelVersionResponse,
    summary="Get a registered model version",
)
async def get_model_version(model_name: str, version: str, request: Request, svc: Services = Depends(get_svc)):
    """Get a model version."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    v = svc.model.get_version(tenant, m["id"], version)
    if not v:
        raise not_found()
    return v


@router.delete(
    "/models/{model_name}/versions/{version}",
    response_model=ModelDeleteResponse,
    responses=_MODEL_DELETE_RESPONSES,
    summary="Delete one model registry version",
    dependencies=[Depends(require_platform_super_admin)],
)
async def delete_model_version(model_name: str, version: str, request: Request, svc: Services = Depends(get_svc)):
    """Archive the Hub version and delete its local record while retaining artifacts."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    try:
        deleted = svc.model.delete_version(tenant, m["id"], version)
    except ModelDeleteConflict as exc:
        raise error(409, str(exc)) from exc
    if not deleted:
        raise not_found()
    return {"deleted": True}


@router.get("/models/{model_name}/versions/{version}/files")
async def list_model_version_files(model_name: str, version: str, request: Request, svc: Services = Depends(get_svc)):
    """List every file in a version's S3 prefix (the full artifact bundle).

    The versions table only shows the single primary file_path (ONNX after
    deploy); this endpoint exposes the siblings (original .pt, extra exports).
    """
    tenant = request.headers.get("X-Tenant-Id", "default")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    files = svc.model.list_version_files(tenant, m["id"], version)
    if files is None:
        raise not_found()
    return {"model_name": model_name, "version": version, "files": files, "total": len(files)}


@router.get("/models/{model_name}/versions/{version}/download")
async def download_model_version(
    model_name: str, version: str, request: Request,
    path: Optional[str] = None,
    svc: Services = Depends(get_svc),
):
    """Stream a model version's artifact bytes.

    Without ``path`` this streams the registered primary artifact; with
    ``path`` it streams one named sibling from the version's S3 prefix (see
    GET .../files).
    """
    tenant = request.headers.get("X-Tenant-Id", "default")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    v = svc.model.get_version(tenant, m["id"], version)
    if not v or not v.get("file_path"):
        raise not_found()
    import io

    buf = io.BytesIO()
    if path:
        uri = svc.model.stream_version_file(tenant, m["id"], version, path, buf)
        if uri is None:
            raise not_found()
        fname = uri.rstrip("/").rsplit("/", 1)[-1]
    else:
        svc.model.stream_artifact(tenant, m["id"], version, buf)
        # Preserve the artifact's real extension (onnx/pt/gguf/...) instead of .bin
        file_path = v.get("file_path") or ""
        fname = (file_path.rstrip("/").rsplit("/", 1)[-1] or f"{m['name']}-{version}")
        # IR weights companions are fetched individually via ?path=.bin; the
        # primary IR download is the .xml graph (the files API also lists bin).
        if not fname.lower().endswith((
            ".onnx", ".pt", ".gguf", ".tflite", ".ir", ".plan",
            ".xml", ".bin", ".engine", ".trt",
        )):
            fname = f"{fname}.bin"
    return StreamingResponse(iter([buf.getvalue()]), media_type="application/octet-stream",
                             headers={"Content-Disposition": f"attachment; filename={fname}"})


@router.get(
    "/models/{model_name}/versions/{version}/packages",
    response_model=EdgePackageListResponse,
    summary="List ready edge model packages",
)
async def list_model_packages(
    model_name: str, version: str, request: Request,
    svc: Services = Depends(get_svc),
):
    """List the version's precision packages available for OH passthrough.

    Returns one entry per ready ``nereid-model`` package (FP32/FP16/INT8) with
    its revision, package digest, quality gate, and device-verification summary.
    This read-only endpoint feeds NH/NL discovery and operator inspection; it
    never triggers a build or invents a numerical-verification result.
    """
    from app.services.model_packages import ModelPackageService

    tenant = request.headers.get("X-Tenant-Id", "default")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    v = svc.model.get_version(tenant, m["id"], version)
    if not v:
        raise not_found()
    records = ModelPackageService.available_packages(v.get("metadata") or {})
    return {
        "model_id": m["id"],
        "version": version,
        "packages": [
            {
                "precision": str(r.get("precision") or "").upper(),
                "status": r.get("status"),
                "revision": r.get("revision"),
                "package_uri": r.get("package_uri"),
                "package_sha256": r.get("package_sha256"),
                "package_bytes": r.get("package_bytes"),
                "detection_count": r.get("detection_count"),
                "devices": r.get("devices") or [],
                "quality_gate_passed": r.get("quality_gate_passed"),
                "map50_baseline": r.get("map50_baseline"),
                "map50_candidate": r.get("map50_candidate"),
            }
            for r in records
        ],
    }


@router.get(
    "/models/{model_name}/versions/{version}/package",
    summary="Download a ready edge model package",
)
async def download_model_package(
    model_name: str, version: str, request: Request, precision: str = "fp16",
    svc: Services = Depends(get_svc),
):
    """Download a ready precision package (``.tar.gz``) as byte passthrough.

    NH/NL streams these bytes unchanged; ``precision`` is ``fp32``, ``fp16``
    (default), or ``int8``. Device selection never appears here. Direct ONNX
    FP32 packages and verified IR precision packages share this byte contract.
    """
    import io

    from app.services.model_packages import PRECISIONS, ModelPackageService

    tenant = request.headers.get("X-Tenant-Id", "default")
    precision = precision.strip().lower()
    if precision not in PRECISIONS:
        raise error(400, f"precision must be one of: {', '.join(sorted(PRECISIONS))}")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    v = svc.model.get_version(tenant, m["id"], version)
    if not v:
        raise not_found()
    records = ModelPackageService.list_package_records(v.get("metadata") or {})
    rec = next((r for r in records if r.get("precision") == precision), None)
    if not rec:
        raise error(404, f"no {precision} package has been built for this version")
    if rec.get("status") != "ready":
        raise error(409, f"{precision} package is {rec.get('status', 'unknown')}")
    uri = str(rec.get("package_uri") or "")
    if not uri.startswith("s3://"):
        raise error(404, "package URI is missing")
    try:
        data = svc.s3.read_object(svc.s3.s3_uri_to_filer_path(uri))
    except Exception as e:  # noqa: BLE001
        raise error(404, f"package object is missing in object storage: {e}")
    fname = uri.rstrip("/").rsplit("/", 1)[-1] or f"{model_name}-{version}-{precision}.tar.gz"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/gzip",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


@router.get(
    "/models/{model_name}/versions/{version}/try/schema",
    response_model=TrialSchemaResponse,
    response_model_exclude_unset=True,
    summary="Inspect a model trial artifact and its decoder options",
    responses=_TRIAL_ERROR_RESPONSES,
)
async def inspect_model_trial_schema(
    model_name: str,
    version: str,
    request: Request,
    path: Optional[str] = None,
    svc: Services = Depends(get_svc),
) -> dict[str, Any] | JSONResponse:
    """inspect_model_trial_schema inspects the artifact resolved by the trial endpoint."""
    from fastapi.concurrency import run_in_threadpool

    tenant = request.headers.get("X-Tenant-Id", "default")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    try:
        return await run_in_threadpool(
            svc.model.inspect_trial, tenant, m["id"], version, artifact_path=path
        )
    except ModelTrialError as exc:
        return _trial_error_response(exc)
    except ValueError as exc:
        raise error(400, str(exc)) from exc


@router.post(
    "/models/{model_name}/versions/{version}/try",
    response_model=ModelTrialResponse,
    response_model_exclude_unset=True,
    summary="Run an image trial with an optional decoder selection",
    responses=_TRIAL_ERROR_RESPONSES,
)
async def try_model_version(
    model_name: str,
    version: str,
    request: Request,
    image: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.7),
    cam: bool = Form(False),
    class_index: Optional[int] = Form(None),
    decoder_config: Optional[str] = Form(None),
    top_k: int = Form(5, ge=1, le=100),
    path: Optional[str] = None,
    svc: Services = Depends(get_svc),
) -> dict[str, Any] | JSONResponse:
    """try_model_version runs one uploaded image without creating a deployment.

    The optional ``path`` query selects one specific file from the version's
    artifact prefix. ``decoder_config`` is a JSON object; when omitted, the
    service reuses the artifact's saved selection or requests automatic decoding.
    """
    import base64
    from fastapi.concurrency import run_in_threadpool

    tenant = request.headers.get("X-Tenant-Id", "default")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    raw = await image.read()
    if not raw:
        raise error(400, "empty image")

    parsed_decoder: Optional[dict[str, Any]] = None
    if decoder_config is not None:
        try:
            candidate = json.loads(decoder_config)
        except (json.JSONDecodeError, TypeError) as exc:
            raise error(400, "decoder_config must be a JSON object") from exc
        if not isinstance(candidate, dict):
            raise error(400, "decoder_config must be a JSON object")
        parsed_decoder = candidate

    try:
        result = await run_in_threadpool(
            svc.model.try_predict,
            tenant, m["id"], version,
            base64.b64encode(raw).decode("ascii"),
            conf=conf, iou=iou, cam=cam, class_index=class_index,
            artifact_path=path, decoder_config=parsed_decoder, top_k=top_k,
        )
    except ModelTrialError as exc:
        return _trial_error_response(exc)
    except ValueError as exc:
        # User-fixable trial errors (no weights, non-.pt Grad-CAM, unknown
        # artifact path) surface as 400 with the real message, not a 500.
        raise error(400, str(exc)) from exc
    return result


@router.post("/models/{model_name}/versions/{version}/deploy")
async def deploy_model_version(
    model_name: str, version: str, req: DeployRequest, request: Request, svc: Services = Depends(get_svc),
):
    """Deploy a model version to KServe (async: enqueues a model_deploy task).

    The response returns immediately (milliseconds) with the predicted
    endpoint. The worker performs the long ONNX conversion (5-10 minutes for
    big artifacts) and the KServe rollout; the UI follows progress through
    GET .../serving. IngressRoute + InferenceService are created together by
    the deploy pipeline once conversion finishes.
    """
    import logging
    from app.clients.k8s import sanitize_k8s_name, serving_isvc_name
    from app.core.util import new_id
    from app.models.orm import AsyncTask

    log = logging.getLogger("app.deploy")
    log.info("deploy enqueue: model=%s version=%s runtime=%s", model_name, version, req.runtime_type)

    tenant = request.headers.get("X-Tenant-Id", "default")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    v = svc.model.get_version(tenant, m["id"], version)
    if not v:
        # The model may exist only in the hub without a PG artifact record;
        # deployment needs a physical artifact (S3 file), so surface a clear error.
        raise error(400, "model version has no deployable artifact in NereidForge storage")
    md = v.get("metadata") or {}
    if not artifact_capability_permitted(md, "deploy"):
        artifact_kind = str(md.get("artifact_format") or "this artifact")
        raise error(
            400,
            f"{artifact_kind} is registration/download only in this release; "
            "no compatible deployment workflow is available",
        )

    # Use model_uri from request if provided, otherwise fall back to version metadata
    model_uri = req.model_uri or str(md.get("onnx_uri", "") or md.get("weights_uri", "") or v.get("file_path", ""))
    if not model_uri:
        log.error("deploy failed: no artifact found for model=%s version=%s metadata=%s", model_name, version, md)
        raise error(400, "model has no deployable artifact (onnx/pt)")
    log.info("deploy: model_uri=%s", model_uri)

    explicit_runtime = (req.runtime_type or req.runtime or "").strip()
    runtime_type = explicit_runtime or "kserve-tritonserver"

    # Vendor-native artifacts are bound to their matching serving runtime:
    # OpenVINO IR (.xml/.bin) only works on OVMS. When the caller leaves the
    # choice to the backend we auto-select; an explicitly different runtime is
    # a user error worth surfacing (silent rewrite would hide a UI mismatch).
    artifact_fmt = serving_format(model_uri)
    if artifact_fmt == FORMAT_OPENVINO_IR:
        if explicit_runtime and "openvino" not in explicit_runtime.lower():
            raise error(
                400,
                "this version is an OpenVINO IR artifact (.xml/.bin, produced by "
                "an Intel/OpenVINO optimization job); deploy it with the OpenVINO "
                "runtime instead",
            )
        runtime_type = "kserve-openvino"
    elif artifact_fmt == FORMAT_TENSORRT_PLAN:
        # TensorRT engines will be served by Triton's tensorrt_plan backend;
        # the Triton IR layout for plans lands with the NVIDIA optimization work.
        raise error(
            400,
            "TensorRT engine (.plan) serving is not available in this release; "
            "deploy the ONNX version or run an NVIDIA optimization after the "
            "TensorRT serving path ships",
        )

    runtime_image = req.runtime_image or ""
    if not runtime_image:
        # Infer the image from the configured serving.runtime_images mapping
        # (customer-tunable; falls back to triton).
        rt_lower = runtime_type.lower()
        img_url = config.serving.runtime_images.get(
            "triton" if "triton" in rt_lower else
            ("openvino" if "openvino" in rt_lower else
             ("torch" if ("torch" in rt_lower or "pytorch" in rt_lower) else
              ("huggingface" if "huggingface" in rt_lower else "")))
        )
        runtime_image = img_url or serving_image("triton")

    is_name = serving_isvc_name(m["id"], version)
    ns = config.serving.namespace or "nereidforge"
    if tenant and tenant != "default":
        ns = f"nereidforge-{tenant}"

    # Build inference domain from platform_domain + serving.domain_suffix when
    # configured. system_config (Admin settings) wins; fall back to the
    # serving.platform_domain from the config file. Without either, no
    # IngressRoute is created and the predictor stays cluster-internal.
    domain = ""
    syscfg = svc.admin.get_system_config()
    platform_domain = str(syscfg.get("platform_domain", "") or config.serving.platform_domain or "")
    if platform_domain:
        model_type = sanitize_k8s_name(m["id"])[:24]
        short_ver = sanitize_k8s_name(version)
        if len(short_ver) > 12:
            short_ver = short_ver[-12:]
        suffix = config.serving.domain_suffix or "inference"
        domain = f"{model_type}-{short_ver}.{suffix}.{platform_domain}"

    # Reject duplicate submits: one in-flight deploy per model version. The
    # worker flips the status to deployed/deploy_failed when it finishes.
    if (v.get("status") or "") == "deploying":
        raise error(409, "a deployment is already in progress for this model version")

    # Conversion details ride along in the task params; the worker runs the
    # (potentially 5-10 minute) ONNX export before submitting the deploy KFP
    # run. The HTTP request never waits for either stage.
    source = str(md.get("source", "") or "auto").strip().lower()
    architecture = str(md.get("architecture", "") or "").strip()
    task_id = new_id("task")
    task_params = {
        "tenant": tenant,
        "model_id": m["id"],
        "model_name": model_name,
        "version": version,
        "model_uri": model_uri,
        "runtime_type": runtime_type,
        "runtime_image": runtime_image,
        # Device selection: req.gpu carries the vendor/resource chosen in the
        # deploy dialog ("0" = CPU). req.target is the deploy flavor ("kserve")
        # and must NOT be used as the device (it made every deploy land on CPU).
        "gpu": (req.gpu if req.gpu and req.gpu != "0" else "cpu"),
        "is_name": is_name,
        "ns": ns,
        "domain": domain,
        "ingress_class": config.serving.ingress_class or "traefik",
        "source": source,
        "architecture": architecture,
    }

    # Persist the deploying state first so a crash between task insert and
    # state update can never strand the version in "ready" while a task runs.
    svc.model.mark_deploying(tenant, m["id"], version, is_name, runtime_type, domain, task_id)
    task = AsyncTask(
        id=task_id,
        tenant_id=tenant,
        type="model_deploy",
        status="pending",
        progress=0,
        params=task_params,
        created_by="",
    )
    svc.db.add(task)
    svc.db.commit()
    log.info("deploy task enqueued: task=%s service=%s ns=%s", task_id, is_name, ns)
    return {
        "task_id": task_id,
        "state": "deploying",
        "service": is_name,
        "namespace": ns,
        "runtime": runtime_type,
        "image": runtime_image,
        "domain": domain,
        "url": f"http://{domain}" if domain else "",
        "message": "deployment started",
    }


@router.post("/models/{model_name}/versions/{version}/undeploy")
async def undeploy_model_version(model_name: str, version: str, request: Request, svc: Services = Depends(get_svc)):
    """Undeploy a model version (submits KFP deletion, verifies removal async)."""
    from app.clients.k8s import serving_isvc_name
    from app.core.util import new_id
    from app.models.orm import AsyncTask

    tenant = request.headers.get("X-Tenant-Id", "default")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    is_name = serving_isvc_name(m["id"], version)
    ns = _tenant_namespace(tenant)
    # Block a duplicate submit while the previous undeploy is being verified.
    existing = svc.model.get_version(tenant, m["id"], version) or {}
    if str(((existing.get("metadata") or {}).get("deploy_status")) or "") == "terminating":
        raise error(409, "undeploy already in progress for this model version")
    # Cancel a still-running conversion/deploy task first so a late worker
    # stage cannot recreate the InferenceService we are about to delete.
    svc.model.cancel_deploy_task(tenant, m["id"], version)
    try:
        run_id = await asyncio.to_thread(svc.k8s.delete_inference_service, is_name, ns, False)
    except Exception as e:
        log.error("undeploy submit failed for %s: %s", is_name, e)
        raise error(500, f"undeploy failed: {e}")
    task_id = _enqueue_undeploy_verification(
        svc, AsyncTask, new_id, tenant, m["id"], version, is_name, ns, run_id,
    )
    log.info("undeploy submitted: service=%s ns=%s run=%s task=%s", is_name, ns, run_id, task_id)
    return {"undeployed": True, "service": is_name, "run_id": run_id, "task_id": task_id}


@router.get("/models/{model_name}/versions/{version}/serving")
async def serving_status(model_name: str, version: str, request: Request, svc: Services = Depends(get_svc)):
    """Return the serving status for a model version.

    States (version.status + metadata.deploy_status):
      deploying     -> {deployed:false, state:"deploying"} (UI shows spinner)
      deployed      -> {deployed:true, ready:<live KServe readiness>}
      deploy_failed -> {deployed:false, state:"failed", error}
      anything else -> {deployed:false}
    """
    from app.clients.k8s import serving_isvc_name, serving_reader

    tenant = request.headers.get("X-Tenant-Id", "default")
    m = svc.model.get_model(tenant, model_name)
    if not m:
        raise not_found()
    is_name = serving_isvc_name(m["id"], version)
    v = svc.model.get_version(tenant, m["id"], version)
    status = (v or {}).get("status") or ""
    meta = (v or {}).get("metadata") or {}
    domain = str(meta.get("domain", "") or "").strip()
    url = f"http://{domain}" if domain else ""
    base = {"service": is_name, "url": url, "domain": domain}

    if status == "deploying" or meta.get("deploy_status") == "deploying":
        return {**base, "deployed": False, "state": "deploying"}
    if meta.get("deploy_status") == "terminating":
        return {**base, "deployed": True, "state": "terminating"}
    if status == "deploy_failed" or meta.get("deploy_status") == "failed":
        return {
            **base, "deployed": False, "state": "failed",
            "error": str(meta.get("deploy_error", "") or ""),
        }
    # undeploy_failed: the service was never actually removed, so it stays
    # "deployed" with the verification error visible for a retry.
    if meta.get("deploy_status") == "undeploy_failed":
        return {
            **base, "deployed": True, "state": "undeploy_failed",
            "error": str(meta.get("deploy_error", "") or ""),
        }
    if status != "deployed":
        return {**base, "deployed": False}

    # Deployed: report live readiness so the UI can show rollout degradation.
    ns = _tenant_namespace(tenant)
    ready = True
    err = ""
    if serving_reader.available:
        live = await asyncio.to_thread(serving_reader.describe, is_name, ns)
        ready = bool(live.get("ready"))
        err = str(live.get("error", "") or "")
    return {**base, "deployed": True, "state": "ready", "ready": ready, "error": err}
