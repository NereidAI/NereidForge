# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""System routes: tenants, users, roles, system config, audit, dashboard, ml-backends."""
from __future__ import annotations

from typing import Optional, Union

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from app.api.helpers import error, get_svc, not_found
from app.core.util import iso, paginate, page_bounds
from app.models.orm import AuditLog as AuditLogORM
from app.services import Services
from sqlalchemy import func, select

router = APIRouter()


class TenantCreate(BaseModel):
    """Create tenant request body."""
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = ""
    # NF user who administers this tenant (KF Profile owner)
    admin_username: Optional[str] = ""


class TenantUpdate(BaseModel):
    """Update tenant request body."""
    display_name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    admin_username: Optional[str] = None
    kfp_userid: Optional[str] = None
    kfp_experiment_id: Optional[str] = None


class UserCreate(BaseModel):
    """Create user request body."""
    username: str
    email: Optional[str] = ""
    full_name: Optional[str] = ""
    password: Optional[str] = ""
    tenant_id: Optional[str] = "default"
    roles: Optional[list[str]] = None


class UserUpdate(BaseModel):
    """Update user request body."""
    email: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    roles: Optional[list[str]] = None
    is_active: Optional[bool] = None


class RoleCreate(BaseModel):
    """Create role request body."""
    name: str
    description: Optional[str] = ""
    permissions: Optional[list[str]] = None


class RoleUpdate(BaseModel):
    """Update role request body."""
    description: Optional[str] = None
    permissions: Optional[list[str]] = None


class SystemConfigUpdate(BaseModel):
    """Update system config request body (merge into platform JSON)."""
    data: dict


class MLBackendCreate(BaseModel):
    """Create ML backend request body."""
    name: str
    url: Optional[str] = ""
    model_name: Optional[str] = ""
    model_version: Optional[str] = ""
    model_type: Optional[str] = "cnn"
    confidence_threshold: Optional[float] = 0.6


class MLBackendUpdate(BaseModel):
    """Update ML backend request body."""
    name: Optional[str] = None
    url: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_type: Optional[str] = None
    confidence_threshold: Optional[float] = None
    status: Optional[str] = None


class MLBackendCustomCreate(BaseModel):
    """Create custom ML backend request body."""
    model_name: str
    model_version: str
    s3_path: str  # s3://bucket/key
    backend_type: str  # det/seg/ocr/etc
    framework: str  # paddlex/pytorch/tensorflow/onnxruntime/paddleocr


class MLBackendCustomFromRegistry(BaseModel):
    """Create custom ML backend from Model Registry."""
    model_name: str
    model_version: str


# -- dashboard --------------------------------------------------------------


@router.get("/dashboard/stats")
async def dashboard_stats(request: Request, svc: Services = Depends(get_svc)):
    """Return dashboard aggregate stats."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    return svc.dashboard.stats(tenant)


@router.get("/hardware-capabilities")
async def hardware_capabilities(request: Request):
    """Return which optimization targets can actually run on this cluster.

    Semantics per target (mirrors what the optimizer image can execute):
      - cpu    : ONNX Runtime graph/INT8 passes -- always available.
      - intel  : OpenVINO IR conversion (ovc) + CPU benchmark/serving -- a
                 CPU-only toolchain, so always available regardless of Intel
                 GPUs. (Intel XPU training is a separate, plugin-gated path.)
      - nvidia : TensorRT engine builds require trtexec ON an NVIDIA GPU node;
                 reported from node allocatable (nvidia.com/gpu > 0).
      - amd    : ROCm execution requires an AMD GPU node (amd.com/gpu > 0).
      - huawei : Ascend toolchain image is not shipped yet; always unavailable.
    """
    from app.clients.gpu_capacity import gpu_capacity

    vendors = gpu_capacity().get("vendors") or {}

    def _gpu_entry(vendor: str, label: str) -> dict:
        info = vendors.get(vendor)
        if not info:
            return {"available": False, "detail": f"{label} not advertised by any Ready node"}
        return {
            "available": True,
            "detail": f"{label}: {info['total']} device(s) on {len(info['nodes'])} node(s)",
        }

    return {
        "cpu": {"available": True, "detail": "ONNX Runtime (CPU)"},
        "intel": {"available": True, "detail": "OpenVINO IR conversion + CPU runtime"},
        "nvidia": _gpu_entry("nvidia", "NVIDIA GPU"),
        "amd": _gpu_entry("amd", "AMD GPU (ROCm)"),
        "huawei": {"available": False, "detail": "Ascend optimization toolchain not shipped"},
        "resources": vendors,
    }


@router.get("/gpu-capacity")
async def get_gpu_capacity():
    """Return per-vendor GPU/vGPU capacity from node allocatable resources.

    Counts include shared devices advertised by device plugins
    (shared-dev-num), so the training form can bound GPU count by real
    cluster capacity instead of a hardcoded value. An empty result means
    node reads are unavailable and the UI leaves the field unbounded.
    """
    from app.clients.gpu_capacity import gpu_capacity

    return gpu_capacity()


@router.get("/nodes")
async def list_nodes(request: Request):
    """Return the cluster node names.

    Deprecated: node discovery required cluster reads; the NereidForge
    container no longer performs them. Jobs run on the cluster scheduler.
    """
    return {"items": [], "note": "node pinning is managed by cluster ops / the Kubeflow UI"}


# -- tenants ----------------------------------------------------------------


@router.get("/tenants")
async def list_tenants(request: Request, search: str = "", page: int = 1, page_size: int = 20, svc: Services = Depends(get_svc)):
    """List tenants with search."""
    items, total = svc.admin.list_tenants(page, page_size, search)
    return paginate(items, total, page, page_size)


@router.post("/tenants")
async def create_tenant(req: TenantCreate, svc: Services = Depends(get_svc)):
    """Create a tenant (provisions the nereidforge-<name> KF Profile/ns)."""
    return svc.admin.create_tenant(
        req.name, req.display_name or req.name, req.description or "",
        admin_username=req.admin_username or "",
    )


@router.get("/tenants/{id}")
async def get_tenant(id: str, svc: Services = Depends(get_svc)):
    """Get a tenant."""
    t = svc.admin.get_tenant(id)
    if not t:
        raise not_found()
    return t


@router.patch("/tenants/{id}")
async def update_tenant(id: str, req: TenantUpdate, svc: Services = Depends(get_svc)):
    """Update a tenant."""
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    t = svc.admin.update_tenant(id, patch)
    if not t:
        raise not_found()
    return t


@router.delete("/tenants/{id}")
async def delete_tenant(id: str, request: Request, svc: Services = Depends(get_svc)):
    """Delete a tenant (namespace kept unless ?namespace=delete)."""
    drop_ns = request.query_params.get("namespace", "keep") == "delete"
    try:
        if not svc.admin.delete_tenant(id, keep_namespace=not drop_ns):
            raise not_found()
    except ValueError as e:
        raise error(400, str(e))
    return {"deleted": True}


# -- users ------------------------------------------------------------------


@router.get("/settings/users")
async def list_users(request: Request, search: str = "", page: int = 1, page_size: int = 20, svc: Services = Depends(get_svc)):
    """List users with search."""
    items, total = svc.admin.list_users(page, page_size, search)
    return paginate(items, total, page, page_size)


@router.post("/settings/users")
async def create_user(req: UserCreate, svc: Services = Depends(get_svc)):
    """Create a user."""
    return svc.admin.create_user(
        req.username, req.email or "", req.full_name or "", req.password or "",
        req.tenant_id or "default", req.roles or [],
    )


@router.patch("/settings/users/{id}")
async def update_user(id: str, req: UserUpdate, svc: Services = Depends(get_svc)):
    """Update a user."""
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    u = svc.admin.update_user(id, patch)
    if not u:
        raise not_found()
    return u


@router.delete("/settings/users/{id}")
async def delete_user(id: str, svc: Services = Depends(get_svc)):
    """Delete a user."""
    if not svc.admin.delete_user(id):
        raise not_found()
    return {"deleted": True}


# -- roles ------------------------------------------------------------------


@router.get("/settings/roles")
async def list_roles(svc: Services = Depends(get_svc)):
    """List roles."""
    return {"items": svc.admin.list_roles()}


@router.post("/settings/roles")
async def create_role(req: RoleCreate, svc: Services = Depends(get_svc)):
    """Create a role."""
    return svc.admin.create_role(req.name, req.description or "", req.permissions or [])


@router.patch("/settings/roles/{name}")
async def update_role(name: str, req: RoleUpdate, svc: Services = Depends(get_svc)):
    """Update a role."""
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    r = svc.admin.update_role(name, patch)
    if not r:
        raise not_found()
    return r


@router.delete("/settings/roles/{name}")
async def delete_role(name: str, svc: Services = Depends(get_svc)):
    """Delete a role."""
    if not svc.admin.delete_role(name):
        raise not_found()
    return {"deleted": True}


# -- system config ----------------------------------------------------------


@router.get("/system/config")
async def get_system_config(svc: Services = Depends(get_svc)):
    """Get the platform system config."""
    return svc.admin.get_system_config()


@router.put("/system/config")
async def update_system_config(req: SystemConfigUpdate, svc: Services = Depends(get_svc)):
    """Update the platform system config (merge)."""
    return svc.admin.update_system_config(req.data)


@router.get("/system/preflight")
async def system_preflight(svc: Services = Depends(get_svc)):
    """Read-only data-plane dependency checks (k8s/S3/KFP/images/backends).

    Surfaces configuration drift (missing serving namespace, unreachable
    bucket/KFP, empty ML backends table, missing runtime images) before it
    turns into a cryptic deploy-time failure.
    """
    import asyncio

    from app.services.preflight import run_preflight

    return await asyncio.to_thread(run_preflight, svc.db)


# -- audit ------------------------------------------------------------------


@router.get("/system/audit-logs")
async def list_audit_logs(
    request: Request, action: str = "", search: str = "",
    page: int = 1, page_size: int = 20, svc: Services = Depends(get_svc),
):
    """List audit logs with an action filter, name/id search, and pagination."""
    from sqlalchemy import or_ as sql_or
    q = select(AuditLogORM)
    if action:
        q = q.where(AuditLogORM.action == action)
    if search:
        needle = f"%{search.strip()}%"
        q = q.where(sql_or(
            AuditLogORM.user_id.ilike(needle),
            AuditLogORM.resource_id.ilike(needle),
        ))
    total = svc.db.scalar(select(func.count()).select_from(q.subquery())) or 0
    limit, offset = page_bounds(page, page_size)
    rows = svc.db.scalars(
        q.order_by(AuditLogORM.created_at.desc()).limit(limit).offset(offset)
    ).all()
    items = [{
        "id": r.id, "tenant_id": r.tenant_id, "user_id": r.user_id,
        "action": r.action, "resource_type": r.resource_type,
        "resource_id": r.resource_id, "details": r.details or {},
        "ip": r.ip, "created_at": iso(r.created_at),
    } for r in rows]
    return paginate(items, total, page, page_size)


@router.get("/system/compute-resources")
async def list_compute_resources(svc: Services = Depends(get_svc)):
    """List deployable compute devices (CPU + live cluster GPU capacity).

    Drives the deploy dialog device selector. GPU counts come from node
    allocatable extended resources (gpu.intel.com/xe, nvidia.com/gpu,
    amd.com/gpu) via the shared gpu_capacity module; when the cluster API
    is unreachable the selector still offers CPU.
    """
    import asyncio

    from app.clients.gpu_capacity import gpu_capacity

    def _collect() -> list[dict]:
        vendors = (gpu_capacity() or {}).get("vendors") or {}
        profiles = {
            "intel": "Intel XPU (Arc / Battlemage)",
            "nvidia": "NVIDIA CUDA",
            "amd": "AMD ROCm",
        }
        out = []
        for vendor, name in profiles.items():
            info = vendors.get(vendor)
            if not info:
                continue
            nodes = [
                {"node": n["name"], "count": n["devices"], "gpu_model": ""}
                for n in info.get("nodes", [])
            ]
            out.append({
                "device": vendor, "unit": info["resource"], "name": name,
                "count": info["total"], "nodes": nodes,
            })
        return out

    resources = [{"device": "cpu", "unit": "", "name": "CPU", "count": 1}]
    try:
        resources.extend(await asyncio.to_thread(_collect))
    except Exception:  # noqa: BLE001 - endpoint must answer offline too
        pass
    return {"resources": resources}


# -- container images (train / prediction) ---------------------------------


class SamSegmentRequest(BaseModel):
    """Interactive SAM segmentation request (click-to-segment)."""
    image_url: str = ""
    image_base64: str = ""
    points: list[list[float]] = []
    negative_points: list[list[float]] = []
    box: Optional[list[float]] = None
    ml_backend_id: str = ""
    # User-selected SAM model from the registry; resolved to an s3 weights URI.
    model_name: str = ""
    model_version: str = ""


@router.get("/base-models")
async def list_base_models(request: Request, svc: Services = Depends(get_svc)):
    """List S3 base-model pool entries (folders with a config.json)."""
    return {"items": svc.image.list_base_models(), "total": 0}


@router.post("/sam/segment")
async def sam_segment(req: SamSegmentRequest, request: Request, svc: Services = Depends(get_svc)):
    """Proxy interactive SAM segmentation to the shared ML backend.

    Accepts an http(s) image URL the backend can reach (S3 gateway links),
    or base64 JPEG data (used by editors holding a frame locally).
    Returns {"polygon": [[x,y],...], "bbox": [x1,y1,x2,y2]}.
    """
    tenant = request.headers.get("X-Tenant-Id", "default")
    if not req.image_url and not req.image_base64:
        raise error(400, "image_url or image_base64 required")
    if not req.points and not req.negative_points and not req.box:
        raise error(400, "points, negative_points or box required")
    if not req.model_name or not req.model_version:
        raise error(
            400,
            "no SAM model selected: upload a SAM (interactive segmentation) model "
            "and select it in the editor",
        )
    try:
        return svc.project.sam_segment(
            tenant, req.image_url, req.image_base64,
            req.points, req.negative_points, req.box, req.ml_backend_id,
            req.model_name, req.model_version,
        )
    except FileNotFoundError as e:
        raise error(400, str(e))
    except ValueError as e:
        raise error(502, str(e))


class ImageCreate(BaseModel):
    """Register a user image in the catalog."""
    name: str
    category: str  # train | prediction
    repository: str  # full image repository (registry root follows the active profile)
    tag: str = "latest"
    description: str = ""
    build_context: str = ""  # builders/<dir>; empty = externally built
    entrypoint_hint: str = ""
    # Capability announcement (`nf.trainer-profile/1`). Optional: the
    # registration endpoint prefers the component's own announcement when the
    # build context is a directory in this repository (ADR-015).
    task_types: list[str] = []
    profile: Optional[dict] = None


class ImageUpdate(BaseModel):
    """Patch an image entry."""
    name: Optional[str] = None
    category: Optional[str] = None
    repository: Optional[str] = None
    tag: Optional[str] = None
    description: Optional[str] = None
    build_context: Optional[str] = None
    entrypoint_hint: Optional[str] = None
    task_types: Optional[list[str]] = None


class ImageProfile(BaseModel):
    """Capability announcement supplied by the registrant.

    Accepts the document itself or a JSON string (a pasted
    `trainer-profile.json`); sending `null` clears it.
    """
    profile: Optional[Union[dict, str]] = None


class AnnouncementNode(BaseModel):
    """AnnouncementNode preserves component-specific metadata in a lockable node."""
    model_config = ConfigDict(extra="allow")

    id: str
    label: str = ""
    tier: str = "core"
    requires: list[str] = Field(default_factory=list)
    locked: bool = False
    missing: list[str] = Field(default_factory=list)


class AnnouncedArchitecture(AnnouncementNode):
    """AnnouncedArchitecture describes one selectable trainer architecture."""


class AnnouncedTask(AnnouncementNode):
    """AnnouncedTask describes a trainer task and its architecture catalog."""
    default_arch: str = ""
    architectures: list[AnnouncedArchitecture] = Field(default_factory=list)


class AnnouncedCapability(AnnouncementNode):
    """AnnouncedCapability describes an optional component capability."""


class CapabilityImage(BaseModel):
    """CapabilityImage describes one train catalog row for the training form."""
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    repository: str
    tag: str
    full_ref: str
    is_builtin: bool
    task_types: list[str]
    devices: list[str]
    announced: bool
    profile_error: str = ""
    tasks: list[AnnouncedTask]
    capabilities: list[AnnouncedCapability]


class CapabilitiesResponse(BaseModel):
    """CapabilitiesResponse is the trainer capability and entitlement snapshot."""
    entitlements: list[str]
    images: list[CapabilityImage]


@router.get("/images")
async def list_images(
    request: Request, category: str = "", task_type: str = "", search: str = "",
    page: int = 1, page_size: int = 20,
    svc: Services = Depends(get_svc),
):
    """List catalog images with server-side pagination (category/task_type/search)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    svc.image.seed_builtins(tenant)
    items, total = svc.image.list_paged(tenant, category, task_type, search, page, page_size)
    return paginate(items, total, page, page_size)


@router.post("/images")
async def create_image(req: ImageCreate, request: Request, svc: Services = Depends(get_svc)):
    """Register an image (auto-seeds builtins on first call)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    try:
        return svc.image.create(tenant, req.model_dump(), user)
    except ValueError as e:
        raise error(400, str(e))


@router.patch("/images/{image_id}")
async def update_image(image_id: str, req: ImageUpdate, request: Request, svc: Services = Depends(get_svc)):
    """Update an image entry."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    img = svc.image.update(tenant, image_id, patch)
    if not img:
        raise not_found()
    return img


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def list_capabilities(
    request: Request, svc: Services = Depends(get_svc),
) -> CapabilitiesResponse:
    """Announced trainer capabilities of this deployment, with lock states.

    Aggregates every registered train image's `nf.trainer-profile/1`
    announcement (tasks, architectures, defaults, devices) and marks extended
    nodes locked/unlocked against the deployment's entitlements (ADR-015).
    The train dialog renders from this instead of a hardcoded list.
    """
    tenant = request.headers.get("X-Tenant-Id", "default")
    return svc.image.capabilities(tenant)


@router.post("/images/{image_id}/profile")
async def set_image_profile(
    image_id: str, req: ImageProfile, request: Request, svc: Services = Depends(get_svc),
):
    """Store the capability announcement of an image (paste/upload the document).

    The platform registers what a component announces; it never invents the
    task list (ADR-015). Built-ins normally announce from the repository —
    `POST .../profile/refresh` re-reads that file — but this endpoint is the
    path for user-registered images and for corrections.
    """
    tenant = request.headers.get("X-Tenant-Id", "default")
    img = svc.image.set_profile(tenant, image_id, req.profile)
    if not img:
        raise not_found()
    return img


@router.post("/images/{image_id}/profile/refresh")
async def refresh_image_profile(
    image_id: str, request: Request, svc: Services = Depends(get_svc),
):
    """Re-read a built-in component's announcement from its build context."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    try:
        img = svc.image.refresh_profile(tenant, image_id)
    except ValueError as e:
        raise error(400, str(e))
    if not img:
        raise not_found()
    return img


@router.delete("/images/{image_id}")
async def delete_image(image_id: str, request: Request, svc: Services = Depends(get_svc)):
    """Delete an image entry (built-ins refuse)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    try:
        if not svc.image.delete(tenant, image_id):
            raise not_found()
    except ValueError as e:
        raise error(400, str(e))
    return {"deleted": True}


@router.post("/system/registry/test")
async def test_container_registry(request: Request, body: dict = None, svc: Services = Depends(get_svc)):
    """Test connectivity + auth against the container registry (Harbor v2 API).

    Accepts overrides (host/project/username/password) so the Settings form
    can test BEFORE saving; falls back to the stored system_config values.
    Never echoes the password back.
    """
    import httpx

    cfg = svc.admin.get_system_config()
    body = body or {}
    host = str(body.get("host") or cfg.get("container_registry_host") or "").strip()
    project = str(body.get("project") or cfg.get("container_registry_project") or "nereidai").strip()
    username = str(body.get("username") or cfg.get("container_registry_username") or "").strip()
    password = str(body.get("password") or cfg.get("container_registry_password") or "").strip()
    insecure = bool(body.get("insecure", cfg.get("container_registry_insecure", False)))
    if not host:
        raise error(400, "registry host is required")

    auth = (username, password) if (username and password) else None
    # insecure = skip TLS certificate verification (self-signed / private
    # CA). The scheme is always https first; plain http only when the user
    # explicitly wrote a scheme into the host.
    if "://" in host:
        base = host.split("://", 1)[1].rstrip("/")
    else:
        base = host.rstrip("/")
    verify = not insecure

    result: dict = {"host": host, "project": project}
    try:
        async def probe(url: str):
            return await _hclient_get(url, auth, verify)

        # 1. reachability: Harbor API root (https; 308 upgrade handled by
        # the redirect-following client)
        r = await probe(f"https://{base}/api/v2.0/health")
        if r.status_code >= 500 or r.status_code == 404:
            # try plain http as a last resort (lab registries without TLS)
            try:
                alt = await probe(f"http://{base}/api/v2.0/health")
                if alt.status_code < r.status_code:
                    r = alt
                    base = f"http://{base}"
                else:
                    base = f"https://{base}"
            except Exception:
                base = f"https://{base}"
        else:
            base = f"https://{base}"
        result["reachable"] = r.status_code < 500
        # 2. auth + project: query the project itself
        r2 = await probe(f"{base}/api/v2.0/projects/{project}")
        result["status_code"] = r2.status_code
        if r2.status_code == 200:
            info = r2.json()
            result["ok"] = True
            result["project_exists"] = True
            result["repo_count"] = info.get("repo_count")
            result["message"] = "connected"
        elif r2.status_code == 401:
            result["ok"] = False
            result["message"] = "authentication failed (check username/password)"
        elif r2.status_code == 404:
            result["ok"] = False
            result["project_exists"] = False
            result["message"] = f"registry reachable but project '{project}' not found"
        else:
            result["ok"] = False
            result["message"] = f"unexpected status {r2.status_code}"
    except httpx.ConnectError as e:
        result["ok"] = False
        result["reachable"] = False
        result["message"] = f"cannot connect: {str(e)[:120]}"
    except Exception as e:
        result["ok"] = False
        result["message"] = f"error: {str(e)[:120]}"
    return result


async def _hclient_get(url: str, auth, verify: bool):
    """GET following same-/cross-scheme redirects, short timeout, no raise."""
    import httpx

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(8.0),
        verify=verify,
        follow_redirects=True,
        # cross-scheme redirect targets must also skip TLS verify in insecure mode
        trust_env=False,
    ) as client:
        return await client.get(url, auth=auth)


# -- ml backends ------------------------------------------------------------


@router.get("/ml-backends")
async def list_ml_backends(request: Request, status: str = "", search: str = "", page: int = 1, page_size: int = 20, svc: Services = Depends(get_svc)):
    """List ML backends with filters/search."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    items, total = svc.ml_backend.list(tenant, page, page_size, status, search)
    return paginate(items, total, page, page_size)


@router.post("/ml-backends")
async def create_ml_backend(req: MLBackendCreate, request: Request, svc: Services = Depends(get_svc)):
    """Create an ML backend."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    return svc.ml_backend.create(
        tenant, req.name, req.url or "", req.model_name or "", req.model_version or "",
        req.model_type or "cnn", req.confidence_threshold or 0.6, user,
    )


@router.get("/ml-backends/{id}")
async def get_ml_backend(id: str, request: Request, svc: Services = Depends(get_svc)):
    """Get an ML backend."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    b = svc.ml_backend.get(tenant, id)
    if not b:
        raise not_found()
    return b


@router.patch("/ml-backends/{id}")
async def update_ml_backend(id: str, req: MLBackendUpdate, request: Request, svc: Services = Depends(get_svc)):
    """Update an ML backend.

    Routes through the system-aware updater so the shared built-in backend
    keeps a fixed URL/model while still allowing confidence/health edits.
    """
    tenant = request.headers.get("X-Tenant-Id", "default")
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    b = svc.ml_backend.update_system_backend(tenant, id, patch)
    if not b:
        raise not_found()
    return b


@router.delete("/ml-backends/{id}")
async def delete_ml_backend(id: str, request: Request, svc: Services = Depends(get_svc)):
    """Delete an ML backend."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    try:
        if not svc.ml_backend.delete(tenant, id):
            raise not_found()
    except ValueError as e:
        return error(str(e), status_code=400)
    return {"deleted": True}


@router.post("/ml-backends/custom")
async def create_custom_ml_backend(
    req: MLBackendCustomCreate,
    request: Request,
    svc: Services = Depends(get_svc),
):
    """Create a custom ML backend service (K8s Deployment + Service)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    
    try:
        return svc.ml_backend.create_custom_backend(
            tenant_id=tenant,
            model_name=req.model_name,
            model_version=req.model_version,
            s3_path=req.s3_path,
            backend_type=req.backend_type,
            framework=req.framework,
            created_by=user,
        )
    except Exception as e:
        return error(f"Failed to create custom ML backend: {str(e)}", status_code=500)


@router.post("/ml-backends/custom/from-registry")
async def create_custom_ml_backend_from_registry(
    req: MLBackendCustomFromRegistry,
    request: Request,
    svc: Services = Depends(get_svc),
):
    """Create a custom ML backend service from Model Registry."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    user = request.headers.get("X-Username", "anonymous")
    
    try:
        return svc.ml_backend.create_custom_backend_from_registry(
            tenant_id=tenant,
            model_name=req.model_name,
            model_version=req.model_version,
            created_by=user,
        )
    except Exception as e:
        return error(f"Failed to create custom ML backend from registry: {str(e)}", status_code=500)


@router.get("/ml-backends/{id}/status")
async def get_ml_backend_status(id: str, request: Request, svc: Services = Depends(get_svc)):
    """Get ML backend status (DB record; live health via POST /health)."""
    tenant = request.headers.get("X-Tenant-Id", "default")
    b = svc.ml_backend.get(tenant, id)
    if not b:
        raise not_found()
    from app.k8s.ml_backend_deployer import generate_service_name

    service_name = generate_service_name(b.get("model_name", ""), b.get("model_version", ""))
    return {
        "service_name": service_name,
        "url": b.get("url", ""),
        "status": b.get("status", "unknown"),
        "k8s": "unavailable from NereidForge",
    }


@router.post("/ml-backends/{id}/health")
async def check_ml_backend_health(id: str, request: Request, svc: Services = Depends(get_svc)):
    """Live health-check an ML backend by probing its /health endpoint.

    Returns the same shape the frontend expects: {status, latency_ms}.
    """
    import time
    import requests as http_requests

    tenant = request.headers.get("X-Tenant-Id", "default")
    b = svc.ml_backend.get(tenant, id)
    if not b:
        raise not_found()

    url = (b.get("url") or "").rstrip("/")
    if not url:
        return {"status": "unreachable", "latency_ms": 0, "detail": "no url configured"}

    start = time.perf_counter()
    try:
        r = http_requests.get(f"{url}/health", timeout=5)
        latency = round((time.perf_counter() - start) * 1000)
        data = r.json()
        status = "healthy" if r.status_code == 200 and data.get("status") == "UP" else "unhealthy"
        return {"status": status, "latency_ms": latency}
    except Exception:
        latency = round((time.perf_counter() - start) * 1000)
        return {"status": "unhealthy", "latency_ms": latency, "detail": "probe failed"}
