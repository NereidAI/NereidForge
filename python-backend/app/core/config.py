# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Configuration loading.

Loads config.base.yaml deep-merged with config.{env}.yaml, plus NEREIDFORGE_*
env overrides, semantically aligned with the Go backend (internal/config/config.go):
- NEREIDFORGE_ENV=dev|prod selects config.{env}.yaml (default dev); values are
  merged over the shared config.base.yaml
- search paths: /etc/nereidforge -> . -> ./configs -> repo configs/
- NEREIDFORGE_ prefixed env vars override any key (dot -> underscore)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.core.images import nf_image, serving_image


def _dur(s: Any) -> int:
    """Convert a Go duration string ('12h','5m','30s') to seconds.

    Numeric input is returned as-is.
    """
    if s is None:
        return 0
    if isinstance(s, (int, float)):
        return int(s)
    m = re.fullmatch(r"(\d+)([smhd])", str(s).strip())
    if not m:
        return 0
    n = int(m.group(1))
    unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    return n * unit


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins). Returns a new dict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_yaml(env: str) -> dict[str, Any]:
    """Load config: config.base.yaml deep-merged with config.{env}.yaml.

    Shared values live in config.base.yaml; env-specific overrides live in
    config.{env}.yaml (e.g. config.dev.yaml, config.prod.yaml). Search order for
    each file: /etc/nereidforge, current dir, ./configs, then the repo's
    configs/ directory. The env file overrides the base file; NEREIDFORGE_*
    env vars override both (see _apply_env).
    """
    def _find(name: str) -> Path | None:
        for d in ["/etc/nereidforge", ".", "./configs"]:
            p = Path(d) / name
            if p.exists():
                return p
        # repo-local fallback: walk up from this file to the nearest configs/
        here = Path(os.path.abspath(__file__))
        for anc in here.parents:
            p = anc / "configs" / name
            if p.exists():
                return p
        return None

    merged: dict[str, Any] = {}
    base_path = _find("config.base.yaml")
    if base_path:
        with open(base_path) as f:
            merged = yaml.safe_load(f) or {}
    env_path = _find(f"config.{env}.yaml")
    if env_path:
        with open(env_path) as f:
            merged = _deep_merge(merged, yaml.safe_load(f) or {})
    return merged


def _env_value(key: str) -> Any:
    """Read a value from NEREIDFORGE_* env vars (viper dot->underscore semantics)."""
    env_key = "NEREIDFORGE_" + key.replace(".", "_").upper()
    if env_key in os.environ:
        return os.environ[env_key]
    return None


def _apply_env(data: dict[str, Any], prefix: str = "") -> None:
    """Recursively override config with env vars (NEREIDFORGE_AUTH_JWT_SECRET -> auth.jwt_secret)."""
    for k, v in data.items():
        if isinstance(v, dict):
            _apply_env(v, f"{prefix}{k}.")
        else:
            val = _env_value(f"{prefix}{k}")
            if val is not None:
                # coerce bool/int
                lv = val.lower()
                if lv in ("true", "false"):
                    data[k] = lv == "true"
                else:
                    try:
                        data[k] = int(val)
                    except ValueError:
                        data[k] = val


# These config paths hold Go duration strings in the yaml and must be converted to seconds
_DUR_KEYS = {
    "auth.access_ttl", "auth.refresh_ttl",
    "sso.oidc.state_ttl",
    "kfp.timeout", "hub.timeout", "trainer.timeout",
    "labelstudio.timeout", "cvat.timeout", "server.read_timeout", "server.write_timeout", "server.idle_timeout",
    "worker.poll_interval", "worker.kfp_completion_ttl", "worker.event_publish_retry",
    "event.webhook.timeout",
}


def _normalize_durations(data: dict[str, Any], prefix: str = "") -> None:
    for k, v in data.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            _normalize_durations(v, path + ".")
        elif path in _DUR_KEYS:
            data[k] = _dur(v)


def require_s3_credentials(access_key: str, secret_key: str, *, scope: str) -> tuple[str, str]:
    """Require a complete S3 access/secret pair before constructing its consumer.

    ``scope`` is the configuration prefix, such as ``storage.s3`` or
    ``kfp.object_store``. The returned pair is unchanged. Missing or whitespace-only
    values raise ``ValueError`` naming only configuration and environment keys, so
    operators get an actionable error without credential material entering logs.
    """
    fields = {
        f"{scope}_access_key": access_key,
        f"{scope}_secret_key": secret_key,
    }
    missing = [name for name, value in fields.items() if not str(value or "").strip()]
    if missing:
        env_names = [
            "NEREIDFORGE_" + name.replace(".", "_").upper()
            for name in missing
        ]
        raise ValueError(
            f"Missing required S3 credential configuration: {', '.join(missing)}; "
            f"set {', '.join(env_names)}"
        )
    return access_key, secret_key


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "nereidforge"
    password: str = ""
    name: str = "nereidforge"
    ssl_mode: str = "disable"

    @property
    def dsn(self) -> str:
        """Build the synchronous PostgreSQL DSN after requiring its password.

        A missing password is a deployment error, not an invitation to retry a
        shared tracked credential or create an engine with an empty secret.
        """
        if not self.password.strip():
            raise ValueError(
                "database.password is required; set NEREIDFORGE_DATABASE_PASSWORD"
            )
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}?sslmode={self.ssl_mode}"


class AuthConfig(BaseModel):
    enabled: bool = False
    jwt_secret: str = ""
    issuer: str = "nereidforge"
    access_ttl: int = 12 * 3600
    refresh_ttl: int = 7 * 24 * 3600


# Suite-role (NereidIdentity realm roles user/operator/admin) -> NF roles.
# The UI admin gate is the literal role "admin"; "user" is the baseline role.
_DEFAULT_ROLE_MAP: dict[str, list[str]] = {
    "admin": ["admin", "user"],
    "operator": ["admin", "user"],
    "user": ["user"],
}


class SSONsConfig(BaseModel):
    """Mode `ns` (standard section 4): ticket exchange against NereidSphere."""

    base_url: str = ""
    product_id: str = "forge"
    client_secret: str = ""
    callback_path: str = "/auth/ns-callback"
    # Reconcile the suite-managed tenant binding + roles on every NS login.
    # False freezes an already-provisioned user's tenant/roles (display fields
    # still refresh) — an escape hatch when NF must override what NS assigns.
    reconcile_memberships: bool = True


class SSOOidcConfig(BaseModel):
    """Mode `oidc` (standard section 6): direct confidential RP to NereidIdentity."""

    issuer: str = ""
    client_id: str = ""
    client_secret: str = ""
    redirect_url: str = ""
    frontend_base_url: str = ""
    scopes: list[str] = Field(default_factory=lambda: ["openid", "profile", "email", "roles"])
    roles_claim: str = "realm_access.roles"
    # Suite role -> NF roles; unknown suite roles fall back to default_roles
    # (or reject the login when reject_unknown_roles is set).
    role_map: dict[str, list[str]] = Field(default_factory=lambda: dict(_DEFAULT_ROLE_MAP))
    default_roles: list[str] = Field(default_factory=lambda: ["user"])
    reject_unknown_roles: bool = False
    # Lifetime of the state/nonce/PKCE record held between /sso/login and
    # /sso/callback (seconds; Go duration strings allowed in yaml).
    state_ttl: int = 600
    # RP-initiated logout: when true the frontend also hits the IdP
    # end_session_endpoint after clearing the local session.
    frontchannel_logout: bool = True


class SSOConfig(BaseModel):
    """Suite SSO (NereidIdentity docs/sso-integration-standard.md v2).

    mode: local (break-glass password login only) | ns | oidc.
    When mode is ns/oidc the auth.enabled=false dev bypass is force-disabled
    and identities enter only through the server-side onboarding path.
    """

    mode: str = "ns"
    ns: SSONsConfig = Field(default_factory=SSONsConfig)
    oidc: SSOOidcConfig = Field(default_factory=SSOOidcConfig)
    # Tenant attached to JIT-provisioned SSO users (identity plane defines no
    # tenant semantics; this is product-local configuration).
    jit_tenant: str = "default"

    @property
    def active(self) -> bool:
        """True when SSO gates authentication (mode ns or oidc)."""
        return self.mode in ("ns", "oidc")


class KFPConfig(BaseModel):
    host: str = ""
    namespace: str = "kubeflow"
    service_account: str = "pipeline-runner"
    timeout: int = 300
    poll_interval: int = 10
    # KFP stores run artifacts/logs in an S3 object store.
    # dev: point at a port-forwarded seaweedfs:9000; prod: http://seaweedfs.kubeflow:9000
    object_store_endpoint: str = ""
    object_store_access_key: str = ""
    object_store_secret_key: str = ""
    object_store_bucket: str = "mlpipeline"
    # Namespace under which KFP stores private-artifacts/{ns}/... logs.
    object_store_namespace: str = "kubeflow-user-example-com"


class HubConfig(BaseModel):
    host: str = ""
    timeout: int = 30


class TrainerConfig(BaseModel):
    host: str = ""
    timeout: int = 30


class LabelStudioConfig(BaseModel):
    host: str = ""
    token: str = ""
    public_host: str = ""
    ml_backend_url: str = ""
    ml_backend_cluster_url: str = ""
    timeout: int = 30


class CvatConfig(BaseModel):
    """CVAT integration config (ADR-010 D3)."""
    host: str = ""
    token: str = ""
    public_host: str = ""
    timeout: int = 30


class StorageConfig(BaseModel):
    backend: str = "seaweedfs"
    endpoint: str = ""
    s3_endpoint: str = ""
    s3_cluster_endpoint: str = ""
    filer_cluster_endpoint: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    bucket: str = "nereidforge"
    # Upload staging / limits. Files are buffered fully in the API pod before
    # being written to S3, so keep the per-request caps bounded.
    upload_max_files: int = 200
    upload_max_bytes: int = 1024 * 1024 * 1024


class ServingConfig(BaseModel):
    """Deployment (KServe InferenceService) runtime settings.

    These were previously hard-coded in the k8s client / deploy router.
    Customers can override them in env config files (or via NEREIDFORGE_SERVING_*
    env vars) without touching code.
    """
    # Base HTTP URL that predictor containers use to download model artifacts
    # from SeaweedFS S3 (must be reachable from inside the cluster).
    download_url_base: str = "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"
    # Runtime name -> image mapping. Keys match the "runtime_type" sent by the UI
    # / the ClusterServingRuntime names (openvino/triton/torch/huggingface).
    # openvino_xpu is the NF-built OVMS image carrying the Intel NEO compute
    # userspace (Arc/Battlemage); selected automatically for Intel GPU deploys.
    runtime_images: dict[str, str] = {
        "openvino": serving_image("openvino"),
        "openvino_xpu": nf_image("inference/model-server-openvino-xpu"),
        "triton": serving_image("triton"),
        "torch": serving_image("torch"),
        "huggingface": "docker.io/huggingface/text-generation-inference:latest",
    }
    # Triton config.pbtxt body written inside the serving container. Keep it
    # minimal so the onnxruntime backend auto-completes input/output shapes,
    # avoiding brittle shape hand-writing for each exported model.
    triton_config_pbtxt: str = (
        'name: "model"\n'
        'platform: "onnxruntime_onnx"\n'
        "max_batch_size: 0\n"
        'default_model_filename: "model.onnx"\n'
    )
    # Predictor HTTP port served for inference (must match KServe probes).
    serving_port: int = 8080
    # Traefik ingress routing the ISVC domain -> predictor svc.
    ingress_class: str = "traefik"
    # Default namespace for InferenceServices (may be overridden per tenant).
    namespace: str = "nereidforge"
    # Domain-safe suffix appended between the model name and the platform domain
    # (e.g. <model>-<ver>.inference.<platform_domain>).
    domain_suffix: str = "inference"
    # Root domain for inference endpoints (for example, example.com).
    platform_domain: str = ""
    # Trainer image used for ONNX conversion Jobs.
    trainer_image: str = nf_image("trainer/torch-cpu")


class WorkerConfig(BaseModel):
    enabled: bool = True
    poll_interval: int = 3
    concurrency: int = 5
    kfp_completion_ttl: int = 86400
    event_publish_retry: int = 5


class EventConfig(BaseModel):
    webhook_enabled: bool = False
    webhook_url: str = ""
    kafka_enabled: bool = False


class LoggingConfig(BaseModel):
    level: str = "debug"
    format: str = "json"


class TracingConfig(BaseModel):
    enabled: bool = False


class Config(BaseModel):
    env: str = "dev"
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    sso: SSOConfig = Field(default_factory=SSOConfig)
    kfp: KFPConfig = Field(default_factory=KFPConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    labelstudio: LabelStudioConfig = Field(default_factory=LabelStudioConfig)
    cvat: CvatConfig = Field(default_factory=CvatConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    event: EventConfig = Field(default_factory=EventConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)


def load_config(config_path: Path | None = None) -> Config:
    """Load the full Config: rendered /etc file, else repo yaml, then NEREIDFORGE_* env overrides.

    Pass config_path in tests to point at a fixture file.
    """
    import yaml
    from pathlib import Path

    # Rendered config file is the single source of truth; fall back to the
    # repo config.base.yaml + config.{env}.yaml so a bare host works too.
    if config_path is None:
        config_path = Path("/etc/nereidforge/config.yaml")
    env = os.environ.get("NEREIDFORGE_ENV", "dev")
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = _load_yaml(env)
    # Remove env from raw if present (already in rendered config)
    raw.pop("env", None)
    # NEREIDFORGE_* env vars override file values (auth.jwt_secret, auth.enabled, ...)
    _apply_env(raw)
    # Repository environment files use Go duration strings; rendered files may
    # already contain integer seconds, which the same normalizer preserves.
    _normalize_durations(raw)
    cfg = Config(env=env, **raw)
    # storage endpoint fallback: s3_endpoint wins when set (matches service.go semantics)
    if not cfg.storage.s3_endpoint and cfg.storage.endpoint:
        cfg.storage.s3_endpoint = cfg.storage.endpoint
    # Configuration is loaded before module-level S3/service clients exist, making
    # this the earliest common boundary that rejects an incomplete credential pair.
    require_s3_credentials(
        cfg.storage.s3_access_key,
        cfg.storage.s3_secret_key,
        scope="storage.s3",
    )
    return cfg


config: Config = load_config()
