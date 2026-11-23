# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""ORM models mirroring the 13 tables defined in the Go schema (internal/store/schema.go)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, Float, Index, Integer, String, Text,
    UniqueConstraint, text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Tenant administrator: NF user who owns this tenant (maps to the KF
    # Profile owner). The platform super-admin is NOT stored here — it is
    # the built-in user 'admin' in the 'default' tenant.
    admin_username: Mapped[str] = mapped_column(String, default="")
    kfp_userid: Mapped[str] = mapped_column(String, default="")
    kfp_experiment_id: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))


class Role(Base):
    __tablename__ = "roles"
    name: Mapped[str] = mapped_column(String, primary_key=True)
    description: Mapped[str] = mapped_column(Text, default="")
    permissions: Mapped[dict] = mapped_column(JSONB, default=list)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String, default="")
    full_name: Mapped[str] = mapped_column(String, default="")
    password_hash: Mapped[str] = mapped_column(String, default="")
    roles: Mapped[list] = mapped_column(JSONB, default=list)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    # Suite SSO linkage (standard section 5): the stable Identity subject and
    # the issuer it came from (mode oidc keys by issuer+sub). Empty for local
    # password users.
    external_id: Mapped[str] = mapped_column(String, default="")
    external_issuer: Mapped[str] = mapped_column(String, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    uri: Mapped[str] = mapped_column(String, default="")
    format: Mapped[str] = mapped_column(String, default="coco")
    version: Mapped[int] = mapped_column(Integer, default=1)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, default="active")
    created_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version"),)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String, nullable=False)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    uri: Mapped[str] = mapped_column(String, default="")
    format: Mapped[str] = mapped_column(String, default="coco")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    commit_hash: Mapped[str] = mapped_column(String, default="")
    created_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    parent_version_id: Mapped[str] = mapped_column(String, default="")
    source_project_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_annotated: Mapped[bool] = mapped_column(Boolean, default=False)
    is_draft: Mapped[bool] = mapped_column(Boolean, default=False)


class AnnotationProject(Base):
    __tablename__ = "annotation_projects"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    annotation_tool: Mapped[str] = mapped_column(String, default="label-studio")
    task_type: Mapped[str] = mapped_column(String, default="image-detection")
    status: Mapped[str] = mapped_column(String, default="pending")
    schema_: Mapped[dict] = mapped_column("schema", JSONB)
    import_uri: Mapped[str] = mapped_column(String, default="")
    dataset_id: Mapped[str] = mapped_column(String, default="")
    dataset_version: Mapped[int] = mapped_column(Integer, default=1)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    annotated_items: Mapped[int] = mapped_column(Integer, default=0)
    assignee_id: Mapped[str] = mapped_column(String, default="")
    external_id: Mapped[str] = mapped_column(String, default="")
    external_url: Mapped[str] = mapped_column(String, default="")
    # Per-project pre-annotation model preference: {"model_name": str,
    # "model_version": str} resolved to s3 weights at pre-annotation time.
    # Empty/null = no model chosen yet (pre-annotate is disabled in the UI).
    preannotate_model: Mapped[dict] = mapped_column(JSONB, nullable=True)
    created_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("idx_jobs_type", "type"),
        Index("idx_jobs_tenant", "tenant_id"),
        Index("idx_jobs_status", "status"),
        Index("idx_jobs_batch", "batch_id"),
    )
    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    batch_id: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="pending")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    params: Mapped[dict] = mapped_column(JSONB, default=dict)
    config: Mapped[dict] = mapped_column(JSONB, default=dict)
    # results/metrics are NOT NULL in the database; default to an empty JSON
    # object on insert so a freshly created job never violates the constraint.
    results: Mapped[dict] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    logs: Mapped[str] = mapped_column(Text, default="")
    kfp_run_id: Mapped[str] = mapped_column(String, default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    # Nullable: a pending job has not started and a non-terminal job has not finished.
    started_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)


class Model(Base):
    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("tenant_id", "name"),)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    framework: Mapped[str] = mapped_column(String, default="pytorch")
    task_type: Mapped[str] = mapped_column(String, default="classification")
    created_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_id", "version"),)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, nullable=False)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    file_path: Mapped[str] = mapped_column(String, default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    # "metadata" is reserved by SQLAlchemy Declarative, so the attribute is
    # named metadata_ but maps to the physical "metadata" column.
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    status: Mapped[str] = mapped_column(String, default="ready")
    artifacts: Mapped[list] = mapped_column(JSONB, default=list)
    created_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))


class AsyncTask(Base):
    __tablename__ = "async_tasks"
    __table_args__ = (Index("idx_async_tasks_status", "status"),)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="pending")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    params: Mapped[dict] = mapped_column(JSONB, default=dict)
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    # Lifecycle timestamps are written by the worker while a task runs (and
    # stay NULL for enqueued/pending tasks); the DB must allow NULL here.
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)


class MLBackend(Base):
    __tablename__ = "ml_backends"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, default="")
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String, default="unknown")
    health_check_interval: Mapped[int] = mapped_column(Integer, default=60)
    last_health_check: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    capabilities: Mapped[list] = mapped_column(JSONB, default=list)
    model_name: Mapped[str] = mapped_column(String, default="")
    model_version: Mapped[str] = mapped_column(String, default="")
    model_type: Mapped[str] = mapped_column(String, default="cnn")
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.6)
    annotations_schema: Mapped[dict] = mapped_column(JSONB, nullable=True, default=dict)
    created_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("idx_audit_tenant_time", "tenant_id", "created_at"),)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, default="")
    user_id: Mapped[str] = mapped_column(String, default="")
    action: Mapped[str] = mapped_column(String, nullable=False)
    resource_type: Mapped[str] = mapped_column(String, default="")
    resource_id: Mapped[str] = mapped_column(String, default="")
    details: Mapped[dict] = mapped_column(JSONB, default=dict)
    ip: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))


class SystemConfig(Base):
    __tablename__ = "system_config"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB, default=dict)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))


class ContainerImage(Base):
    """User-manageable image registry entry (train / prediction categories).

    Mirrors hub repos under nereidforge/train/* and nereidforge/prediction/*.
    build_context points at a builders/<dir> in the repo when NF owns the
    Dockerfile; empty for externally built images pushed straight to hub.
    """
    __tablename__ = "container_images"
    __table_args__ = (UniqueConstraint("tenant_id", "repository", "tag"),)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)  # train | prediction
    repository: Mapped[str] = mapped_column(String, nullable=False)  # registry root + repository, resolved via the registry profile
    tag: Mapped[str] = mapped_column(String, default="latest")
    description: Mapped[str] = mapped_column(Text, default="")
    build_context: Mapped[str] = mapped_column(String, default="")  # builders/<dir>, empty = external
    entrypoint_hint: Mapped[str] = mapped_column(String, default="")  # e.g. "python /app/train.py"
    task_types: Mapped[list] = mapped_column(JSONB, default=list)  # ["detection","classification","llm"...]
    # Capability announcement (nf.trainer-profile/1) as reported by the
    # component itself; task_types/default arch are derived from it (ADR-015).
    profile: Mapped[dict] = mapped_column(JSONB, default=dict)
    profile_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    profile_error: Mapped[str] = mapped_column(Text, default="")
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))


class BaseModel(Base):
    """User-uploaded entry in the S3 base-model pool.

    ``cv`` rows are single PyTorch checkpoints
    (``s3://nereidforge/base-models/cv/<file>.pt``), one row per file.
    ``llm`` rows are HF-style model directories
    (``s3://nereidforge/base-models/llm/<name>/``), one row per directory.
    """
    __tablename__ = "base_models"
    __table_args__ = (
        UniqueConstraint("tenant_id", "kind", "name"),
        Index("idx_base_models_kind", "kind"),
    )
    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)  # cv | llm
    name: Mapped[str] = mapped_column(String, nullable=False)  # file name (cv) / dir name (llm)
    uri: Mapped[str] = mapped_column(String, nullable=False)  # s3:// location
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    # Framework the artifact runs on (pytorch | tensorflow | onnx | ...).
    framework: Mapped[str] = mapped_column(String, default="pytorch")
    # CV task (detection | segmentation | classification); llm rows use llm/vlm.
    task: Mapped[str] = mapped_column(String, default="detection")
    # Lifecycle tag: original (upstream pretrained) | develop | production.
    status: Mapped[str] = mapped_column(String, default="develop")
    description: Mapped[str] = mapped_column(String, default="")
    # How the row entered the pool: manual | s3-reconcile | seed.
    source: Mapped[str] = mapped_column(String, default="manual")
    uploaded_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))


class KvObject(Base):
    """Postgres-backed short-TTL key/value store for the captcha answers
    and OIDC state/nonce/PKCE records. Namespace + key form the primary
    key; rows are consumed once (pop) and lazily expired."""

    __tablename__ = "kv_store"
    __table_args__ = (
        Index("idx_kv_store_expires_at", "expires_at"),
    )

    namespace: Mapped[str] = mapped_column(String(64), primary_key=True)
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()")
    )
