# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""ML backend service: CRUD for Label Studio ML backend instances.

Mirrors the Go MLBackendService. Each ML backend points to a /predict service
(YOLO etc.) used for semi-auto annotation.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.core.util import iso, new_id, page_bounds
from app.models.orm import MLBackend

logger = logging.getLogger(__name__)


def _backend_dict(b: MLBackend) -> dict[str, Any]:
    """Serialize an MLBackend to the API response shape."""
    return {
        "id": b.id,
        "tenant_id": b.tenant_id,
        "name": b.name,
        "url": b.url,
        "description": b.description,
        "status": b.status,
        "health_check_interval": b.health_check_interval,
        "last_health_check": iso(b.last_health_check),
        "capabilities": b.capabilities or [],
        "model_name": b.model_name,
        "model_version": b.model_version,
        "model_type": b.model_type,
        "confidence_threshold": b.confidence_threshold,
        "annotations_schema": b.annotations_schema or {},
        "created_by": b.created_by,
        "created_at": iso(b.created_at),
        "updated_at": iso(b.updated_at),
    }


class MLBackendService:
    """Manages ML backend instances."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self, tenant_id: str, name: str, url: str, model_name: str, model_version: str,
        model_type: str, confidence_threshold: float, created_by: str,
    ) -> dict:
        """Create an ML backend."""
        b = MLBackend(
            id=new_id("mlb"),
            tenant_id=tenant_id,
            name=name,
            url=url or "",
            model_name=model_name or "",
            model_version=model_version or "",
            model_type=model_type or "cnn",
            confidence_threshold=confidence_threshold or 0.6,
            status="unknown",
            created_by=created_by or "anonymous",
        )
        self.db.add(b)
        self.db.commit()
        self.db.refresh(b)
        return _backend_dict(b)

    def list(
        self, tenant_id: str, page: int, page_size: int, status: str = "", search: str = "",
    ) -> tuple[list[dict], int]:
        """List ML backends with pagination, optionally filtered by status/name."""
        from sqlalchemy import func

        base = select(MLBackend).where(MLBackend.tenant_id == tenant_id)
        if status:
            base = base.where(MLBackend.status == status)
        if search:
            base = base.where(MLBackend.name.ilike(f"%{search.strip()}%"))
        total = self.db.scalar(
            select(func.count()).select_from(base.subquery())
        ) or 0
        limit, offset = page_bounds(page, page_size)
        rows = self.db.scalars(
            base.order_by(MLBackend.created_at.desc()).limit(limit).offset(offset)
        ).all()
        return [_backend_dict(b) for b in rows], total

    def get(self, tenant_id: str, id: str) -> Optional[dict]:
        """Get an ML backend by id."""
        b = self.db.scalar(
            select(MLBackend).where(and_(MLBackend.id == id, MLBackend.tenant_id == tenant_id))
        )
        return _backend_dict(b) if b else None

    def update(self, tenant_id: str, id: str, patch: dict[str, Any]) -> Optional[dict]:
        """Update ML backend fields."""
        b = self.db.scalar(
            select(MLBackend).where(and_(MLBackend.id == id, MLBackend.tenant_id == tenant_id))
        )
        if not b:
            return None
        for k, v in patch.items():
            if hasattr(b, k) and k not in ("id", "created_at"):
                setattr(b, k, v)
        self.db.commit()
        self.db.refresh(b)
        return _backend_dict(b)

    def delete(self, tenant_id: str, id: str) -> bool:
        """Delete an ML backend by id."""
        b = self.db.scalar(
            select(MLBackend).where(and_(MLBackend.id == id, MLBackend.tenant_id == tenant_id))
        )
        if not b:
            return False
        
        # Check if this is a system preset backend (cannot be deleted). Match
        # both the short svc name and the fully-qualified in-cluster domain.
        import re

        if b.url and re.search(r"ml-backend\.nereidforge(\.svc\.cluster\.local)?:9090", b.url):
            raise ValueError("Cannot delete system preset ML backend")
        
        self.db.delete(b)
        self.db.commit()
        return True

    def create_custom_backend(
        self,
        tenant_id: str,
        model_name: str,
        model_version: str,
        s3_path: str,
        backend_type: str,
        framework: str,
        created_by: str,
    ) -> dict:
        """
        Create a custom ML backend service.
        
        This creates K8s Deployment + Service and registers the backend in DB.
        
        Args:
            tenant_id: Tenant ID
            model_name: Model name
            model_version: Model version
            s3_path: S3 path to model (s3://bucket/key)
            backend_type: Backend type (det/seg/ocr/etc)
            framework: ML framework (ultralytics/pytorch/etc)
            created_by: User who created the backend
        
        Returns:
            Dict with backend info
        """
        from app.k8s.ml_backend_deployer import create_ml_backend_deployment
        
        # Create K8s Deployment + Service
        k8s_result = create_ml_backend_deployment(
            model_name=model_name,
            version=model_version,
            s3_path=s3_path,
            backend_type=backend_type,
            framework=framework,
        )
        
        service_name = k8s_result["service_name"]
        url = k8s_result["url"]
        
        # Register in DB
        b = MLBackend(
            id=new_id("mlb"),
            tenant_id=tenant_id,
            name=f"{model_name}-{model_version}",
            url=url,
            description=f"Custom ML backend for {model_name} v{model_version}",
            status="creating",
            model_name=model_name,
            model_version=model_version,
            model_type=backend_type,
            confidence_threshold=0.6,
            created_by=created_by,
            capabilities=[framework],
        )
        self.db.add(b)
        self.db.commit()
        self.db.refresh(b)
        
        logger.info(f"Created custom ML backend: {service_name} at {url}")
        return _backend_dict(b)

    def create_custom_backend_from_registry(
        self,
        tenant_id: str,
        model_name: str,
        model_version: str,
        created_by: str,
    ) -> dict:
        """
        Create a custom ML backend service from Model Registry.
        
        Fetches metadata from Model Registry (s3_path, backend_type, framework)
        and creates K8s Deployment + Service.
        
        Args:
            tenant_id: Tenant ID
            model_name: Model name (from Model Registry)
            model_version: Model version (from Model Registry)
            created_by: User who created the backend
        
        Returns:
            Dict with backend info
        """
        from app.clients.hub import hub_client
        
        # Get model metadata from Model Registry
        if not hub_client.enabled():
            raise ValueError("Model Registry is not available")
        
        model_info = hub_client.get_model(model_name)
        if not model_info:
            raise ValueError(f"Model not found in Model Registry: {model_name}")
        
        # Find the specific version
        version_info = None
        for v in model_info.get("versions", []):
            if v.get("name") == model_version or v.get("version") == model_version:
                version_info = v
                break
        
        if not version_info:
            raise ValueError(f"Model version not found: {model_name}:{model_version}")
        
        # Extract metadata
        metadata = version_info.get("metadata", {})
        s3_path = metadata.get("s3_path", version_info.get("uri", ""))
        backend_type = metadata.get("backend_type", "det")
        framework = metadata.get("framework", "ultralytics")
        
        if not s3_path:
            raise ValueError(f"S3 path not found for model: {model_name}:{model_version}")
        
        # Create custom backend
        return self.create_custom_backend(
            tenant_id=tenant_id,
            model_name=model_name,
            model_version=model_version,
            s3_path=s3_path,
            backend_type=backend_type,
            framework=framework,
            created_by=created_by,
        )

    def update_system_backend(
        self,
        tenant_id: str,
        id: str,
        patch: dict[str, Any],
    ) -> Optional[dict]:
        """
        Update system preset ML backend (restricted).
        
        System backends cannot change URL or model.
        """
        b = self.db.scalar(
            select(MLBackend).where(and_(MLBackend.id == id, MLBackend.tenant_id == tenant_id))
        )
        if not b:
            return None
        
        # Check if this is a system preset backend (short svc name or FQDN).
        import re

        is_system = bool(
            b.url and re.search(r"ml-backend\.nereidforge(\.svc\.cluster\.local)?:9090", b.url)
        )
        if is_system:
            # System backend: its URL/model are fixed, but operators may still
            # tune the confidence threshold and health metadata.
            allowed_fields = {"status", "last_health_check", "confidence_threshold"}
            restricted_patch = {k: v for k, v in patch.items() if k in allowed_fields}
            if not restricted_patch:
                logger.warning(f"Attempted to modify restricted fields on system backend: {id}")
                return _backend_dict(b)
            patch = restricted_patch
        
        for k, v in patch.items():
            if hasattr(b, k) and k not in ("id", "created_at"):
                setattr(b, k, v)
        self.db.commit()
        self.db.refresh(b)
        return _backend_dict(b)
