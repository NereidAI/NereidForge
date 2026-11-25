# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Service container: wires DB session and all services together.

Mirrors the Go service.NewServices dependency injection. A fresh container is
created per request (FastAPI dependency) so the SQLAlchemy session is scoped
correctly; long-lived clients (S3/LS/KFP) are singletons.
"""
from __future__ import annotations

from typing import Callable

from sqlalchemy.orm import Session

from app.clients.cvat import CvatClient
from app.clients.kfp import KFPClient
from app.clients.labelstudio import LabelStudioClient
from app.clients.s3 import SeaweedFSClient
from app.core.config import config
from app.core.images import nf_image
from app.db.session import SessionLocal
from app.services.admin import AdminService

# Shared singletons for stateless external clients.
_s3: SeaweedFSClient = SeaweedFSClient()
_ls: LabelStudioClient = LabelStudioClient()
_cvat: CvatClient = CvatClient()


def kfp_config_fn(db: Session) -> Callable[[], dict[str, str]]:
    """Return a closure reading KFP config from system_config on each call.

    Matches the Go kfpConfigFn: reads kfp_host/kfp_userid/kfp_experiment_id
    and the three pipeline ids from the platform system_config row.
    """
    def _read() -> dict[str, str]:
        admin = AdminService(db)
        cfg = admin.get_system_config()
        return {
            "kfp_host": str(cfg.get("kfp_host", "") or config.kfp.host),
            "kfp_token": str(cfg.get("kfp_token", "") or ""),
            "kfp_userid": str(cfg.get("kfp_userid", "") or "admin@nereidforge.local"),
            "kfp_experiment_id": str(cfg.get("kfp_experiment_id", "") or ""),
            "kfp_training_pipeline_id": str(cfg.get("kfp_training_pipeline_id", "") or ""),
            "kfp_optimization_pipeline_id": str(cfg.get("kfp_optimization_pipeline_id", "") or ""),
            "kfp_evaluation_pipeline_id": str(cfg.get("kfp_evaluation_pipeline_id", "") or ""),
            "trainer_image": str(cfg.get("trainer_image", "") or ""),
        }

    return _read


class Services:
    """Holds all service instances for a request session."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.admin = AdminService(db)
        self.s3 = _s3
        self.ls = _ls
        self.cvat = _cvat
        self.kfp = KFPClient(kfp_config_fn(db))
        # Dataset / Job services (wired eagerly since they are core).
        from app.clients.hub import hub_client
        from app.services.dataset import DatasetService
        from app.services.jobs import JobService
        from app.services.models import ModelService

        self.dataset = DatasetService(db, self.s3, self.kfp)

        self.model = ModelService(db, self.s3, hub=hub_client)
        self.job = JobService(
            db,
            self.s3,
            kfp=self.kfp,
            dataset=self.dataset,
            trainer_config_fn=self._trainer_config_fn(),
            models=self.model,
        )
        from app.services.images import ImageService
        from app.services.mlbackend import MLBackendService
        from app.services.project import ProjectService

        self.image = ImageService(db)
        self.ml_backend = MLBackendService(db)
        from app.clients.k8s import K8sService

        self.k8s = K8sService("nereidforge")
        self.project = ProjectService(
            db, self.s3, self.ls, cvat=self.cvat, dataset=self.dataset, ml_backend=self.ml_backend,
            s3_access_key=config.storage.s3_access_key,
            s3_secret_key=config.storage.s3_secret_key,
        )
        from app.services.dashboard import DashboardService

        self.dashboard = DashboardService(db)
        from app.services.base_models import BaseModelService

        self.base_model = BaseModelService(
            db, self.s3, bucket=config.storage.bucket or "nereidforge"
        )

    def _trainer_config_fn(self):
        """Return a closure reading trainer resource config from system_config.

        Matches the Go trainerCfgFn: accelerator/gpu/cpu/mem/image defaults.
        """
        admin = self.admin

        def _read() -> tuple[str, str, str, str, str, str, str]:
            cfg = admin.get_system_config()
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


def get_db():
    """FastAPI dependency: yield a scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_services(db: Session = None) -> Services:
    """Build a Services container (used by FastAPI dependencies)."""
    session = db or SessionLocal()
    return Services(session)
