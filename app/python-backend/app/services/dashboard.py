# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Dashboard service: aggregate stats for the dashboard page.

Mirrors the Go DashboardService: counts across datasets/models/jobs/annotations/
ml_backends plus the 8 most recent jobs.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.util import iso
from app.models.orm import AnnotationProject, Dataset, Job, MLBackend, Model


def _stats_dict(
    datasets: int, models: int, training: int, optimization: int, evaluation: int,
    annotations: int, ml_backends: int, running: int, recent_jobs: list[dict],
) -> dict[str, Any]:
    """Assemble the dashboard stats response shape."""
    return {
        "datasets_count": datasets,
        "models_count": models,
        "training_jobs_count": training,
        "optimization_jobs_count": optimization,
        "evaluation_jobs_count": evaluation,
        "annotation_tasks_count": annotations,
        "ml_backends_count": ml_backends,
        "running_jobs_count": running,
        "recent_jobs": recent_jobs,
    }


class DashboardService:
    """Provides aggregate counts and recent jobs for the dashboard."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def stats(self, tenant_id: str) -> dict[str, Any]:
        """Return aggregate stats and recent jobs for a tenant."""
        datasets = self.db.scalar(
            select(func.count(Dataset.id)).where(Dataset.tenant_id == tenant_id)
        ) or 0
        models = self.db.scalar(
            select(func.count(Model.id)).where(Model.tenant_id == tenant_id)
        ) or 0
        training = self.db.scalar(
            select(func.count(Job.id)).where(Job.tenant_id == tenant_id, Job.type == "training")
        ) or 0
        optimization = self.db.scalar(
            select(func.count(Job.id)).where(Job.tenant_id == tenant_id, Job.type == "optimization")
        ) or 0
        evaluation = self.db.scalar(
            select(func.count(Job.id)).where(Job.tenant_id == tenant_id, Job.type == "evaluation")
        ) or 0
        annotations = self.db.scalar(
            select(func.count(AnnotationProject.id)).where(AnnotationProject.tenant_id == tenant_id)
        ) or 0
        ml_backends = self.db.scalar(
            select(func.count(MLBackend.id)).where(MLBackend.tenant_id == tenant_id)
        ) or 0
        running = self.db.scalar(
            select(func.count(Job.id)).where(
                Job.tenant_id == tenant_id, Job.status.in_(["pending", "running"])
            )
        ) or 0

        rows = self.db.scalars(
            select(Job)
            .where(Job.tenant_id == tenant_id)
            .order_by(Job.created_at.desc())
            .limit(8)
        ).all()
        recent = [
            {
                "id": j.id,
                "name": j.name,
                "type": j.type,
                "status": j.status,
                "created_at": iso(j.created_at),
            }
            for j in rows
        ]
        return _stats_dict(
            datasets, models, training, optimization, evaluation,
            annotations, ml_backends, running, recent,
        )
