#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Backfill job.results.registered_model for completed training jobs.

Older worker versions auto-registered the trained model (PG + hub) but never
persisted the produced-model link on the job, so the training list "Model"
column was empty. The link is recoverable from the model version metadata
(metadata.job_id == job.id). Idempotent: jobs that already carry
results.registered_model are skipped.

Run inside the api pod:
    kubectl cp backfill_job_models.py <api-pod>:/tmp/
    kubectl exec ... env PYTHONPATH=/app python /tmp/backfill_job_models.py
"""

from app.db.session import SessionLocal
from app.models.orm import Job, Model, ModelVersion


def main() -> None:
    """Backfill produced-model links on completed training jobs."""
    db = SessionLocal()
    try:
        jobs = (
            db.query(Job)
            .filter(Job.type == "training", Job.status == "completed")
            .order_by(Job.created_at.desc())
            .all()
        )
        updated = 0
        skipped = 0
        missing = 0
        for job in jobs:
            # Copy to a NEW dict so SQLAlchemy sees the JSONB column as dirty;
            # mutating job.results in place and reassigning the same object
            # never triggers an UPDATE (commit silently no-ops).
            results = dict(job.results) if isinstance(job.results, dict) else {}
            if isinstance(results.get("registered_model"), dict) and results["registered_model"].get("id"):
                skipped += 1
                continue
            version = (
                db.query(ModelVersion)
                .filter(
                    ModelVersion.tenant_id == job.tenant_id,
                    ModelVersion.metadata_["job_id"].astext == job.id,
                )
                .order_by(ModelVersion.created_at.desc())
                .first()
            )
            if not version:
                missing += 1
                print(f"MISS {job.id}: no model version with metadata.job_id")
                continue
            model = db.query(Model).filter(Model.id == version.model_id).first()
            if not model:
                missing += 1
                print(f"MISS {job.id}: version {version.version} has no parent model")
                continue
            results["registered_model"] = {
                "id": str(model.id),
                "name": model.name,
                "version": version.version,
            }
            job.results = results
            print(f"LINK {job.id} -> {model.id} {model.name} {version.version}")
            updated += 1
        db.commit()
        print(f"done: updated={updated} skipped={skipped} missing={missing} total={len(jobs)}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
