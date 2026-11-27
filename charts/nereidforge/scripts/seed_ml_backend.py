# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Idempotently seed the system-preset ML backend registration.

The ML Backend UI page reads the Postgres ``ml_backends`` table; a running
ml-backend Deployment never creates that row on its own (recurrence of
pitfalls 171/315: every fresh database starts with an empty table and the
page stays blank). This post-install hook guarantees the ``default`` tenant
owns the protected system preset pointing at the in-cluster ml-backend
Service. The worker health loop owns the ``status`` column afterwards, so
the row is created with ``unknown`` and flips to ``connected`` on the next
poll without any action from this script.

The preset URL defaults to ``labelstudio.ml_backend_url`` from the rendered
``/etc/nereidforge/config.yaml`` (the same single source the annotation
flows use) and may be overridden with ``NF_MLBACKEND_URL`` for split
topologies.
"""

import os

from sqlalchemy import select

from app.core.config import config
from app.db.session import SessionLocal
from app.models.orm import MLBackend
from app.services.mlbackend import MLBackendService

# The system preset belongs to the break-glass super-admin tenant; tenant
# namespaces manage their own custom backends through the UI.
SYSTEM_TENANT = "default"


def main() -> None:
    """Create the preset row once; reruns on upgrade must stay no-op.

    Raises:
        RuntimeError: when neither ``NF_MLBACKEND_URL`` nor the rendered
            config provides an ml-backend URL, because seeding an empty URL
            would produce a preset the worker can never health-check.
    """
    url = (os.environ.get("NF_MLBACKEND_URL") or config.labelstudio.ml_backend_url).strip()
    if not url:
        raise RuntimeError(
            "ml backend url not configured: set labelstudio.ml_backend_url in "
            "config.yaml or NF_MLBACKEND_URL in the hook environment"
        )

    name = os.environ.get("NF_MLBACKEND_NAME", "system-ml-backend")
    model_name = os.environ.get("NF_MLBACKEND_MODEL_NAME", "PP-YOLOE")
    model_version = os.environ.get("NF_MLBACKEND_MODEL_VERSION", "v1")
    model_type = os.environ.get("NF_MLBACKEND_MODEL_TYPE", "cnn")
    confidence_threshold = float(os.environ.get("NF_MLBACKEND_CONFIDENCE", "0.6"))
    created_by = os.environ.get("NF_MLBACKEND_CREATED_BY", "system-seed")

    db = SessionLocal()
    try:
        # Match on (tenant, url), not name: operators may rename the preset in
        # the UI, but the in-cluster URL is the identity the regex guards use.
        existing = db.scalar(
            select(MLBackend).where(
                MLBackend.tenant_id == SYSTEM_TENANT,
                MLBackend.url == url,
            )
        )
        if existing:
            print(f"system ml backend already registered: {existing.id} -> {url}")
            return

        service = MLBackendService(db)
        record = service.create(
            SYSTEM_TENANT,
            name,
            url,
            model_name,
            model_version,
            model_type,
            confidence_threshold,
            created_by,
        )
        print(
            f"system ml backend seeded: {record['id']} -> {url} "
            f"(status=unknown; the worker health loop will mark it connected)"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
