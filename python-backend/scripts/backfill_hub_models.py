# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""One-off/idempotent backfill: register all PG model versions into Hub.

The worker best-effort-registers trained models to the Kubeflow Model Registry,
but for a period the hub host pointed at a non-existent service
(model-registry-service.nereidforge; it lives in `default`), so
hub.enabled() was effectively false and every registration was skipped — PG
kept the rows while the Hub dashboard stayed empty.

This script reconciles PG -> Hub: models/versions already present in the Hub
are skipped, so it is safe to re-run. Hub custom properties must be strings,
so nested metadata is JSON-serialized.

Run inside the api pod:
    kubectl cp python-backend/scripts/backfill_hub_models.py <api-pod>:/tmp/
    kubectl exec <api-pod> -- env PYTHONPATH=/app python /tmp/backfill_hub_models.py
"""
import json

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.orm import Model as ModelORM
from app.models.orm import ModelVersion as ModelVersionORM
from app.services import Services


def fmt_of(uri: str, framework: str) -> str:
    """Derive the hub format tag from the primary file extension."""
    low = (uri or "").lower()
    if low.endswith(".onnx"):
        return "onnx"
    if low.endswith(".pt") or low.endswith(".pth"):
        return "pt"
    if low.endswith(".json"):
        return "safetensors" if framework == "llm" else "json"
    return "safetensors" if framework == "llm" else "1"


def flatten_metadata(meta: dict) -> dict:
    """Hub custom properties only accept string values."""
    out = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        out[k] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return out


def main() -> None:
    """Register every missing model/version; print a per-version result."""
    db = SessionLocal()
    svc = Services(db)
    hub = svc.model.hub
    if not hub or not hub.enabled():
        raise SystemExit("hub is not enabled (check hub.host)")

    models = db.scalars(select(ModelORM)).all()
    registered = 0
    skipped = 0
    failed = 0
    for m in models:
        versions = db.scalars(
            select(ModelVersionORM)
            .where(ModelVersionORM.model_id == m.id)
            .order_by(ModelVersionORM.created_at)
        ).all()
        existing: set[str] = set()
        try:
            for hv in hub.get_model_versions(m.name) or []:
                existing.add(getattr(hv, "name", ""))
        except Exception:
            existing = set()
        for v in versions:
            meta = dict(v.metadata_ or {})
            uri = str(meta.get("onnx_uri", "") or v.file_path or "")
            if not uri:
                print(f"SKIP {m.name}/{v.version}: no uri")
                skipped += 1
                continue
            if v.version in existing:
                print(f"SKIP {m.name}/{v.version}: already in hub")
                skipped += 1
                continue
            meta.setdefault("framework", m.framework or "pytorch")
            try:
                ok = svc.model.register_to_hub(
                    m.name, uri, v.version, fmt_of(uri, m.framework),
                    flatten_metadata(meta),
                )
                print(f"{'OK  ' if ok else 'FAIL'} {m.name}/{v.version} -> {uri}")
                registered += 1 if ok else 0
                failed += 0 if ok else 1
            except Exception as e:  # noqa: BLE001 - keep backfilling other versions
                print(f"FAIL {m.name}/{v.version}: {e}")
                failed += 1
    print(f"done: registered={registered} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
