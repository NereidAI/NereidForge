# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Short-TTL key/value store for single-use records (captcha, OIDC state).

Postgres is the primary store: the API runs multiple uvicorn workers and
multiple replicas, so the answer to ``GET /captcha`` and the subsequent
``POST /login`` may land on different processes. A bounded in-process dict
remains as a fallback (DB briefly unreachable, unit tests); it is not
correct across processes and must not be relied on in production.

Semantics: ``put`` writes with a TTL, ``pop`` is single-use
(read + delete) and returns None for missing or expired rows.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import delete

from app.db.session import SessionLocal
from app.models.orm import KvObject

log = logging.getLogger(__name__)

# Namespace constants.
CAPTCHA = "captcha"
SSO_STATE = "sso_state"

_MAX_MEM = 2000
_mem: dict[tuple[str, str], tuple[float, str]] = {}
_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _mem_put(namespace: str, key: str, value: Any, ttl: int) -> None:
    with _lock:
        now = time.time()
        for k in [k for k, (exp, _) in _mem.items() if exp <= now]:
            _mem.pop(k, None)
        overflow = len(_mem) - _MAX_MEM + 1
        if overflow > 0:
            for k in sorted(_mem, key=lambda k: _mem[k][0])[:overflow]:
                _mem.pop(k, None)
        _mem[(namespace, key)] = (now + ttl, json.dumps(value))


def _mem_pop(namespace: str, key: str) -> Optional[dict]:
    with _lock:
        entry = _mem.pop((namespace, key), None)
    if entry is None:
        return None
    exp, raw = entry
    if exp <= time.time():
        return None
    return json.loads(raw)


def kv_put(namespace: str, key: str, value: dict, ttl: int) -> None:
    """Store a JSON-serializable value with TTL seconds."""
    try:
        db = SessionLocal()
        try:
            db.merge(
                KvObject(
                    namespace=namespace,
                    key=key,
                    value=json.dumps(value),
                    expires_at=_now() + timedelta(seconds=ttl),
                )
            )
            db.commit()
            # Cheap opportunistic garbage collection: ~1% of writes sweep
            # expired rows; the table holds only short-lived records.
            if hash((namespace, key)) % 100 == 0:
                db.execute(delete(KvObject).where(KvObject.expires_at < _now()))
                db.commit()
        finally:
            db.close()
    except Exception:  # noqa: BLE001 — KV failures must degrade, not break login
        log.warning("kv_store: DB put failed, using in-memory fallback", exc_info=True)
        _mem_put(namespace, key, value, ttl)


def kv_pop(namespace: str, key: str) -> Optional[dict]:
    """Consume a value once; None when missing or expired."""
    try:
        db = SessionLocal()
        try:
            row = db.get(KvObject, (namespace, key))
            if row is not None:
                db.delete(row)
                db.commit()
                expires_at = row.expires_at
                if expires_at.tzinfo is None:  # sqlite returns naive; PG timestamptz not
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                if expires_at < _now():
                    return None
                return json.loads(row.value)
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        log.warning("kv_store: DB pop failed, using in-memory fallback", exc_info=True)
    # Covered when the row never reached the DB (put happened during a DB flap).
    return _mem_pop(namespace, key)
