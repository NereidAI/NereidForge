# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities: ID generation, pagination, paginated response, datetime helpers."""
from __future__ import annotations

import secrets
import time
from datetime import datetime, timezone
from typing import Any


def new_id(prefix: str) -> str:
    """Generate an ID matching the Go NewID format: {prefix}-{hex12}-{ms%100000}."""
    rand_hex = secrets.token_hex(6)
    ms = int(time.time() * 1000) % 100000
    return f"{prefix}-{rand_hex}-{ms}"


def page_bounds(page: int, page_size: int) -> tuple[int, int]:
    """Return (limit, offset) for a page/page_size pair."""
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20
    return page_size, (page - 1) * page_size


def total_pages(total: int, page_size: int) -> int:
    """Compute the number of pages."""
    if page_size <= 0:
        return 1
    return (total + page_size - 1) // page_size


def paginate(items: list[Any], total: int, page: int, page_size: int) -> dict[str, Any]:
    """Build the standard paginated response envelope."""
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages(total, page_size),
    }


def iso(dt: datetime | None) -> str | None:
    """Serialize a datetime to ISO-8601 (Go-compatible)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def now_utc() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)
