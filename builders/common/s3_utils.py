# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Shared S3 utilities for all builder entrypoints (trainer, evaluator, optimizer).

Provides a unified S3 client factory, URI parsing, and file operations against
SeaweedFS (S3-compatible). All credentials come from environment variables so
nothing is baked into images.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import boto3


def get_s3_endpoint() -> str:
    """Return the S3 endpoint URL from env, falling back to the in-cluster default."""
    return os.environ.get(
        "S3_ENDPOINT",
        "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333",
    )


def make_s3_client() -> Any:
    """make_s3_client creates a boto3 client with explicit operator credentials.

    S3_ENDPOINT remains optional for in-cluster jobs. S3_ACCESS_KEY and
    S3_SECRET_KEY must both be non-empty; RuntimeError is raised before client
    construction so a misconfigured job cannot attempt anonymous S3 access.
    """
    access_key = os.environ.get("S3_ACCESS_KEY", "")
    secret_key = os.environ.get("S3_SECRET_KEY", "")
    if not access_key.strip() or not secret_key.strip():
        raise RuntimeError(
            "S3_ACCESS_KEY and S3_SECRET_KEY must contain non-empty "
            "operator-supplied credentials before S3 access."
        )
    return boto3.client(
        "s3",
        endpoint_url=get_s3_endpoint(),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
        config=boto3.session.Config(signature_version="s3v4"),
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an s3://bucket/key URI into (bucket, key).

    Leading/trailing slashes are trimmed from the key.
    """
    rest = uri[len("s3://"):]
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key.rstrip("/")


def list_s3_objects(s3: Any, bucket: str, prefix: str) -> list[str]:
    """List all object keys under *prefix* in *bucket*."""
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def upload_progress(s3: Any, bucket: str, okey: str, epoch: int, total: int, loss: float) -> None:
    """Upload a small progress.json so the API/UI can show live training progress."""
    try:
        data = {
            "current_epoch": epoch,
            "total_epochs": total,
            "progress_pct": int(round(epoch * 100 / total)) if total else 0,
            "loss": loss,
            "updated_at": time.time(),
        }
        s3.put_object(Bucket=bucket, Key=okey.rstrip("/") + "/progress.json", Body=json.dumps(data))
    except Exception as e:
        print(f"[common] progress upload failed: {e}", flush=True)
