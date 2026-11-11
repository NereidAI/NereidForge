# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""S3 access for job inputs and artifacts (SeaweedFS gateway in-cluster)."""

from __future__ import annotations

import os
from typing import Any, Optional

S3_ENDPOINT = os.environ.get(
    "S3_ENDPOINT", "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"
)
S3_REGION = os.environ.get("S3_REGION", "us-east-1")


def s3_client():
    """s3_client builds a client only after validating explicit job credentials.

    Credentials are read at the access boundary rather than import time so local
    artifact output remains usable without S3 and tests/job wrappers can inject
    them before the first request. Blank values raise RuntimeError before boto3
    can construct a client or attempt credential discovery.
    """
    access_key = os.environ.get("S3_ACCESS_KEY", "")
    secret_key = os.environ.get("S3_SECRET_KEY", "")
    if not access_key.strip() or not secret_key.strip():
        raise RuntimeError(
            "S3_ACCESS_KEY and S3_SECRET_KEY must contain non-empty "
            "operator-supplied credentials before S3 access."
        )

    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=S3_REGION,
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )


def parse_uri(uri: str) -> tuple[str, str]:
    """Split s3://bucket/prefix into (bucket, prefix)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri}")
    rest = uri[len("s3://"):]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.strip("/")


def list_objects(client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: Optional[str] = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        keys.extend(obj["Key"] for obj in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            return keys
        token = resp.get("NextContinuationToken")


def download_dataset(input_uri: str, local_root: str = "/data") -> str:
    """Download every object under *input_uri* into *local_root*."""
    client = s3_client()
    bucket, prefix = parse_uri(input_uri)
    keys = list_objects(client, bucket, prefix)
    if not keys:
        raise RuntimeError(f"no objects under s3://{bucket}/{prefix}")

    os.makedirs(local_root, exist_ok=True)
    for key in keys:
        rel = key[len(prefix):].lstrip("/")
        if not rel or rel.endswith("/"):
            continue
        dst = os.path.join(local_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        client.download_file(bucket, key, dst)
    print(f"[data] {len(keys)} objects -> {local_root}", flush=True)
    return local_root


def upload_file(local_path: str, output_uri: str, name: str) -> str:
    """Upload one local file to s3://<output_uri>/<name> and return its URI."""
    client = s3_client()
    bucket, prefix = parse_uri(output_uri)
    key = f"{prefix}/{name}" if prefix else name
    client.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def publish_file(local_path: str, output_uri: str, name: str) -> str:
    """Publish one file to s3://... or to a local directory (dev/tests)."""
    if not output_uri:
        return ""
    if not output_uri.startswith("s3://"):
        import shutil

        os.makedirs(output_uri, exist_ok=True)
        dst = os.path.join(output_uri, name)
        shutil.copyfile(local_path, dst)
        return dst
    return upload_file(local_path, output_uri, name)
