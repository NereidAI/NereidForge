#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Idempotently provision and authenticate the NereidForge S3 buckets."""

import os

BUCKETS = ("mlpipeline", "nereidforge", "labelstudio", "cvat")


def _response_status(response):
    metadata = response.get("ResponseMetadata", {}) if isinstance(response, dict) else {}
    return metadata.get("HTTPStatusCode")


def _require_success(response, operation, bucket):
    status = _response_status(response)
    if status != 200:
        raise RuntimeError(
            f"{operation} failed for bucket {bucket!r}: HTTP status {status!r}"
        )


def _error_details(error):
    response = getattr(error, "response", {})
    if not isinstance(response, dict):
        return "", None
    error_info = response.get("Error", {})
    code = str(error_info.get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code, status


def _is_missing_bucket(error):
    code, status = _error_details(error)
    # Never let a contradictory or non-standard error payload turn an
    # authentication/authorization failure into a create attempt.
    if status is not None and status != 404:
        return False
    return status == 404 or code in ("404", "NoSuchBucket", "NotFound")


def ensure_bucket(s3, bucket):
    """Create a missing bucket, then prove authenticated read/list access."""
    created = False
    try:
        head_response = s3.head_bucket(Bucket=bucket)
    except Exception as error:
        if not _is_missing_bucket(error):
            raise
        try:
            create_response = s3.create_bucket(Bucket=bucket)
        except Exception as create_error:
            code, _ = _error_details(create_error)
            # A concurrent idempotent installer may win the creation race. It is
            # accepted only after the authenticated HeadBucket check below.
            if code != "BucketAlreadyOwnedByYou":
                raise
        else:
            _require_success(create_response, "CreateBucket", bucket)
        created = True

    if created:
        head_response = s3.head_bucket(Bucket=bucket)
    _require_success(head_response, "HeadBucket", bucket)

    # This authenticated operation verifies more than bucket existence and
    # leaves no probe objects behind. In particular, a 403 is never success.
    list_response = s3.list_objects_v2(Bucket=bucket, MaxKeys=1)
    _require_success(list_response, "ListObjectsV2", bucket)


def ensure_buckets(s3, buckets=BUCKETS):
    for bucket in buckets:
        ensure_bucket(s3, bucket)
        print(f"Provisioned and verified S3 bucket: {bucket}")


def main():
    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )
    ensure_buckets(s3)


if __name__ == "__main__":
    main()
