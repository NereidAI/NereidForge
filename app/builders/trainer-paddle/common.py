# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Runtime helpers for the NF Paddle trainer — provided by the Trainer SDK.

Same thin facade as the torch trainer: the implementation lives in the published
`nereidforge_trainer` package (ADR-016) so in-tree trainers and customer-owned
trainers share one contract.
"""

from nereidforge_trainer import (  # noqa: F401
    MANIFEST_SCHEMA_VERSION,
    Progress,
    build_manifest,
    build_training_metrics,
    download_dataset,
    list_objects,
    parse_uri,
    publish_file,
    read_config,
    s3_client,
    upload_file,
    validate_profile,
    write_json,
)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "Progress",
    "build_manifest",
    "build_training_metrics",
    "download_dataset",
    "list_objects",
    "parse_uri",
    "publish_file",
    "read_config",
    "s3_client",
    "upload_file",
    "validate_profile",
    "write_json",
]
