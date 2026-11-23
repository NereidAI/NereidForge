# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Runtime helpers for the NF vision trainer — now provided by the Trainer SDK.

Kept as a thin, stable import surface so every trainer module reads
`from common import …` (the historical contract) while the implementation lives
in the published SDK (`nereidforge_trainer`, ADR-016). Nothing here is
framework-specific.

When adding a helper, add it to the SDK, not to this file.
"""

from nereidforge_trainer import (  # noqa: F401
    AUG_BOUNDS,
    AUG_PRESETS,
    AUG_UNSUPPORTED,
    MANIFEST_SCHEMA_VERSION,
    Progress,
    augmentation_summary,
    build_manifest,
    build_training_metrics,
    download_dataset,
    list_objects,
    parse_uri,
    publish_file,
    read_config,
    resolve_augmentation,
    s3_client,
    upload_file,
    validate_profile,
    write_json,
)

__all__ = [
    "AUG_BOUNDS",
    "AUG_PRESETS",
    "AUG_UNSUPPORTED",
    "MANIFEST_SCHEMA_VERSION",
    "Progress",
    "augmentation_summary",
    "build_manifest",
    "build_training_metrics",
    "download_dataset",
    "list_objects",
    "parse_uri",
    "publish_file",
    "read_config",
    "resolve_augmentation",
    "s3_client",
    "upload_file",
    "validate_profile",
    "write_json",
]
