# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NereidForge trainer SDK (`nereidforge-trainer`).

One contract for every training component, in-tree or customer-owned:

    export INPUT_URI=s3://bucket/datasets/x/   # dataset root
    export OUTPUT_DIR=s3://bucket/artifacts/y/ # artifact destination
    export CONFIG_JSON='{"task": "detection", ...}'
    export ARTIFACT_ID=job-123
    python train.py

The platform never imports a trainer; a trainer never imports the platform.
Everything the two sides share lives here. See `builders/sdk/README.md`
for the capability profile that each image announces.
"""

from __future__ import annotations

from .augment import (
    AUG_BOUNDS,
    AUG_PRESETS,
    AUG_UNSUPPORTED,
    augmentation_summary,
    resolve_augmentation,
)
from .manifest import MANIFEST_SCHEMA_VERSION, build_manifest
from .profile import (
    PROFILE_SCHEMA,
    architecture_entry,
    capability_entry,
    effective_tier,
    emit,
    load_profile,
    missing_entitlements,
    profile_requirements,
    task_entry,
    validate_profile,
)
from .runtime import (
    TRAINING_METRICS_SCHEMA,
    Progress,
    build_training_metrics,
    read_config,
    validate_training_metrics,
    write_json,
)
from .s3 import (
    download_dataset,
    list_objects,
    parse_uri,
    publish_file,
    s3_client,
    upload_file,
)

#: Rolling dev-channel default under the tag-sourced version model (root
#: treaty §19.5): the repository has no VERSION file and off-tag checkouts are
#: unreleased, so the Python package carries the PEP 440 spelling of the
#: `0.0.0-dev` image channel (guarded by the SDK tests).
__version__ = "0.0.0.dev0"

__all__ = [
    "AUG_BOUNDS",
    "AUG_PRESETS",
    "AUG_UNSUPPORTED",
    "MANIFEST_SCHEMA_VERSION",
    "PROFILE_SCHEMA",
    "TRAINING_METRICS_SCHEMA",
    "Progress",
    "__version__",
    "augmentation_summary",
    "build_manifest",
    "build_training_metrics",
    "architecture_entry",
    "capability_entry",
    "download_dataset",
    "effective_tier",
    "emit",
    "load_profile",
    "list_objects",
    "missing_entitlements",
    "parse_uri",
    "profile_requirements",
    "publish_file",
    "read_config",
    "resolve_augmentation",
    "s3_client",
    "task_entry",
    "upload_file",
    "validate_profile",
    "validate_training_metrics",
    "write_json",
]
