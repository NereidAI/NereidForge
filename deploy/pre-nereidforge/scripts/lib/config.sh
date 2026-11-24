#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Environment and defaults consumed by pre-NereidForge scripts only.

set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$LIB_DIR/common.sh"

# Operator output remains outside the module. Full-stack callers inherit their
# existing paths; standalone archives use a sibling .deployment directory.
WORKSPACE_ROOT="$(cd "$(dirname "${REPO_ROOT:-$PRE_ROOT}")" && pwd)"
: "${DEPLOYMENT_ROOT:=$WORKSPACE_ROOT/.deployment}"
: "${DEPLOY_VALUES_DIR:=$DEPLOYMENT_ROOT/deploy-values/pre-nereidforge}"
: "${DEPLOY_LOG_DIR:=$DEPLOYMENT_ROOT/deploy-logs/pre-nereidforge}"

if [[ -z "${PRE_ENV_FILE:-}" ]]; then
    if [[ -n "${NF_ENV_FILE:-}" ]]; then
        PRE_ENV_FILE="$NF_ENV_FILE"
    else
        PRE_ENV_FILE="$DEPLOY_VALUES_DIR/site.env"
    fi
fi
if [[ -z "${PRE_HANDOFF_DIR:-}" ]]; then
    PRE_HANDOFF_DIR="${NF_HANDOFF_DIR:-$DEPLOY_VALUES_DIR}"
fi
# Existing full-stack scripts and saved handoffs keep their established names.
NF_ENV_FILE="$PRE_ENV_FILE"
NF_HANDOFF_DIR="$PRE_HANDOFF_DIR"

PRE_RELEASE_TAG="$(pre_version)"

# Infrastructure and Kubeflow site inputs.
: "${BASE_DOMAIN:=}"
: "${STORAGE_CLASS:=nfs-csi}"
: "${POD_CIDR:=10.42.0.0/16}"
: "${SERVICE_CIDR:=10.43.0.0/16}"
: "${KUBECTL:=kubectl}"

# Only third-party images directly consumed by pre scripts belong here.
: "${POSTGRES_VERSION:=16.4}"
: "${VALKEY_VERSION:=8.0-alpine}"
: "${SEAWEEDFS_VERSION:=4.34}"
: "${LABEL_STUDIO_VERSION:=1.23.0}"
: "${IMG_POSTGRESQL:=ghcr.io/cloudnative-pg/postgresql:$POSTGRES_VERSION}"
: "${IMG_VALKEY:=valkey/valkey:$VALKEY_VERSION}"
: "${IMG_SEAWEEDFS:=chrislusf/seaweedfs:$SEAWEEDFS_VERSION}"
: "${IMG_LABEL_STUDIO:=heartexlabs/label-studio:$LABEL_STUDIO_VERSION}"
: "${IMG_BUSYBOX:=busybox:1.36}"

: "${PG_CLUSTER_NAME:=nereidforge-postgresql}"
: "${PG_INSTANCES:=3}"
: "${PG_STORAGE_SIZE:=50Gi}"
: "${PG_DB_NAME:=nereidforge}"
: "${PG_DB_USER:=nereidforge}"
: "${PG_ADMIN_PASSWORD:=}"

: "${SEAWEEDFS_MASTER_STORAGE:=10Gi}"
: "${SEAWEEDFS_VOLUME_STORAGE:=200Gi}"
: "${SEAWEEDFS_S3_ACCESS_KEY:=}"
: "${SEAWEEDFS_S3_SECRET_KEY:=}"
: "${LABEL_STUDIO_STORAGE:=20Gi}"

# Dex is a Kubeflow input: installation consumes the hash and mr-check consumes
# the plaintext identity. The vendored Dex, Profile, and RBAC manifests currently
# use this fixed identity; 30_install-kubeflow.sh rejects a conflicting override.
# No application JWT or NF application secret is made.
: "${DEX_STATIC_EMAIL:=admin@nereidforge.local}"
: "${DEX_STATIC_PASSWORD:=}"
: "${DEX_STATIC_PASSWORD_HASH:=}"

MANIFESTS_DIR="$PRE_ROOT/platform/kubeflow-manifests"
CHARTS_DIR="$PRE_ROOT/charts"
: "${OFFLINE_DIR:=$PRE_ROOT/offline-package}"

export_all() {
    export PRE_ROOT REPO_ROOT LIB_DIR WORKSPACE_ROOT PRE_RELEASE_TAG SOURCE_COMMIT
    export DEPLOYMENT_ROOT DEPLOY_VALUES_DIR DEPLOY_LOG_DIR
    export PRE_ENV_FILE PRE_HANDOFF_DIR NF_ENV_FILE NF_HANDOFF_DIR
    export BASE_DOMAIN STORAGE_CLASS POD_CIDR SERVICE_CIDR KUBECTL
    export POSTGRES_VERSION VALKEY_VERSION SEAWEEDFS_VERSION LABEL_STUDIO_VERSION
    export IMG_POSTGRESQL IMG_VALKEY IMG_SEAWEEDFS IMG_LABEL_STUDIO IMG_BUSYBOX
    export PG_CLUSTER_NAME PG_INSTANCES PG_STORAGE_SIZE PG_DB_NAME PG_DB_USER PG_ADMIN_PASSWORD
    export SEAWEEDFS_MASTER_STORAGE SEAWEEDFS_VOLUME_STORAGE
    export SEAWEEDFS_S3_ACCESS_KEY SEAWEEDFS_S3_SECRET_KEY LABEL_STUDIO_STORAGE
    export DEX_STATIC_EMAIL DEX_STATIC_PASSWORD DEX_STATIC_PASSWORD_HASH
    export MANIFESTS_DIR CHARTS_DIR OFFLINE_DIR
}

if [[ -f "$PRE_ENV_FILE" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PRE_ENV_FILE"
    set +a
fi

export_all
