#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# config.sh - Environment variables, defaults, config loading

set -euo pipefail

# Resolve shared repository and centralized operator paths before loading values.
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LIB_DIR/common.sh"

# ========== Three-tier fallback: env var / interactive input / defaults ==========

# Site-specific ingress and NFS inputs stay empty in tracked configuration.
# The production deployment entry point validates them before changing a cluster;
# local development uses the localhost endpoints in infra/configs/config.dev.yaml.
: "${BASE_DOMAIN:=}"
: "${WILDCARD_DOMAIN:=}"
# Optional explicit NF frontend host. When empty, deployment derives
# <tenant>.<base domain> after validating that BASE_DOMAIN is present.
: "${FRONTEND_HOST:=}"

: "${NFS_SERVER:=}"
: "${NFS_SHARE:=}"
: "${STORAGE_CLASS:=nfs-csi}"

# ========== Image versions (tags only) ==========
: "${CNPC_VERSION:=0.29.0}"
: "${POSTGRES_VERSION:=16.4}"
: "${VALKEY_VERSION:=8.0-alpine}"
: "${SEAWEEDFS_VERSION:=4.34}"
: "${LABEL_STUDIO_VERSION:=1.23.0}"
: "${CVAT_VERSION:=2.73.0}"
: "${ISTIO_VERSION:=1.22.3}"
: "${KNATIVE_VERSION:=1.11.0}"
: "${KFP_VERSION:=2.0.0}"
: "${KATIB_VERSION:=0.17.0}"
: "${KSERVE_VERSION:=0.11.0}"
: "${MR_VERSION:=0.3.14}"

# ========== Image coordinates (single public scheme; root treaty section 18) ==========
# All tracked config carries public coordinates only:
#   - class-A self-built images -> ghcr.io/nereidai/<flat-name>:<RELEASE_TAG>,
#     resolved by nf_ref() in common.sh (flat two-level names);
#   - class-B third-party images -> their true upstream references (below).
# There is no registry/profile switch and no internal hostname. Air-gapped
# sites resolve the same refs through the node-level containerd registry
# mirror rendered by NereidOverview deploy/scripts/harbor-sync.sh; the
# manifests stay byte-identical online and offline.
# No VERSION file (root treaty §19.5): RELEASE_TAG is the bare SemVer of the
# exact v* git tag, or 0.0.0-dev on any non-tag checkout; an operator/CI
# environment may still override it explicitly.
: "${RELEASE_TAG:=$(nf_version)}"
export RELEASE_TAG

# class B: pristine upstream references
IMG_POSTGRESQL="ghcr.io/cloudnative-pg/postgresql:$POSTGRES_VERSION"
IMG_POSTGRESQL_INSTANCE_MGR="ghcr.io/cloudnative-pg/postgresql-instance-manager:$POSTGRES_VERSION"
IMG_VALKEY="valkey/valkey:$VALKEY_VERSION"
IMG_SEAWEEDFS="chrislusf/seaweedfs:$SEAWEEDFS_VERSION"
IMG_OPENVINO="openvino/model_server:latest"
IMG_TRITONSERVER="nvcr.io/nvidia/tritonserver:26.01-py3"
IMG_PYTORCH="pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime"
IMG_TGI="ghcr.io/huggingface/text-generation-inference:latest"
IMG_CVAT_SERVER="cvat/server:v$CVAT_VERSION"
IMG_CVAT_UI="cvat/ui:v$CVAT_VERSION"
IMG_LABEL_STUDIO="heartexlabs/label-studio:$LABEL_STUDIO_VERSION"
IMG_LABEL_STUDIO_ML_BACKEND="heartexlabs/label-studio-ml-backend:$LABEL_STUDIO_VERSION"
IMG_OPA="openpolicyagent/opa:1.12.2"
IMG_PYTHON_BASE="python:3.11-slim"
IMG_PYTHON_314_BASE="python:3.14-slim"
IMG_LLAMAFACTORY="hiyouga/llamafactory:0.9.4"
# Upstream has no short "rocm6.3" tag; use the full py3.12/ubuntu24.04 tag.
IMG_ROCM_PYTORCH="rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0"
IMG_JUPYTER_BASE="quay.io/jupyter/base-notebook:python-3.12.9"
IMG_BUSYBOX="busybox:1.36"
# NF-owned utility containers follow the suite Alpine baseline. Vendored
# upstream products keep their own product-pinned Alpine versions.
IMG_ALPINE="alpine:3.24.4"
IMG_CURL="curlimages/curl:latest"
IMG_KUBECTL="alpine/k8s:1.36.2"

# Image paths used by scripts (env may override any of them).
# NOTE: cert-manager + csi-driver-nfs images moved to the suite prerequisites
# (NereidOverview/deploy/prereqs, 2026-10-19); no IMG_CERT_MANAGER_*/IMG_CSI_* here.
: "${IMG_POSTGRESQL:?}"; : "${IMG_POSTGRESQL_INSTANCE_MGR:?}"
: "${IMG_VALKEY:?}"; : "${IMG_SEAWEEDFS:?}"
: "${IMG_OPENVINO:?}"; : "${IMG_TRITONSERVER:?}"; : "${IMG_PYTORCH:?}"; : "${IMG_TGI:?}"
: "${IMG_CVAT_SERVER:?}"; : "${IMG_CVAT_UI:?}"; : "${IMG_LABEL_STUDIO:?}"
: "${IMG_LABEL_STUDIO_ML_BACKEND:?}"; : "${IMG_OPA:?}"
: "${IMG_PYTHON_BASE:?}"; : "${IMG_PYTHON_314_BASE:?}"; : "${IMG_LLAMAFACTORY:?}"; : "${IMG_ROCM_PYTORCH:?}"; : "${IMG_JUPYTER_BASE:?}"
: "${IMG_BUSYBOX:?}"; : "${IMG_ALPINE:?}"; : "${IMG_CURL:?}"; : "${IMG_KUBECTL:?}"
export IMG_POSTGRESQL IMG_POSTGRESQL_INSTANCE_MGR IMG_VALKEY IMG_SEAWEEDFS
export IMG_OPENVINO IMG_TRITONSERVER IMG_PYTORCH IMG_TGI
export IMG_CVAT_SERVER IMG_CVAT_UI IMG_LABEL_STUDIO IMG_LABEL_STUDIO_ML_BACKEND IMG_OPA
export IMG_PYTHON_BASE IMG_PYTHON_314_BASE IMG_LLAMAFACTORY IMG_ROCM_PYTORCH IMG_JUPYTER_BASE
export IMG_BUSYBOX IMG_ALPINE IMG_CURL IMG_KUBECTL

# Build network endpoints for self-built images. Generic empty values select
# pristine upstream defaults (PyPI, apt, Hugging Face). XPU/ROCm/Paddle-GPU
# indexes have no generic substitute: their build owners reject empty values.
# Unset-only defaults preserve explicitly injected values, including empties.
: "${PIP_INDEX_URL:=}"
: "${PIP_TRUSTED_HOST:=}"
: "${APT_MIRROR:=}"
: "${TORCH_CPU_INDEX:=}"
: "${TORCH_XPU_INDEX:=}"
: "${TORCH_ROCM_INDEX:=}"
: "${HF_ENDPOINT:=}"
: "${PADDLE_GPU_INDEX:=}"
export PIP_INDEX_URL PIP_TRUSTED_HOST APT_MIRROR TORCH_CPU_INDEX TORCH_XPU_INDEX TORCH_ROCM_INDEX HF_ENDPOINT PADDLE_GPU_INDEX

# Self-built images (class A). nf_ref() always renders the public flat GHCR
# reference ghcr.io/nereidai/<flat-name>:<RELEASE_TAG>.
: "${NF_BACKEND_IMAGE:=$(nf_ref nereidforge-backend "$RELEASE_TAG")}"
: "${NF_FRONTEND_IMAGE:=$(nf_ref nereidforge-frontend "$RELEASE_TAG")}"
: "${NF_TRAINER_IMAGE:=$(nf_ref trainer/torch-cpu "$RELEASE_TAG")}"
: "${IMG_ML_BACKEND:=$(nf_ref ml-backend/ml-backend "$RELEASE_TAG")}"
# Self-built vendor base (Ubuntu + Intel XPU userspace + torch+xpu).
IMG_XPU_BASE="$(nf_ref trainer/xpu-base "$RELEASE_TAG")"
export NF_BACKEND_IMAGE NF_FRONTEND_IMAGE NF_TRAINER_IMAGE IMG_ML_BACKEND IMG_XPU_BASE

# kubectl wrapper (local cluster access, not 60_verify test pod images).
# Defaults to bare kubectl; can be overridden via env (e.g., KUBECTL="sudo kubectl").
: "${KUBECTL:=kubectl}"

# PostgreSQL config
: "${PG_CLUSTER_NAME:=nereidforge-postgresql}"
: "${PG_INSTANCES:=3}"
: "${PG_STORAGE_SIZE:=50Gi}"
: "${PG_DB_NAME:=nereidforge}"
: "${PG_DB_USER:=nereidforge}"
: "${PG_ADMIN_PASSWORD:=}"

# Valkey config

# SeaweedFS config. Credentials are operator inputs; tracked defaults must
# never create a working object-store identity.
: "${SEAWEEDFS_MASTER_STORAGE:=10Gi}"
: "${SEAWEEDFS_VOLUME_STORAGE:=200Gi}"
: "${SEAWEEDFS_S3_ACCESS_KEY:=}"
: "${SEAWEEDFS_S3_SECRET_KEY:=}"

# Label Studio config
: "${LABEL_STUDIO_STORAGE:=20Gi}"

# CVAT config
: "${CVAT_STORAGE:=50Gi}"

# JWT secret for the NF API. Must be non-empty (the app refuses to boot with an
# empty HMAC key). No secret is baked into the repository: when the operator
# does not supply one, generate an ephemeral per-environment key and warn,
# because a regenerated key invalidates every outstanding token. Pin an
# explicit JWT_SECRET (the full deploy chain validates it as required) for any
# environment whose tokens must survive a rerun.
if [ -z "${JWT_SECRET:-}" ]; then
  JWT_SECRET="$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | xxd -p -c 64)"
  echo "WARNING: JWT_SECRET not provided; generated an ephemeral key. Set JWT_SECRET explicitly for stable deploys." >&2
fi
: "${JWT_SECRET:?JWT secret generation failed}"

# Dex identity and credentials are operator inputs. The plaintext is used only
# by the authenticated verification probe; the bcrypt hash is rendered into the
# temporary Dex Secret. Neither value has a tracked fallback.
: "${DEX_STATIC_EMAIL:=}"
: "${DEX_STATIC_PASSWORD:=}"
: "${DEX_STATIC_PASSWORD_HASH:=}"

# Cluster pod/service CIDRs (k3s defaults). The tenant mesh-RBAC carve-out uses
# both ranges for non-meshed in-cluster callers; site operators must override
# them when their cluster uses different networks.
: "${POD_CIDR:=10.42.0.0/16}"
: "${SERVICE_CIDR:=10.43.0.0/16}"

# cert-manager needs an email only for a Let's Encrypt issuer, not internal-ca.
: "${LETSENCRYPT_EMAIL:=}"

# Paths. Tracked chart values and templates stay in Git; only generated or
# operator-supplied sensitive values live under the centralized deployment root.
MANIFESTS_DIR="$ROOT_DIR/infra/third_party/kubeflow-manifests"
CHARTS_DIR="$ROOT_DIR/infra/charts"
APP_CHARTS_DIR="$ROOT_DIR/app/charts"
VALUES_FILE="$ROOT_DIR/values-prod.yaml"
SECRETS_FILE="${SECRETS_FILE:-$DEPLOY_VALUES_DIR/secrets.yaml}"
SECRETS_TEMPLATE="$ROOT_DIR/secrets.yaml.template"
: "${OFFLINE_DIR:=$ROOT_DIR/offline-package}"

# Export all variables for child scripts
export_all() {
    export BASE_DOMAIN WILDCARD_DOMAIN FRONTEND_HOST
    export NFS_SERVER NFS_SHARE STORAGE_CLASS
    export CNPC_VERSION POSTGRES_VERSION VALKEY_VERSION SEAWEEDFS_VERSION
    export LABEL_STUDIO_VERSION CVAT_VERSION ISTIO_VERSION KNATIVE_VERSION
    export KFP_VERSION KATIB_VERSION KSERVE_VERSION MR_VERSION
    export NF_BACKEND_IMAGE NF_FRONTEND_IMAGE NF_TRAINER_IMAGE IMG_ML_BACKEND IMG_XPU_BASE
    export RELEASE_TAG
    export PG_CLUSTER_NAME PG_INSTANCES PG_STORAGE_SIZE PG_DB_NAME PG_DB_USER PG_ADMIN_PASSWORD
    export SEAWEEDFS_MASTER_STORAGE SEAWEEDFS_VOLUME_STORAGE SEAWEEDFS_S3_ACCESS_KEY SEAWEEDFS_S3_SECRET_KEY
    export LABEL_STUDIO_STORAGE CVAT_STORAGE
    export JWT_SECRET DEX_STATIC_EMAIL DEX_STATIC_PASSWORD DEX_STATIC_PASSWORD_HASH
    export LETSENCRYPT_EMAIL POD_CIDR SERVICE_CIDR
    export MANIFESTS_DIR CHARTS_DIR APP_CHARTS_DIR VALUES_FILE SECRETS_FILE SECRETS_TEMPLATE OFFLINE_DIR
    export ROOT_DIR LIB_DIR WORKSPACE_ROOT KUBECTL
    export DEPLOYMENT_ROOT DEPLOY_VALUES_DIR DEPLOY_LOG_DIR NF_ENV_FILE NF_DEV_ENV_FILE NF_HANDOFF_DIR
    # Export all image variables
    export IMG_POSTGRESQL IMG_POSTGRESQL_INSTANCE_MGR
    export IMG_VALKEY IMG_SEAWEEDFS
    export IMG_OPENVINO IMG_TRITONSERVER IMG_PYTORCH IMG_TGI
    export IMG_CVAT_SERVER IMG_CVAT_UI IMG_LABEL_STUDIO IMG_LABEL_STUDIO_ML_BACKEND IMG_OPA
    export IMG_PYTHON_BASE IMG_PYTHON_314_BASE IMG_LLAMAFACTORY IMG_ROCM_PYTORCH IMG_JUPYTER_BASE
    export IMG_BUSYBOX IMG_ALPINE IMG_CURL IMG_KUBECTL
    export PIP_INDEX_URL PIP_TRUSTED_HOST APT_MIRROR
    export TORCH_CPU_INDEX TORCH_XPU_INDEX TORCH_ROCM_INDEX HF_ENDPOINT PADDLE_GPU_INDEX
}

# Load centralized site values when present. NF_ENV_FILE is resolved by
# common.sh from DEPLOY_VALUES_DIR unless the operator overrides it directly.
if [[ -f "$NF_ENV_FILE" ]]; then
    set -a
    source "$NF_ENV_FILE"
    set +a
fi

# Derive the wildcard only after site values are loaded, and never render "*"
# when the operator has not supplied a base domain.
if [[ -n "$BASE_DOMAIN" && -z "$WILDCARD_DOMAIN" ]]; then
    WILDCARD_DOMAIN="*.$BASE_DOMAIN"
fi

# Export all variables
export_all
