#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# ml-backend.sh deploys the YOLO pre-annotation ML Backend Helm chart.

set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)"
source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

ROOT_DIR="$(cd "$LIB_DIR/../.." && pwd)"
APP_CHARTS_DIR="$ROOT_DIR/charts"
NAMESPACE="nereidforge"
# RELEASE_TAG (bare SemVer, exported by lib/config.sh) is the tag-sourced
# release version; the chart's image tag is overridden with it below.
VERSION="${RELEASE_TAG:?config.sh failed to provide RELEASE_TAG}"

log_info "========== Deploy ML Backend (YOLO pre-annotation, Helm) =========="

# Use the fixed Label Studio DRF token (same source as backend.sh). The
# bootstrap owner writes it below NF_HANDOFF_DIR; development values remain an
# optional fallback. Never include any part of the token in console/deploy logs.
LS_TOKEN="${NEREIDFORGE_LABELSTUDIO_TOKEN:-}"
TOKEN_FILE="$NF_HANDOFF_DIR/service-tokens.env"
if [[ -z "$LS_TOKEN" && -f "$TOKEN_FILE" ]]; then
    # shellcheck disable=SC1091
    source "$TOKEN_FILE"
    LS_TOKEN="${NEREIDFORGE_LABELSTUDIO_TOKEN:-}"
fi
if [[ -z "$LS_TOKEN" && -f "$NF_DEV_ENV_FILE" ]]; then
    # shellcheck disable=SC1091
    source "$NF_DEV_ENV_FILE"
    LS_TOKEN="${NEREIDFORGE_LABELSTUDIO_TOKEN:-}"
fi
if [[ -z "$LS_TOKEN" ]]; then
    log_err "NEREIDFORGE_LABELSTUDIO_TOKEN is missing from $TOKEN_FILE and $NF_DEV_ENV_FILE; cannot inject LABEL_STUDIO_API_KEY"
    exit 1
fi

log_info "Using the Label Studio token from centralized deployment configuration (credential not logged)"

cd "$APP_CHARTS_DIR/ml-backend"
# The chart's default image is the public flat GHCR reference; only the
# VERSION-derived tag is overridden here. Air-gapped nodes resolve the public
# ref via the containerd registry mirror, no in-pod pull secret.
helm upgrade --install ml-backend . \
    --namespace "$NAMESPACE" --create-namespace \
    --set "lsToken=$LS_TOKEN" \
    --set "image.tag=$VERSION" \
    --wait \
    --timeout 10m

wait_deployment_ready "ml-backend" "$NAMESPACE"

log_ok "ML Backend deployment complete"
log_info "In-cluster service: http://ml-backend.$NAMESPACE.svc.cluster.local:9090"
