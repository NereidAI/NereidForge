#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# cvat.sh - Deploy CVAT (official helm chart v2.73.0)
# Uses charts/cvat + infra/configs/cvat-values.yaml (external PG/Valkey/S3,
# public upstream images, ingress signed by internal-ca). nginx sidecar /
# smokescreen / migration are handled by the official chart itself.

set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)"
source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

ROOT_DIR="$(cd "$LIB_DIR/../../.." && pwd)"
CHART="$ROOT_DIR/infra/charts/cvat"
VALUES="$ROOT_DIR/infra/configs/cvat-values.yaml"

log_info "========== Deploy CVAT (official chart) =========="

# Secrets (cvat-postgres-secret / cvat-valkey-secret / cvat-kvrocks-secret) are
# rendered by the chart (templates/zz-nereidforge-secrets.yaml) from the CNPG
# bootstrap secret via lookup.

# The overlay (infra/configs/cvat-values.yaml) defaults to public upstream
# images. Air-gapped nodes resolve them through the containerd registry
# mirror; no image override and no in-pod pull secret (root treaty section 18).

# Stage the vendored subchart tgz files (committed at the chart root for
# offline installs; charts/ itself is gitignored working state). Rebuild
# charts/ from scratch: a partial unpacking left by an interrupted
# `helm dependency build` breaks loading with "error unpacking subchart ...
# Chart.yaml file is missing". Helm 3.18 hard-requires every dependency in
# charts/ even when its condition is disabled, so all six permissible tgz files
# (incl. the disabled nuclio/vector/traefik/redis) are vendored. Grafana is not
# a dependency: its AGPL server and chart are outside the delivery policy.
rm -rf "$CHART/charts"
mkdir -p "$CHART/charts"
cp -f "$CHART"/*.tgz "$CHART/charts/"

helm upgrade --install cvat "$CHART" -n nereidforge --create-namespace -f "$VALUES" \
    --set ingress.hostname="cvat.$BASE_DOMAIN" \
    --wait --timeout 15m

wait_deployment_ready "cvat-backend-server" "nereidforge"

log_ok "CVAT deployment complete"
log_info "Access: https://cvat.$BASE_DOMAIN"
