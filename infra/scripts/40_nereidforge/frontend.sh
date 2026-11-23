#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# frontend.sh - Deploy nereidforge-frontend

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LIB_DIR="$REPO_ROOT/infra/scripts/lib"
APP_CHARTS_DIR="$REPO_ROOT/charts"
HELM="helm"

source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

TENANT="${1:-nereidforge}"

# Single-domain installs may set FRONTEND_HOST explicitly (for example,
# forge.example.com); otherwise derive <tenant>.<base domain> after validation.
if [[ -z "${FRONTEND_HOST}" ]]; then
    validate_required BASE_DOMAIN || exit 1
    FRONTEND_HOST="$TENANT.$BASE_DOMAIN"
fi
validate_required FRONTEND_HOST || exit 1
export FRONTEND_HOST

log_info "========== Deploy NereidForge Frontend ($TENANT) =========="
log_info "Frontend host: $FRONTEND_HOST"

# Namespace (istio-injection) and the optional registry pull secret are
# rendered by the chart (templates/namespace.yaml, registry-secret.yaml). The
# secret renders only when registry.host, registry.username and
# registry.password are all supplied; default public installs render no secret.

# NF_FRONTEND_IMAGE is the public flat GHCR ref from lib/config.sh.
NF_FRONTEND_IMAGE="${NF_FRONTEND_IMAGE:?config.sh failed to provide NF_FRONTEND_IMAGE}"

# Parse NF_FRONTEND_IMAGE into repository and tag
NF_FRONTEND_REPO="${NF_FRONTEND_IMAGE%:*}"
NF_FRONTEND_TAG="${NF_FRONTEND_IMAGE##*:}"

log_info "Image repo: $NF_FRONTEND_REPO"
log_info "Image tag: $NF_FRONTEND_TAG"

cd "$APP_CHARTS_DIR/nereidforge-frontend"

$HELM upgrade --install nereidforge-frontend . -n "$TENANT" --create-namespace \
    --set image.repository="$NF_FRONTEND_REPO" \
    --set image.tag="$NF_FRONTEND_TAG" \
    --set "ingress.host=$FRONTEND_HOST" \
    --set "ingress.paths[0].path=/" \
    --set "ingress.paths[0].serviceName=nereidforge-frontend" \
    --set "ingress.paths[0].servicePort=80" \
    --set "ingress.paths[1].path=/api" \
    --set "ingress.paths[1].serviceName=nereidforge-api" \
    --set "ingress.paths[1].servicePort=8080" \
    --set "ingress.paths[2].path=/auth/ns-callback" \
    --set "ingress.paths[2].serviceName=nereidforge-api" \
    --set "ingress.paths[2].servicePort=8080" \
    --set "ingress.paths[3].path=/v1/integration/ns" \
    --set "ingress.paths[3].serviceName=nereidforge-api" \
    --set "ingress.paths[3].servicePort=8080" \
    --set 'ingress.tls[0].secretName=nereidforge-frontend-tls' \
    --set "ingress.tls[0].hosts[0]=$FRONTEND_HOST" \
    --set "env[1].value=https://$FRONTEND_HOST/api" \
    --set "env[2].value=wss://$FRONTEND_HOST" \
    --wait --timeout 10m

log_ok "Frontend deployment complete"
