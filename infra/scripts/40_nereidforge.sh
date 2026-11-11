#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# 40_nereidforge.sh - NF application-layer deployment entry point.

set -euo pipefail

# Resolve the repository root (set through the environment or a fixed path during deployment).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/infra/scripts/40_nereidforge"
LIB_DIR="$REPO_ROOT/infra/scripts/lib"

source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

init_deploy_log "40_nereidforge"

TENANT="${1:-nereidforge}"

log_info "========== Deploying the NereidForge application layer ($TENANT) =========="

log_info "Step 0/4: ML Backend (YOLO pre-annotation, app layer)"
"$SCRIPT_DIR/ml-backend.sh"

log_info "Step 1/4: Backend (API + Worker)"
"$SCRIPT_DIR/backend.sh" "$TENANT"

log_info "Step 2/4: Frontend"
"$SCRIPT_DIR/frontend.sh" "$TENANT"

log_info "Step 3/4: Ingress TLS"
"$SCRIPT_DIR/ingress-tls.sh" "$TENANT"

log_ok "========== NereidForge application-layer deployment completed =========="
FRONTEND_HOST="${FRONTEND_HOST:-$TENANT.$BASE_DOMAIN}"
log_info "Frontend: https://$FRONTEND_HOST"
log_info "API (same origin): https://$FRONTEND_HOST/api"
log_info "Model Registry: https://mr-$TENANT.$BASE_DOMAIN"

# ---- Operator hand-off: credentials + deploy logs -------------------------
# Every stage logs below DEPLOY_LOG_DIR; generated credentials live below
# NF_HANDOFF_DIR. Both locations are outside Git. Print an explicit save
# reminder so a fresh CLI install never leaves the operator without the admin
# secret (2026-10-29 requirement: no weak built-in admin passwords).
CRED_FILE="$NF_HANDOFF_DIR/admin-credentials.env"
echo ""
log_warn "================ Save the following deployment information ================"
if [[ -f "$CRED_FILE" ]]; then
    echo "  Platform administrator: admin"
    echo "  Platform administrator password: see $CRED_FILE (mode 600; do not commit to Git)"
else
    echo "  Platform administrator: admin (password not generated in this run; see $CRED_FILE or the rotated credentials)"
fi
echo "  Kubeflow Dex identity and password: see DEX_STATIC_EMAIL and DEX_STATIC_PASSWORD in $NF_ENV_FILE"
echo "  LS/CVAT UI administrators: $NF_HANDOFF_DIR/service-admin-credentials.env (mode 600)"
echo "  LS/CVAT static tokens: $NF_HANDOFF_DIR/service-tokens.env"
echo "  Deployment log directory: $DEPLOY_LOG_DIR (latest-<stage>.log)"
log_warn "================================================================"
