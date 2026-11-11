#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# 00_deploy-all.sh orchestrates the complete NereidForge deployment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/config.sh"

log_info "========== NereidForge full-stack deployment =========="
log_info "Project root: $ROOT_DIR"

# Interactive fallback for values not supplied via the environment.
if [[ -z "${BASE_DOMAIN:-}" ]]; then
    interactive_config
fi

# Validate required config
validate_required BASE_DOMAIN NFS_SERVER NFS_SHARE PG_ADMIN_PASSWORD JWT_SECRET \
    SEAWEEDFS_S3_ACCESS_KEY SEAWEEDFS_S3_SECRET_KEY \
    DEX_STATIC_EMAIL DEX_STATIC_PASSWORD DEX_STATIC_PASSWORD_HASH || {
    log_err "Required configuration is missing; please check"
    exit 1
}

export_all

# Confirm the deployment
echo ""
log_warn "About to deploy with the following configuration:"
echo "  Domain: $BASE_DOMAIN"
echo "  NFS: $NFS_SERVER:$NFS_SHARE"
echo "  PG password: ${PG_ADMIN_PASSWORD:0:4}****"
echo "  JWT: ${JWT_SECRET:0:4}****"
echo ""
read -rp "Start deployment? (y/N): " confirm
[[ "$confirm" != "y" && "$confirm" != "Y" ]] && { log_info "Cancelled"; exit 0; }

# ========== Deployment sequence ==========

# Step 0 loads an offline bundle when one is present.
if [[ -d "$OFFLINE_DIR" ]]; then
    log_info "Step 0/5: load the offline bundle"
    "$SCRIPT_DIR/02_load-offline.sh" "$OFFLINE_DIR"
fi

# Step 1 installs NF-owned infrastructure.
log_info "Step 1/5: infrastructure"
"$SCRIPT_DIR/10_infra.sh"

# Step 2 installs the standalone Kubeflow stack only on greenfield deployments.
log_info "Step 2/5: Kubeflow stack"
"$SCRIPT_DIR/30_install-kubeflow.sh"

# Step 3 installs the NF application layer for the default tenant.
log_info "Step 3/5: NF application layer (nereidforge)"
"$SCRIPT_DIR/40_nereidforge.sh" "nereidforge"

# Step 4 runs end-to-end verification.
log_info "Step 4/5: end-to-end verification"
"$SCRIPT_DIR/60_verify.sh"

log_ok "========== NereidForge full-stack deployment complete =========="
echo ""
log_info "Access URLs:"
echo "  NF Frontend: https://${FRONTEND_HOST:-nereidforge.$BASE_DOMAIN}"
echo "  NF API (same origin): https://${FRONTEND_HOST:-nereidforge.$BASE_DOMAIN}/api"
echo "  Model Registry UI: routed through kubeflow-gateway in the primary tenant namespace (nereidforge)"
echo "  Label Studio: https://labelstudio.$BASE_DOMAIN"
echo "  CVAT: https://cvat.$BASE_DOMAIN"
echo ""
log_info "Administrator credentials (mode 600; never commit them):"
echo "  NF platform: $NF_HANDOFF_DIR/admin-credentials.env"
echo "  LS/CVAT UI: $NF_HANDOFF_DIR/service-admin-credentials.env"
echo "  Kubeflow login identity: $DEX_STATIC_EMAIL (password in $NF_ENV_FILE)"
echo ""
log_info "Next steps:"
echo "  # Create a tenant (KF Profile reconciles ns/RBAC/S3 ConfigMap):"
echo "  kubectl exec -n nereidforge deploy/nereidforge-backend-api -- python -c \"from app.services.admin import provision_tenant_profile; provision_tenant_profile('new-tenant', 'admin@nereidforge.local')\""
echo "  # Or call the API: POST /api/v1/admin/tenants"
echo "  # Remove the historical demo tenant only from legacy deployments:"
echo "  ./50_cleanup.sh"
