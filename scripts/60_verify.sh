#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# 60_verify.sh - End-to-end verification entry point.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/config.sh"

init_deploy_log "60_verify"

log_info "========== End-to-end verification =========="

log_info "Step 1/5: Verifying infrastructure"
"$ROOT_DIR/deploy/pre-nereidforge/scripts/60_verify/infra-check.sh"

log_info "Step 2/5: Verifying Kubeflow"
"$ROOT_DIR/deploy/pre-nereidforge/scripts/60_verify/kubeflow-check.sh"

log_info "Step 3/5: Verifying Model Registry"
"$ROOT_DIR/deploy/pre-nereidforge/scripts/60_verify/mr-check.sh"

log_info "Step 4/5: Verifying the NF application layer"
"$SCRIPT_DIR/60_verify/nf-check.sh" "nereidforge"

log_info "Step 5/5: Verifying tenant provisioning"
"$SCRIPT_DIR/60_verify/tenant-check.sh"

log_ok "========== All checks passed =========="
