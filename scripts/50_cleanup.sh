#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# 50_cleanup.sh - Remove the legacy demo tenant (kubeflow-user-example-com).
#
# Since 2026-08-29, 30_install-kubeflow.sh no longer deploys the default demo
# tenant. ensure_tenant_profile creates the primary nereidforge tenant
# (KF Profile), with the Model Registry stack deployed inside it. This script
# only removes the legacy kubeflow-user-example-com tenant left by older
# versions, deleting both its Profile and Namespace, including any remaining
# Model Registry stack.
#
# Note: Do not touch the Dex static user (admin@nereidai.local); it remains
# the Kubeflow platform login identity. To reset its password, see
# DEX_STATIC_PASSWORD_HASH in lib/config.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

PROFILE_NS="kubeflow-user-example-com"

log_info "========== Removing the default demo tenant =========="

log_info "Deleting Profile: $PROFILE_NS"
kubectl delete profile "$PROFILE_NS" --ignore-not-found=true

log_info "Deleting Namespace: $PROFILE_NS (including its Model Registry stack)"
kubectl delete namespace "$PROFILE_NS" --ignore-not-found=true

log_info "Waiting for the Namespace to be fully removed..."
if kubectl wait --for=delete namespace/"$PROFILE_NS" --timeout=120s; then
    log_ok "Default demo tenant removed"
else
    log_warn "Namespace removal timed out; manually confirm that $PROFILE_NS was deleted"
fi
