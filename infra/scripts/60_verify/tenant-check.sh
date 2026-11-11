#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# tenant-check.sh - Tenant provisioning verification

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/config.sh"

HELM="helm"

log_info "========== Verify Tenant Provisioning =========="

# The main business tenant uses stage-30 static KFP wiring because the upstream
# pipeline-profile hook cannot mint IAM credentials on SeaweedFS (pit 126).
$KUBECTL get configmap artifact-repositories kfp-launcher metadata-grpc-configmap \
    -n nereidforge >/dev/null
log_ok "main business tenant KFP ConfigMaps exist"

# A raw Profile verifies upstream namespace/RBAC reconciliation only; it must
# not expect the unsupported SeaweedFS pipeline-profile hook to add KFP config.
TEST_TENANT="test-tenant-$$"

cleanup() {
    # The Profile controller garbage-collects owned resources; explicit namespace
    # deletion is only a safety net after a partially completed verification.
    log_info "Cleaning up test tenant..."
    "$HELM" uninstall "$TEST_TENANT" -n nereidforge --ignore-not-found 2>/dev/null || true
    $KUBECTL delete ns "$TEST_TENANT" --ignore-not-found=true
}
trap cleanup EXIT

wait_for() {
    local description="$1"
    shift
    for ((attempt = 1; attempt <= 30; attempt++)); do
        if "$@" >/dev/null 2>&1; then
            log_ok "$description"
            return 0
        fi
        sleep 2
    done
    log_err "$description did not become ready within 60s"
    return 1
}

namespace_label_equals() {
    local jsonpath="$1" expected="$2" actual
    actual=$($KUBECTL get ns "$TEST_TENANT" -o "jsonpath=$jsonpath" 2>/dev/null) || return 1
    [[ "$actual" == "$expected" ]]
}

log_info "Creating test tenant via Profile: $TEST_TENANT"
"$HELM" upgrade --install "$TEST_TENANT" "$APP_CHARTS_DIR/tenant-check" \
    -n nereidforge \
    --set "testTenant=$TEST_TENANT" \
    --set "owner.name=test@example.com"

# Profile, RBAC and KFP resources are reconciled asynchronously. A missing
# resource after the wait budget is a verification failure, not a warning.
log_info "Waiting for namespace, Profile and reconciliation products..."
wait_for "namespace/$TEST_TENANT exists" \
    $KUBECTL get ns "$TEST_TENANT"
wait_for "profile/$TEST_TENANT exists" \
    $KUBECTL get profile "$TEST_TENANT"
wait_for "rolebinding/namespaceAdmin exists" \
    $KUBECTL get rolebinding namespaceAdmin -n "$TEST_TENANT"
wait_for "rolebinding/default-editor exists" \
    $KUBECTL get rolebinding default-editor -n "$TEST_TENANT"
wait_for "rolebinding/default-viewer exists" \
    $KUBECTL get rolebinding default-viewer -n "$TEST_TENANT"
wait_for "namespace PSS is baseline" \
    namespace_label_equals '{.metadata.labels.pod-security\.kubernetes\.io/enforce}' baseline
wait_for "namespace has upstream istio-injection=enabled" \
    namespace_label_equals '{.metadata.labels.istio-injection}' enabled

# This raw Profile test verifies upstream controller reconciliation. It differs
# intentionally from the pre-created main NF business namespace, which opts out
# of namespace-wide injection and is promoted to privileged PSS by stage 30.

trap - EXIT
cleanup
log_ok "Tenant Provisioning verification done"
