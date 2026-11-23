#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# infra-check.sh - Verify infrastructure.

set -euo pipefail

PRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PRE_ROOT/../../scripts/lib/common.sh"
source "$PRE_ROOT/../../scripts/lib/config.sh"

log_info "========== Verifying infrastructure =========="

# 1. StorageClass (cluster prerequisite since 2026-10-19: csi-driver-nfs /
#    cert-manager / GPU device plugins ship via NereidOverview/deploy/prereqs;
#    NF only verifies the contract, not the driver pods)
: "${STORAGE_CLASS:=nfs-csi}"
log_info "Checking StorageClass $STORAGE_CLASS (shared cluster prerequisite)..."
$KUBECTL get storageclass "$STORAGE_CLASS"

# 2. PostgreSQL
log_info "Checking PostgreSQL..."
$KUBECTL get cluster.postgresql.cnpg.io "nereidforge-postgresql" -n nereidforge
$KUBECTL wait --for=condition=Ready cluster.postgresql.cnpg.io/nereidforge-postgresql -n nereidforge --timeout=60s

# 3. Valkey (external cache for Label Studio / CVAT)
log_info "Checking Valkey..."
wait_statefulset_ready "valkey-master" "nereidforge"
wait_statefulset_ready "valkey-replica" "nereidforge"
$KUBECTL run -n nereidforge valkey-verify-$$ --rm -i --quiet --restart=Never \
    --image=$IMG_VALKEY -- \
    valkey-cli -h valkey-master.nereidforge ping

# 4. SeaweedFS
log_info "Checking SeaweedFS..."
wait_deployment_ready "seaweedfs-master" "nereidforge"
wait_statefulset_ready "seaweedfs-volume" "nereidforge"
wait_deployment_ready "seaweedfs-filer" "nereidforge"
wait_deployment_ready "seaweedfs-s3" "nereidforge"
$KUBECTL run -n nereidforge sw-verify-$$ --rm -i --quiet --restart=Never \
    --image=$IMG_BUSYBOX -- \
    sh -c "wget -q -O - http://seaweedfs-filer.nereidforge:8888/ | grep -q 'SeaweedFS Filer'" \
    || log_warn "SeaweedFS filer home page probe did not match"

# S3 credentials are always configured by 10_infra. Anonymous ListBuckets must
# therefore be rejected; accepting the old unauthenticated 200 response hid a
# broken security configuration and turned the expected 403 into a warning.
S3_PROBE_RC=0
if S3_PROBE_OUT=$($KUBECTL run -n nereidforge sw-verify2-$$ --rm -i --quiet --restart=Never \
    --image=$IMG_BUSYBOX -- \
    sh -c "wget -q -O - http://seaweedfs-s3.nereidforge:8333/" 2>&1); then
    S3_PROBE_RC=0
else
    S3_PROBE_RC=$?
fi
if grep -q "403 Forbidden" <<<"$S3_PROBE_OUT"; then
    log_ok "SeaweedFS S3 API reachable; anonymous ListBuckets rejected (403)"
else
    printf '%s\n' "$S3_PROBE_OUT"
    log_err "SeaweedFS S3 probe failed (exit $S3_PROBE_RC): expected anonymous 403"
    exit 1
fi

# 5. Label Studio
log_info "Checking Label Studio..."
wait_deployment_ready "labelstudio" "nereidforge"

# 6. CVAT
log_info "Checking CVAT..."
wait_deployment_ready "cvat-backend-server" "nereidforge"
wait_deployment_ready "cvat-frontend" "nereidforge"

log_ok "Infrastructure verification passed"
