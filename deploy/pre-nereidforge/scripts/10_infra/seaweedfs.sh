#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# seaweedfs.sh - Deploy SeaweedFS (Helm chart).

set -euo pipefail

PRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB_DIR="$PRE_ROOT/scripts/lib"
source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

CHARTS_DIR="$PRE_ROOT/charts"

log_info "========== Deploying SeaweedFS (Helm) =========="

: "${SEAWEEDFS_MASTER_STORAGE:=10Gi}"
: "${SEAWEEDFS_VOLUME_STORAGE:=200Gi}"
# The access key has no interactive fallback because deployment normally runs
# unattended. A missing secret may still be entered through the prompt below.
validate_required SEAWEEDFS_S3_ACCESS_KEY || exit 1

read_password() {
    local prompt="$1"
    local var_name="$2"
    local value
    if [[ -n "${!var_name:-}" ]]; then
        value="${!var_name}"
        log_info "$prompt: ******** (from environment variable)"
    else
        read -rsp "$prompt: " value
        echo
    fi
    export "$var_name"="$value"
}

read_password "S3 Secret Key" SEAWEEDFS_S3_SECRET_KEY
validate_required SEAWEEDFS_S3_ACCESS_KEY SEAWEEDFS_S3_SECRET_KEY || exit 1

cd "$CHARTS_DIR/seaweedfs"
ls charts/*.tgz >/dev/null 2>&1 || helm dependency build

helm upgrade --install seaweedfs . -n nereidforge --create-namespace \
    --set image="$IMG_SEAWEEDFS" \
    --set storageClass="$STORAGE_CLASS" \
    --set master.persistence.size="$SEAWEEDFS_MASTER_STORAGE" \
    --set volume.persistence.size="$SEAWEEDFS_VOLUME_STORAGE" \
    --set s3.enabled=true \
    --set s3.accessKey="$SEAWEEDFS_S3_ACCESS_KEY" \
    --set s3.secretKey="$SEAWEEDFS_S3_SECRET_KEY" \
    --wait --timeout 10m

wait_deployment_ready "seaweedfs-master" "nereidforge"
wait_statefulset_ready "seaweedfs-volume" "nereidforge"
wait_deployment_ready "seaweedfs-filer" "nereidforge"
wait_deployment_ready "seaweedfs-s3" "nereidforge"

# Verify each endpoint.
log_info "Verifying SeaweedFS endpoints..."
sleep 10
kubectl run -n nereidforge sw-test-$$ --rm -i --restart=Never \
    --image=$IMG_BUSYBOX -- \
    wget -q -O - http://seaweedfs-master.nereidforge:9333/ | grep -q "SeaweedFS" && log_ok "Master UI is healthy"

kubectl run -n nereidforge sw-test2-$$ --rm -i --restart=Never \
    --image=$IMG_BUSYBOX -- \
    wget -q -O - http://seaweedfs-filer.nereidforge:8888/ | grep -q "SeaweedFS Filer" && log_ok "Filer UI is healthy"

kubectl run -n nereidforge sw-test3-$$ --rm -i --restart=Never \
    --image=$IMG_BUSYBOX -- \
    wget -q -O - http://seaweedfs-s3.nereidforge:8333/ | grep -q "ListAllMyBucketsResult" && log_ok "S3 API is healthy"

log_ok "SeaweedFS deployment completed"
log_info "Connection information:"
echo "  Master: seaweedfs-master.nereidforge.svc.cluster.local:9333"
echo "  Volume: seaweedfs-volume.nereidforge.svc.cluster.local:8080"
echo "  Filer UI: http://seaweedfs-filer.nereidforge.svc.cluster.local:8888"
echo "  Filer gRPC: seaweedfs-filer.nereidforge.svc.cluster.local:18888"
echo "  S3 API: http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"
echo "  S3 credentials: supplied through the site configuration (not logged)"
