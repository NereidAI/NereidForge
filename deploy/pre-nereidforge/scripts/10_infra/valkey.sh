#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# valkey.sh deploys the Valkey Helm chart as an external Label Studio/CVAT cache.
# The NF backend does not consume this service; CAPTCHA and OIDC state remain in
# the PostgreSQL kv_store table.

set -euo pipefail

PRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB_DIR="$PRE_ROOT/../../scripts/lib"
source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

CHARTS_DIR="$PRE_ROOT/charts"

log_info "========== Deploy Valkey (Helm chart) =========="

cd "$CHARTS_DIR/valkey"

# Image ref is the pristine upstream coordinate (lib/config.sh); air-gapped
# nodes resolve it through the containerd registry mirror, no pull secret.
docker_ref_parts "$IMG_VALKEY"
VALKEY_ARGS=(--set image.registry="${_REF_REGISTRY:-docker.io}"
             --set image.repository="$_REF_REPOSITORY"
             --set image.tag="$_REF_TAG")

helm upgrade --install valkey . -n nereidforge --create-namespace \
    "${VALKEY_ARGS[@]}" \
    --wait --timeout 10m

wait_statefulset_ready "valkey-master" "nereidforge"
wait_statefulset_ready "valkey-replica" "nereidforge"

# Verify connectivity with the valkey-cli included in the Valkey image.
log_info "Verify the Valkey connection..."
kubectl run -n nereidforge valkey-test-$$ --rm -i --restart=Never \
    --image=$IMG_VALKEY -- \
    valkey-cli -h valkey-master ping

log_ok "Valkey deployment complete"
log_info "Connection information:"
echo "  Master:   valkey-master.nereidforge.svc.cluster.local:6379 (ondisk port 6666)"
echo "  Replica:  valkey-replica.nereidforge.svc.cluster.local:6379"
