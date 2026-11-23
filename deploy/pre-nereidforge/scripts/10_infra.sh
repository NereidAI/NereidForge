#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# 10_infra.sh - Infrastructure deployment entry (Helm charts)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB_DIR="$PRE_ROOT/scripts/lib"

source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

init_deploy_log "10_infra"

log_info "========== Deploy Infrastructure (Helm) =========="

# 0. Namespace (blank cluster starts here). Images use public coordinates and
#    pull anonymously; air-gapped nodes resolve them via the containerd
#    registry mirror, so no in-pod pull secret is created (root treaty section 18).
log_info "Step 0/8: Namespace"
ensure_ns "nereidforge"

# 1. NFS CSI (must be first, provides StorageClass). A pre-existing driver
#    (e.g. the cluster-wide nfs-csi in kube-system with the nfs-csi SC) is reused.
# StorageClass is a cluster prerequisite owned by the suite layer now
# (NereidOverview/deploy/prereqs): NF no longer installs csi-driver-nfs or
# cert-manager/GPU device plugins. It only verifies the SC exists.
: "${STORAGE_CLASS:=nfs-csi}"
if kubectl get storageclass "$STORAGE_CLASS" >/dev/null 2>&1; then
    log_ok "Step 1/8: StorageClass $STORAGE_CLASS present (cluster prerequisite)"
else
    log_err "StorageClass '$STORAGE_CLASS' not found. csi-driver-nfs (and cert-manager,"
    log_err "GPU device plugins) are cluster prerequisites now shipped outside NereidForge;"
    log_err "see NereidOverview/deploy/prereqs. Install it, then rerun."
    exit 1
fi

# 2. PostgreSQL
log_info "Step 2/8: PostgreSQL (CloudNativePG)"
"$SCRIPT_DIR/10_infra/postgresql.sh"

# 3. Pre-create databases (depends on CNPG ready; LS/CVAT/NF each use separate database)
log_info "Step 3/8: Business databases pre-creation"
"$SCRIPT_DIR/10_infra/bootstrap-db.sh"

# 4. Valkey (external cache for Label Studio / CVAT)
log_info "Step 4/8: Valkey"
"$SCRIPT_DIR/10_infra/valkey.sh"

# 5. SeaweedFS
log_info "Step 5/8: SeaweedFS"
"$SCRIPT_DIR/10_infra/seaweedfs.sh"

# 6. Label Studio
log_info "Step 6/8: Label Studio"
"$SCRIPT_DIR/10_infra/labelstudio.sh"

# 7. CVAT (official chart)
log_info "Step 7/8: CVAT"
"$SCRIPT_DIR/10_infra/cvat.sh"

# 8. LS/CVAT fixed DRF token bootstrap (for NF backend integration)
log_info "Step 7/8: LS/CVAT token bootstrap"
"$SCRIPT_DIR/10_infra/bootstrap-tokens.sh"

log_ok "========== Infrastructure deployment complete =========="
