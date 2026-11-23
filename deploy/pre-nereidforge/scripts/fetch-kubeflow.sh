#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# fetch-kubeflow.sh - Build the Kubeflow tree for step 4 of offline packaging.
# Rebuild upstream 26.03.1 plus NF patches, verify version-sensitive output,
# then generate a self-contained offline copy without vendoring release CRDs
# into the source repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PRE_ROOT/../../scripts/lib/common.sh"
source "$PRE_ROOT/../../scripts/lib/config.sh"

MANIFESTS_DIR="$PRE_ROOT/platform/kubeflow-manifests"

log_info "Rebuilding kubeflow-manifests (26.03.1 + nf-patches)..."
bash "$SCRIPT_DIR/04_fetch-kubeflow-manifests.sh"

# Verify required output directories after the rebuild.
REQUIRED=(
  "common/cert-manager/base"
  "common/dex/overlays/istio"
  "applications/pipeline/upstream/env/platform-agnostic-multi-user"
  "applications/kserve/kserve"
  "applications/trainer/overlays"
  "applications/workspaces/overlays/istio"
  "third-party/argo"
)
for d in "${REQUIRED[@]}"; do
  if [[ ! -d "$MANIFESTS_DIR/$d" ]]; then
    log_err "Missing after rebuild: $d"
    exit 1
  fi
done

# Preserve the NF KFP 2.17.0 override instead of drifting to upstream 2.16.x.
KFP_PIPELINE_KUSTOMIZATION="$MANIFESTS_DIR/applications/pipeline/upstream/base/pipeline/kustomization.yaml"
if ! grep -A1 -- '- name: ghcr.io/kubeflow/kfp-api-server' "$KFP_PIPELINE_KUSTOMIZATION" | grep -q -- 'newTag: 2.17.0'; then
  log_err "Upstream KFP manifest does not pin kfp-api-server to 2.17.0; rebuild is invalid (see platform/kubeflow-patches/README.md)"
  exit 1
fi

log_ok "Kubeflow manifests rebuilt: $MANIFESTS_DIR"

# Generate the offline copy. Large upstream release assets belong only in
# generated render/offline trees; Git keeps upstream refs and NF patches.
mkdir -p "$OFFLINE_DIR/manifests"
rm -rf "$OFFLINE_DIR/manifests/kubeflow-manifests"
cp -r "$MANIFESTS_DIR" "$OFFLINE_DIR/manifests/"
"$SCRIPT_DIR/cache-kubeflow-upstream.sh" "$OFFLINE_DIR/manifests/kubeflow-manifests"
log_ok "Offline package manifests: $OFFLINE_DIR/manifests/kubeflow-manifests"
