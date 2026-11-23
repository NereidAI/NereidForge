#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# 01_prepare-offline.sh - Air-gap bundle build entry point. All image
# references stay public (root treaty section 18); the bundle carries tarballs
# under their public names and target nodes resolve them locally.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/config.sh"
source "$SCRIPT_DIR/lib/images.sh"

log_info "========== Build NereidForge air-gap bundle =========="
log_info "Output directory: $OFFLINE_DIR"

mkdir -p "$OFFLINE_DIR"/{images,charts,manifests,scripts}

# Step 1: pre-pull the public image references into the local cache.
log_info "Step 1/4: Pull public image references"
"$SCRIPT_DIR/01_prepare-offline/pull-images.sh"

# Step 2: export the images to tar.gz archives under their public names.
log_info "Step 2/4: Export images to tar.gz"
export_all_categories "$OFFLINE_DIR/images"
generate_manifest "$OFFLINE_DIR/images-manifest.txt"

# Step 3: package the charts
log_info "Step 3/4: Package Helm charts"
"$SCRIPT_DIR/01_prepare-offline/package-charts.sh"

# Step 4: vendor the kubeflow manifests
log_info "Step 4/4: Vendor Kubeflow manifests"
"$ROOT_DIR/deploy/pre-nereidforge/scripts/fetch-kubeflow.sh"

# Copy the combined deployment tooling and pre-nereidforge tree into the bundle.
log_info "Copying deployment scripts..."
cp -r "$ROOT_DIR/scripts" "$OFFLINE_DIR/"
mkdir -p "$OFFLINE_DIR/deploy"
cp -r "$ROOT_DIR/deploy/pre-nereidforge" "$OFFLINE_DIR/deploy/"
cp "$ROOT_DIR/values-prod.yaml" "$OFFLINE_DIR/"
cp "$ROOT_DIR/secrets.yaml.template" "$OFFLINE_DIR/"

# Generate the bundle metadata
cat > "$OFFLINE_DIR/VERSION" <<EOF
NereidForge Offline Package
Version: 1.0.0
Generated: $(date)
Kubeflow: v26.03.1
CNPC: $CNPC_VERSION
PostgreSQL: $POSTGRES_VERSION
SeaweedFS: $SEAWEEDFS_VERSION
EOF

log_ok "========== Air-gap bundle complete =========="
log_info "Directory: $OFFLINE_DIR"
log_info "Contents:"
ls -lh "$OFFLINE_DIR"
echo ""
log_info "Distribution:"
echo "  scp -r $OFFLINE_DIR user@target:/tmp/"
echo "  # On the target machine: cd /tmp/offline-package && ./scripts/00_deploy-all.sh"
