#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# load-images.sh - Load the offline image bundle.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/config.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/images.sh"

OFFLINE_IMAGES_DIR="${1:-$OFFLINE_DIR/images}"

[[ -d "$OFFLINE_IMAGES_DIR" ]] || { log_err "Offline image directory not found: $OFFLINE_IMAGES_DIR"; exit 1; }

log_info "Loading offline images: $OFFLINE_IMAGES_DIR"

# Check for docker/containerd.
if command -v ctr &>/dev/null; then
    log_info "containerd detected; importing with ctr"
    for file in "$OFFLINE_IMAGES_DIR"/images-*.tar.gz; do
        [[ -f "$file" ]] || continue
        log_info "Importing: $(basename $file)"
        gunzip -c "$file" | ctr -n k8s.io images import -
    done
else
    log_info "Using docker load"
    load_all_categories "$OFFLINE_IMAGES_DIR"
fi

log_ok "All images loaded"
