#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# 02_load-offline.sh - Load the offline package on the target environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/config.sh"

OFFLINE_PKG="${1:-$OFFLINE_DIR}"

[[ -d "$OFFLINE_PKG" ]] || { log_err "Offline package directory not found: $OFFLINE_PKG"; exit 1; }

log_info "========== Loading offline package: $OFFLINE_PKG =========="

# 1. Load images
log_info "Step 1/3: Loading images"
"$SCRIPT_DIR/02_load-offline/load-images.sh" "$OFFLINE_PKG/images"

# 2. Extract charts
log_info "Step 2/3: Extracting charts"
"$SCRIPT_DIR/02_load-offline/extract-charts.sh" "$OFFLINE_PKG/charts/charts-offline.tar.gz"

# 3. Prepare secrets
log_info "Step 3/3: Preparing secrets"
"$SCRIPT_DIR/02_load-offline/prepare-secrets.sh"

log_ok "========== Offline package loaded =========="
