#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# extract-charts.sh - Extract the offline chart bundle (infra install + app install).

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/config.sh"

OFFLINE_CHARTS="${1:-$OFFLINE_DIR/charts/charts-offline.tar.gz}"
TARGET_INFRA="${2:-$CHARTS_DIR}"
TARGET_APP="${3:-$APP_CHARTS_DIR}"
STAGE="$OFFLINE_DIR/.charts-load-stage"

[[ -f "$OFFLINE_CHARTS" ]] || { log_err "Offline chart bundle not found: $OFFLINE_CHARTS"; exit 1; }

log_info "Extracting charts: $OFFLINE_CHARTS"
rm -rf "$STAGE"
mkdir -p "$STAGE"
tar -xzf "$OFFLINE_CHARTS" -C "$STAGE"

# Match the repository-relative archive layout.
log_info "Extracting pre-nereidforge charts -> $TARGET_INFRA"
mkdir -p "$TARGET_INFRA"
cp -a "$STAGE/deploy/pre-nereidforge/charts"/. "$TARGET_INFRA/"

log_info "Extracting app charts -> $TARGET_APP"
mkdir -p "$TARGET_APP"
cp -a "$STAGE/charts"/. "$TARGET_APP/"

rm -rf "$STAGE"
log_ok "Charts extracted"
