#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Preserve the legacy full-stack offline compatibility bundle by composing the
# independent four-chart NF and five-chart pre-NereidForge offline-tree archives.
# Neither module's local packaging interface owns the other module's charts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
source "$SCRIPT_DIR/../lib/config.sh"

mkdir -p "$OFFLINE_DIR/charts"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

CHARTS_OUTPUT_DIR="$OFFLINE_DIR/charts/packages/nf" \
CHARTS_ARCHIVE="$STAGING/nf-charts.tar.gz" \
CHARTS_ARCHIVE_LAYOUT=offline-tree \
HELM="${HELM:-helm}" \
    bash "$SCRIPT_DIR/../03_package-charts.sh"

PRE_CHARTS_OUTPUT_DIR="$OFFLINE_DIR/charts/packages/pre" \
PRE_CHARTS_ARCHIVE="$STAGING/pre-charts.tar.gz" \
PRE_CHARTS_ARCHIVE_LAYOUT=offline-tree \
HELM="${HELM:-helm}" \
    bash "$ROOT_DIR/deploy/pre-nereidforge/scripts/package-charts.sh"

mkdir -p "$STAGING/combined"
tar -xzf "$STAGING/nf-charts.tar.gz" -C "$STAGING/combined"
tar -xzf "$STAGING/pre-charts.tar.gz" -C "$STAGING/combined"
tar -czf "$OFFLINE_DIR/charts/charts-offline.tar.gz" \
    -C "$STAGING/combined" charts deploy/pre-nereidforge/charts

log_ok "Offline chart archive: $OFFLINE_DIR/charts/charts-offline.tar.gz"
