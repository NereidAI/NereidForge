#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Delegate offline chart preparation to the canonical nine-chart packager. Its
# staging area owns dependency downloads, product renames, and release-version
# updates (tag-sourced CHART_VERSION, root treaty §19.5); the offline-tree
# archive layout preserves the loader's deploy/pre-nereidforge/charts contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
source "$SCRIPT_DIR/../lib/config.sh"

mkdir -p "$OFFLINE_DIR/charts"
CHARTS_OUTPUT_DIR="$OFFLINE_DIR/charts/packages" \
CHARTS_ARCHIVE="$OFFLINE_DIR/charts/charts-offline.tar.gz" \
CHARTS_ARCHIVE_LAYOUT=offline-tree \
HELM="${HELM:-helm}" \
    bash "$SCRIPT_DIR/../03_package-charts.sh"

log_ok "Offline chart archive: $OFFLINE_DIR/charts/charts-offline.tar.gz"
