#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# pull-images.sh - Pre-pull every categorized image from its public registry
# into the local runtime cache. This is pull-only: internal-registry
# publication belongs exclusively to NereidOverview harbor-sync.sh (root
# treaty section 18), so this repository never logs into or pushes to a
# private registry.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/config.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/images.sh"

log_info "Pulling all category images from their public registries..."

pull_all_categories

log_ok "All category images pulled into the local runtime cache"
