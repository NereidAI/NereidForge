#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# 05_export-images.sh
# Export images to categorized tarballs (layer deduplication) under their
# public reference names. All coordinates are public (root treaty section 18);
# air-gapped nodes import the archives and resolve the unchanged refs through
# the containerd registry mirror.
#
# Usage:
#   ./05_export-images.sh                    # export all categories
#   ./05_export-images.sh base kubeflow      # export only listed categories
#   ./05_export-images.sh --list             # list all categories
#   ./05_export-images.sh --manifest [path]  # regenerate the public manifest
#
# The category ground truth lives in lib/images.sh. Every image that NF
# references at runtime must appear there or the exported bundle is incomplete.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/config.sh"
source "$SCRIPT_DIR/lib/images.sh"

WORKDIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$WORKDIR/images"

if [[ "${1:-}" == "--list" ]]; then
    list_categories
    exit 0
fi

# Manifest generation is local file I/O only; unlike export it never pulls or
# saves an image. The default refreshes this repository's delivery input.
if [[ "${1:-}" == "--manifest" ]]; then
    generate_manifest "${2:-$ROOT_DIR/images-manifest.txt}"
    exit 0
fi

if [[ $# -eq 0 ]]; then
    export_all_categories "$OUTPUT_DIR"
else
    for cat in "$@"; do
        export_category "$cat" "$OUTPUT_DIR"
    done
fi
