#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# 10_load-images.sh
# Air-gapped environment: load categorized image archives into
# containerd/docker. Archives keep their public reference names (root treaty
# section 18); the node containerd mirror resolves the same refs at run time.
# Usage:
#   ./10_load-images.sh                    # load every category archive found
#   ./10_load-images.sh base kubeflow      # load only the listed categories
#   ./10_load-images.sh --list             # list available category archives

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Archives are produced by 05_export-images.sh and unpacked offline bundles.
# IMAGE_ARCHIVE_DIR supports a relocated bundle without editing this script.
IMAGE_ARCHIVE_DIR="${IMAGE_ARCHIVE_DIR:-$WORKDIR/images}"

echo "=== Load offline images ==="
echo "Archive directory: $IMAGE_ARCHIVE_DIR"
echo ""

# Select the runtime: prefer containerd (k3s), fall back to docker.
if command -v ctr >/dev/null 2>&1 && ctr version >/dev/null 2>&1; then
    RUNTIME="containerd"
    echo "Runtime: containerd (k3s)"
elif command -v docker >/dev/null 2>&1 && docker version >/dev/null 2>&1; then
    RUNTIME="docker"
    echo "Runtime: docker"
else
    echo "Error: neither containerd nor docker found"
    exit 1
fi

# list_available prints the category archives in IMAGE_ARCHIVE_DIR without
# contacting the runtime or failing when the directory has no matching archive.
list_available() {
    echo "Available category archives:"
    for f in "$IMAGE_ARCHIVE_DIR"/images-*.tar.gz; do
        [[ -f "$f" ]] || continue
        cat=$(basename "$f" .tar.gz | sed 's/^images-//')
        size=$(du -h "$f" | cut -f1)
        echo "  $cat ($size)"
    done
}

# load_category imports one named archive into the selected local runtime and
# returns 1 before import when that category archive is absent.
load_category() {
    local cat="$1"
    local archive="$IMAGE_ARCHIVE_DIR/images-${cat}.tar.gz"

    if [[ ! -f "$archive" ]]; then
        echo "Error: category archive not found: $archive"
        list_available
        return 1
    fi

    echo "=== Loading category: $cat ==="
    echo "File: $archive ($(du -h "$archive" | cut -f1))"

    if [[ "$RUNTIME" == "containerd" ]]; then
        gunzip -c "$archive" | ctr -n k8s.io images import -
    else
        docker load -i "$archive"
    fi

    echo "Done: $cat"
    echo ""
}

# load_all imports every category archive in lexical filename order and stops
# on the first runtime import failure through the caller's errexit setting.
load_all() {
    for f in "$IMAGE_ARCHIVE_DIR"/images-*.tar.gz; do
        [[ -f "$f" ]] || continue
        cat=$(basename "$f" .tar.gz | sed 's/^images-//')
        load_category "$cat"
    done
}

# ========== Main logic ==========
if [[ "${1:-}" == "--list" ]]; then
    list_available
    exit 0
fi

if [[ $# -eq 0 ]]; then
    load_all
else
    for cat in "$@"; do
        load_category "$cat"
    done
fi

echo ""
echo "=== Verification ==="
if [[ "$RUNTIME" == "containerd" ]]; then
    echo "Images present in the k8s.io namespace: $(ctr -n k8s.io images ls -q | wc -l)"
else
    docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"
fi
