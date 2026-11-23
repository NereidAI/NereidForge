#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# cache-kubeflow-upstream.sh - Materialize pinned upstream release assets only
# inside generated render/offline trees, leaving the source tree patch-only.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <generated-kubeflow-manifests-directory>" >&2
    exit 2
fi

MANIFESTS_DIR="$(cd "$1" && pwd)"

# cache_release downloads one official asset, verifies its identity, and points
# only the generated tree's kustomization at the cached copy. An exact source
# match is required so an upstream layout change fails instead of being ignored.
cache_release() {
    local component="$1"
    local source_resource="$2"
    local download_url="$3"
    local filename="$4"
    local expected_sha="$5"
    local component_dir="$MANIFESTS_DIR/applications/trainer/upstream/third-party/$component"
    local kustomization="$component_dir/kustomization.yaml"
    local destination="$component_dir/$filename"
    local download="$destination.download"

    [[ -f "$kustomization" ]] || {
        echo "Missing generated kustomization: $kustomization" >&2
        return 1
    }

    # An already-generated offline package needs no network. Reverify its cached
    # asset so repeated renders remain idempotent without trusting stale content.
    if grep -Fqx "  - $filename" "$kustomization"; then
        [[ -f "$destination" ]] || {
            echo "Missing cached upstream asset: $destination" >&2
            return 1
        }
        printf '%s  %s\n' "$expected_sha" "$destination" | sha256sum -c -
        return 0
    fi
    if ! grep -Fqx "  - $source_resource" "$kustomization"; then
        echo "Expected upstream resource in $kustomization: $source_resource" >&2
        return 1
    fi

    rm -f "$download"
    if ! curl -fsSL --retry 2 --connect-timeout 15 --max-time 60 "$download_url" -o "$download"; then
        rm -f "$download"
        return 1
    fi
    if ! printf '%s  %s\n' "$expected_sha" "$download" | sha256sum -c -; then
        rm -f "$download"
        return 1
    fi
    mv "$download" "$destination"

    python3 - "$kustomization" "$source_resource" "$filename" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source_resource = sys.argv[2]
filename = sys.argv[3]
text = path.read_text()
source_line = f"  - {source_resource}\n"
if text.count(source_line) != 1:
    raise SystemExit(f"Expected exactly one upstream resource in {path}: {source_resource}")
path.write_text(text.replace(source_line, f"  - {filename}\n", 1))
PY
}

cache_release \
    jobset \
    https://github.com/kubernetes-sigs/jobset/releases/download/v0.12.0/manifests.yaml \
    https://github.com/kubernetes-sigs/jobset/releases/download/v0.12.0/manifests.yaml \
    manifests-v0.12.0.yaml \
    a41aaf12dd0b7b0a3d626b8d6107f32c3dc889d7e7830adc46b8cdb77a8963bb

cache_release \
    leaderworkerset \
    https://github.com/kubernetes-sigs/lws/releases/download/v0.8.0/manifests.yaml \
    https://github.com/kubernetes-sigs/lws/releases/download/v0.8.0/manifests.yaml \
    manifests-v0.8.0.yaml \
    66416a375610ce31be69818f7ca02ed89887ce76e1034ad23addbc39e9c6854a

echo "Pinned JobSet/LWS assets verified in generated render/offline tree only."
