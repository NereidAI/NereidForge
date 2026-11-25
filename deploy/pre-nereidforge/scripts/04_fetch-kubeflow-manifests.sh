#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# 04_fetch-kubeflow-manifests.sh
# Download the Kubeflow community-distribution manifests (tag 26.03.1), extract
# the required components, then apply platform/kubeflow-patches
# (patches + additions + removals) to produce the final business manifests
# byte-for-byte identical to the vendored directory.
# Version semantics: the patches explicitly restore business versions such as
# KFP 2.17.0 / cert-manager v1.21.1; see nf-patches/README.md.

set -euo pipefail

PRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFESTS_DIR="$PRE_ROOT/platform/kubeflow-manifests"
REPO_URL="https://github.com/kubeflow/community-distribution"
KUBEFLOW_VERSION="26.03.1"
PATCHES_DIR="$PRE_ROOT/platform/kubeflow-patches"

echo "=== Fetching Kubeflow manifests ==="
echo "Target directory: $MANIFESTS_DIR"
echo "Repository: $REPO_URL (tag $KUBEFLOW_VERSION)"
echo "Patch set: $PATCHES_DIR"
echo ""

rm -rf "$MANIFESTS_DIR"
mkdir -p "$MANIFESTS_DIR"

# Shallow clone
git clone --depth 1 --branch "$KUBEFLOW_VERSION" "$REPO_URL" "$MANIFESTS_DIR/tmp-repo"

SRC="$MANIFESTS_DIR/tmp-repo"

echo ">>> Extracting required components..."

# 1. common components (cert-manager, dex, istio, kubeflow-namespace, kubeflow-roles, oauth2-proxy, user-namespace, knative)
mkdir -p "$MANIFESTS_DIR/common"
for comp in cert-manager dex istio knative kubeflow-namespace kubeflow-roles oauth2-proxy user-namespace; do
    if [[ -d "$SRC/common/$comp" ]]; then
        echo "  Copying common/$comp"
        cp -r "$SRC/common/$comp" "$MANIFESTS_DIR/common/"
    else
        echo "  WARNING: common/$comp not found"
    fi
done

# 2. applications components
mkdir -p "$MANIFESTS_DIR/applications"

# pipeline (KFP)
if [[ -d "$SRC/applications/pipeline" ]]; then
    echo "  Copying applications/pipeline"
    cp -r "$SRC/applications/pipeline" "$MANIFESTS_DIR/applications/"
fi

# katib
if [[ -d "$SRC/applications/katib" ]]; then
    echo "  Copying applications/katib"
    cp -r "$SRC/applications/katib" "$MANIFESTS_DIR/applications/"
fi

# kserve
if [[ -d "$SRC/applications/kserve" ]]; then
    echo "  Copying applications/kserve"
    cp -r "$SRC/applications/kserve" "$MANIFESTS_DIR/applications/"
fi

# workspaces (formerly jupyter, Jupyter Web App)
if [[ -d "$SRC/applications/workspaces" ]]; then
    echo "  Copying applications/workspaces"
    cp -r "$SRC/applications/workspaces" "$MANIFESTS_DIR/applications/"
fi

# notebooks-v1 (formerly notebooks; includes notebook-controller, jupyter-web-app, etc.)
if [[ -d "$SRC/applications/notebooks-v1" ]]; then
    echo "  Copying applications/notebooks-v1"
    cp -r "$SRC/applications/notebooks-v1" "$MANIFESTS_DIR/applications/"
fi

# hub (Model Registry)
if [[ -d "$SRC/applications/hub" ]]; then
    echo "  Copying applications/hub"
    cp -r "$SRC/applications/hub" "$MANIFESTS_DIR/applications/"
fi

# dashboard (Central Dashboard)
if [[ -d "$SRC/applications/dashboard" ]]; then
    echo "  Copying applications/dashboard"
    cp -r "$SRC/applications/dashboard" "$MANIFESTS_DIR/applications/"
fi

# trainer
if [[ -d "$SRC/applications/trainer" ]]; then
    echo "  Copying applications/trainer"
    cp -r "$SRC/applications/trainer" "$MANIFESTS_DIR/applications/"
fi

# profile-controller is extracted with common/user-namespace (26.03.1 has no
# standalone applications/admission-webhook or metrics-collector).

# 3. third-party dependencies (under pipeline/upstream/third-party)
# Note: grafana/prometheus/seaweedfs are not required and are not extracted
# (seaweedfs points to the shared S3 through an ExternalName).
mkdir -p "$MANIFESTS_DIR/third-party"
for comp in application argo mysql postgresql metacontroller; do
    if [[ -d "$SRC/applications/pipeline/upstream/third-party/$comp" ]]; then
        echo "  Copying third-party/$comp"
        cp -r "$SRC/applications/pipeline/upstream/third-party/$comp" "$MANIFESTS_DIR/third-party/"
    else
        echo "  WARNING: third-party/$comp not found"
    fi
done

# 4. trainer third-party dependencies
if [[ -d "$SRC/applications/trainer/upstream/third-party" ]]; then
    for comp in $(ls "$SRC/applications/trainer/upstream/third-party/"); do
        echo "  Copying trainer/third-party/$comp"
        cp -r "$SRC/applications/trainer/upstream/third-party/$comp" "$MANIFESTS_DIR/third-party/" 2>/dev/null || true
    done
fi

# Remove the temporary directory.
rm -rf "$MANIFESTS_DIR/tmp-repo"

# 5. Apply kubeflow-patches (patches -> additions -> removals).
if [[ -d "$PATCHES_DIR" ]]; then
    echo ">>> Applying kubeflow-patches (patches -> additions -> removals)..."
    bash "$PATCHES_DIR/apply.sh" "$MANIFESTS_DIR"
else
    echo "!!! WARNING: $PATCHES_DIR not found; the result will be only a pristine 26.03.1 subset without business patches"
fi

echo ""
echo "=== Done ==="
echo "Final manifests directory: $MANIFESTS_DIR"
echo "Layout:"
tree "$MANIFESTS_DIR" -L 3 2>/dev/null || find "$MANIFESTS_DIR" -type f -name "*.yaml" | head -40
