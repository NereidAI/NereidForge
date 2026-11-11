#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Package every NF-owned and infrastructure Helm chart for local/offline use.
# Staging renames product charts and applies the tag-sourced release version
# without modifying source Chart.yaml files. Dependencies are resolved
# fail-fast and no upload path exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKDIR="$ROOT_DIR/infra"
OUTPUT_DIR="${CHARTS_OUTPUT_DIR:-$ROOT_DIR/charts-packaged}"
ARCHIVE="${CHARTS_ARCHIVE:-$ROOT_DIR/charts-offline.tar.gz}"
# packages archives tarballs/index for normal local use; offline-tree archives
# the staged repository-relative chart trees expected by 02_load-offline.
ARCHIVE_LAYOUT="${CHARTS_ARCHIVE_LAYOUT:-packages}"
HELM="${HELM:-helm}"
# No VERSION file (root treaty §19.5): package the bare SemVer from the exact
# v* git tag, or 0.0.0-dev on a non-tag checkout; CHART_VERSION may be injected
# by the operator for a delivery of a specific release.
CHART_VERSION="${CHART_VERSION:-$(ref=$(git -C "$ROOT_DIR" describe --tags --exact-match 2>/dev/null) && echo "${ref#v}" || echo 0.0.0-dev)}"
[[ "$ARCHIVE_LAYOUT" == "packages" || "$ARCHIVE_LAYOUT" == "offline-tree" ]] || {
    echo "Unsupported CHARTS_ARCHIVE_LAYOUT: $ARCHIVE_LAYOUT" >&2
    exit 2
}

# source-dir : package-name : version-mode (CHART_VERSION or upstream Chart.yaml).
# NF product charts gain their DNS-1123 prefix; upstream products retain names.
CHARTS=(
    "infra/charts/postgresql:postgresql:upstream"
    "infra/charts/valkey:valkey:version"
    "infra/charts/seaweedfs:seaweedfs:upstream"
    "infra/charts/labelstudio:labelstudio:upstream"
    "infra/charts/cvat:cvat:upstream"
    "infra/charts/kubeflow:kubeflow:version"
    "app/charts/ml-backend:nereidforge-ml-backend:version"
    "app/charts/nereidforge-backend:nereidforge-backend:version"
    "app/charts/nereidforge-frontend:nereidforge-frontend:version"
    "app/charts/tenant-check:nereidforge-tenant-check:version"
)

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
mkdir -p "$OUTPUT_DIR"

printf '=== Package Helm charts locally (chart version %s) ===\n' "$CHART_VERSION"
printf 'Output directory: %s\n' "$OUTPUT_DIR"

for entry in "${CHARTS[@]}"; do
    IFS=':' read -r relsrc pubname vermode <<< "$entry"
    srcdir="$ROOT_DIR/$relsrc"
    workdir="$STAGING/$relsrc"

    [[ -f "$srcdir/Chart.yaml" ]] || {
        echo "Missing chart definition: $srcdir/Chart.yaml" >&2
        exit 1
    }
    mkdir -p "$(dirname "$workdir")"
    cp -a "$srcdir" "$workdir"

    # Rename only the top-level chart; vendored subcharts retain upstream names.
    sed -i -E "0,/^name:[[:space:]]*\\S/s/^name:[[:space:]]*\\S.*$/name: $pubname/" "$workdir/Chart.yaml"
    # Keep staged tree archives and tarball metadata on the same CHART_VERSION.
    # Only the temporary copy changes; the source Chart.yaml stays byte-identical.
    if [[ "$vermode" == "version" ]]; then
        sed -i -E "0,/^version:[[:space:]]*\\S/s/^version:[[:space:]]*\\S.*$/version: $CHART_VERSION/" "$workdir/Chart.yaml"
    fi

    # Third-party subchart archives are fetched at package time, never vendored
    # into the Apache-2.0 source tree. A dependency failure aborts the whole set.
    if grep -q '^dependencies:' "$workdir/Chart.yaml"; then
        (cd "$workdir" && "$HELM" dependency build)
    fi

    package_args=("$workdir" -d "$OUTPUT_DIR")
    [[ "$vermode" == "version" ]] && package_args+=(--version "$CHART_VERSION")
    "$HELM" package "${package_args[@]}"
done

# A relative index works for offline use; an operator may provide a site URL.
if [[ -n "${CHARTS_REPO_URL:-}" ]]; then
    "$HELM" repo index "$OUTPUT_DIR" --url "$CHARTS_REPO_URL"
else
    "$HELM" repo index "$OUTPUT_DIR"
fi

case "$ARCHIVE_LAYOUT" in
    packages)
        tar -czf "$ARCHIVE" -C "$OUTPUT_DIR" .
        ;;
    offline-tree)
        tar -czf "$ARCHIVE" -C "$STAGING" infra app
        ;;
    *)
        echo "Unsupported CHARTS_ARCHIVE_LAYOUT: $ARCHIVE_LAYOUT" >&2
        exit 2
        ;;
esac
printf '=== Local chart packaging complete; nothing uploaded ===\n'
printf 'Packaged charts: %s\n' "$OUTPUT_DIR"
printf 'Chart archive (%s): %s\n' "$ARCHIVE_LAYOUT" "$ARCHIVE"
