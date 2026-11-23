#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Package the nine active NF-owned and infrastructure Helm charts for local or
# offline use. Staging applies the tag-sourced product release version without
# modifying source Chart.yaml files. The legacy Kubeflow Helm umbrella is not an
# active installation path and is deliberately excluded. Dependencies are
# resolved fail-fast and no upload path exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
WORKDIR="$ROOT_DIR/deploy/pre-nereidforge"
OUTPUT_DIR="${CHARTS_OUTPUT_DIR:-$ROOT_DIR/charts-packaged}"
ARCHIVE="${CHARTS_ARCHIVE:-$ROOT_DIR/charts-offline.tar.gz}"
# packages archives tarballs/index for normal local use; offline-tree archives
# the staged repository-relative chart trees expected by 02_load-offline.
ARCHIVE_LAYOUT="${CHARTS_ARCHIVE_LAYOUT:-packages}"
HELM="${HELM:-helm}"
# No tracked VERSION file (root treaty §19.5): package the explicit
# CHART_VERSION, the release.env/RELEASE_TAG value, the exact v* git tag, or the
# rolling development version, in that order.
CHART_VERSION="${CHART_VERSION:-$(nf_version)}"
[[ "$ARCHIVE_LAYOUT" == "packages" || "$ARCHIVE_LAYOUT" == "offline-tree" ]] || {
    echo "Unsupported CHARTS_ARCHIVE_LAYOUT: $ARCHIVE_LAYOUT" >&2
    exit 2
}

# source-dir : package-name : version-mode (CHART_VERSION or upstream Chart.yaml).
# NF product charts gain their DNS-1123 prefix; upstream products retain names.
CHARTS=(
    "deploy/pre-nereidforge/charts/postgresql:postgresql:upstream"
    "deploy/pre-nereidforge/charts/valkey:valkey:upstream"
    "deploy/pre-nereidforge/charts/seaweedfs:seaweedfs:upstream"
    "deploy/pre-nereidforge/charts/labelstudio:labelstudio:upstream"
    "deploy/pre-nereidforge/charts/cvat:cvat:upstream"
    "charts/ml-backend:nereidforge-ml-backend:version"
    "charts/nereidforge-backend:nereidforge-backend:version"
    "charts/nereidforge-frontend:nereidforge-frontend:version"
    "charts/tenant-check:nereidforge-tenant-check:version"
)

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
mkdir -p "$OUTPUT_DIR"
# A reused output directory must not reintroduce removed charts into the index
# or archive. Only generated chart packages are cleaned; unrelated directories
# and operator files are left alone, and the package archive uses an allowlist.
find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.tgz' -delete
rm -f "$OUTPUT_DIR/index.yaml"
package_files=()

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
        sed -i -E "0,/^appVersion:[[:space:]]*\\S/s/^appVersion:[[:space:]]*\\S.*$/appVersion: \"$CHART_VERSION\"/" "$workdir/Chart.yaml"
    fi

    # Third-party subchart archives are fetched at package time, never vendored
    # into the Apache-2.0 source tree. Helm's locked build also requires every
    # HTTPS repository to exist in its local repository configuration, so derive
    # temporary public entries from Chart.yaml instead of relying on host state.
    if grep -q '^dependencies:' "$workdir/Chart.yaml"; then
        repo_index=0
        while IFS= read -r repository; do
            [[ -n "$repository" ]] || continue
            case "$repository" in
                https://*)
                    "$HELM" repo add "nf-packaging-${repo_index}" "$repository" --force-update
                    repo_index=$((repo_index + 1))
                    ;;
                http://*)
                    echo "Plaintext Helm repositories are not allowed: $repository" >&2
                    exit 1
                    ;;
            esac
        done < <(awk '/^  *repository:/ {print $2}' "$workdir/Chart.yaml" | tr -d '"')
        (cd "$workdir" && "$HELM" dependency build)
    fi

    package_args=("$workdir" -d "$OUTPUT_DIR")
    [[ "$vermode" == "version" ]] && package_args+=(--version "$CHART_VERSION" --app-version "$CHART_VERSION")
    "$HELM" package "${package_args[@]}"
    package_version="$(awk '/^version:/ {print $2; exit}' "$workdir/Chart.yaml" | tr -d '"')"
    package_files+=("${pubname}-${package_version}.tgz")
done
[[ "${#package_files[@]}" -eq 9 ]]

# A relative index works for offline use; an operator may provide a site URL.
if [[ -n "${CHARTS_REPO_URL:-}" ]]; then
    "$HELM" repo index "$OUTPUT_DIR" --url "$CHARTS_REPO_URL"
else
    "$HELM" repo index "$OUTPUT_DIR"
fi

case "$ARCHIVE_LAYOUT" in
    packages)
        tar -czf "$ARCHIVE" -C "$OUTPUT_DIR" index.yaml "${package_files[@]}"
        ;;
    offline-tree)
        tar -czf "$ARCHIVE" -C "$STAGING" deploy charts
        ;;
    *)
        echo "Unsupported CHARTS_ARCHIVE_LAYOUT: $ARCHIVE_LAYOUT" >&2
        exit 2
        ;;
esac
printf '=== Local chart packaging complete; nothing uploaded ===\n'
printf 'Packaged charts: %s\n' "$OUTPUT_DIR"
printf 'Chart archive (%s): %s\n' "$ARCHIVE_LAYOUT" "$ARCHIVE"
