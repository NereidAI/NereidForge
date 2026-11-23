#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Package exactly the five pre-NereidForge supporting-service charts. Formal and
# local packages preserve each chart's independent upstream version/appVersion.
# Publication workflows may set PRE_CHARTS_VERSION only for the commit-qualified
# development channel; appVersion still comes from each upstream Chart.yaml.
# Dependencies are resolved into a temporary copy and no upload path exists.
# Archive metadata is normalized so repackaging one commit is byte-identical,
# which permits safe immutable-tag reuse on workflow reruns.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$PRE_ROOT/scripts/lib/common.sh"

HELM="${HELM:-helm}"
check_cmd "$HELM" tar
OUTPUT_DIR="${PRE_CHARTS_OUTPUT_DIR:-$PRE_ROOT/dist/charts-packaged}"
ARCHIVE="${PRE_CHARTS_ARCHIVE:-$PRE_ROOT/dist/pre-nereidforge-charts.tar.gz}"
ARCHIVE_LAYOUT="${PRE_CHARTS_ARCHIVE_LAYOUT:-packages}"
CHART_VERSION_OVERRIDE="${PRE_CHARTS_VERSION:-}"
[[ "$ARCHIVE_LAYOUT" == "packages" || "$ARCHIVE_LAYOUT" == "offline-tree" ]] || {
    log_err "Unsupported PRE_CHARTS_ARCHIVE_LAYOUT: $ARCHIVE_LAYOUT"
    exit 2
}
if [[ -n "$CHART_VERSION_OVERRIDE" ]] && ! pre_validate_semver "$CHART_VERSION_OVERRIDE"; then
    log_err "PRE_CHARTS_VERSION must be bare SemVer: $CHART_VERSION_OVERRIDE"
    exit 2
fi

CHARTS=(postgresql valkey seaweedfs labelstudio cvat)
[[ "${#CHARTS[@]}" -eq 5 ]]
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
mkdir -p "$OUTPUT_DIR" "$(dirname "$ARCHIVE")"

normalize_chart_package() {
    local archive="$1" name="$2"
    local tree="$STAGING/normalized/$name"
    rm -rf "$tree"
    mkdir -p "$tree"
    tar -xzf "$archive" -C "$tree"
    if [[ ! -f "$tree/$name/Chart.yaml" ]]; then
        log_err "Helm package has an unexpected archive layout: $archive"
        return 1
    fi
    rm -f "$archive"
    tar --sort=name --mtime="@${SOURCE_DATE_EPOCH:-0}" --owner=0 --group=0 \
        --numeric-owner -czf "$archive" -C "$tree" "$name"
}

# Keep repository state and registry credentials out of the packaging process.
# HTTPS dependency repositories are temporary public entries derived from each
# chart; tracked CVAT archives are staged in their canonical charts/ directory.
export HELM_CONFIG_HOME="$STAGING/helm/config"
export HELM_CACHE_HOME="$STAGING/helm/cache"
export HELM_DATA_HOME="$STAGING/helm/data"
export HELM_REGISTRY_CONFIG="$STAGING/helm/registry-config.json"
mkdir -p "$HELM_CONFIG_HOME" "$HELM_CACHE_HOME" "$HELM_DATA_HOME"

find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.tgz' -delete
rm -f "$OUTPUT_DIR/index.yaml"
package_files=()

printf '=== Package five pre-NereidForge charts locally ===\n'
printf 'Output directory: %s\n' "$OUTPUT_DIR"
for name in "${CHARTS[@]}"; do
    srcdir="$PRE_ROOT/charts/$name"
    workdir="$STAGING/tree/deploy/pre-nereidforge/charts/$name"
    [[ -f "$srcdir/Chart.yaml" ]] || {
        log_err "Missing chart definition: $srcdir/Chart.yaml"
        exit 1
    }
    mkdir -p "$(dirname "$workdir")"
    cp -a "$srcdir" "$workdir"

    if [[ "$name" == "cvat" ]]; then
        mkdir -p "$workdir/charts"
        shopt -s nullglob
        vendored=("$workdir"/*.tgz)
        shopt -u nullglob
        [[ "${#vendored[@]}" -eq 6 ]] || {
            log_err "CVAT must have exactly six tracked dependency archives"
            exit 1
        }
        cp "${vendored[@]}" "$workdir/charts/"
        rm -f "${vendored[@]}"
    fi

    if grep -q '^dependencies:' "$workdir/Chart.yaml"; then
        repo_index=0
        while IFS= read -r repository; do
            [[ -n "$repository" ]] || continue
            case "$repository" in
                https://*)
                    "$HELM" repo add "pre-packaging-${name}-${repo_index}" "$repository" --force-update
                    repo_index=$((repo_index + 1))
                    ;;
                http://*)
                    log_err "Plaintext Helm repositories are not allowed: $repository"
                    exit 1
                    ;;
            esac
        done < <(awk '/^  *repository:/ {print $2}' "$workdir/Chart.yaml" | tr -d '"')
        "$HELM" dependency build "$workdir"
    fi

    package_args=("$workdir" --destination "$OUTPUT_DIR")
    if [[ -n "$CHART_VERSION_OVERRIDE" ]]; then
        package_args+=(--version "$CHART_VERSION_OVERRIDE")
    fi
    "$HELM" package "${package_args[@]}"
    metadata="$("$HELM" show chart "$workdir")"
    package_version="$(awk '/^version:/ {print $2; exit}' <<< "$metadata" | tr -d '"')"
    if [[ -n "$CHART_VERSION_OVERRIDE" ]]; then
        package_version="$CHART_VERSION_OVERRIDE"
    fi
    package_file="${name}-${package_version}.tgz"
    normalize_chart_package "$OUTPUT_DIR/$package_file" "$name"
    package_files+=("$package_file")
done
[[ "${#package_files[@]}" -eq 5 ]]

if [[ -n "${PRE_CHARTS_REPO_URL:-}" ]]; then
    "$HELM" repo index "$OUTPUT_DIR" --url "$PRE_CHARTS_REPO_URL"
else
    "$HELM" repo index "$OUTPUT_DIR"
fi

case "$ARCHIVE_LAYOUT" in
    packages)
        tar -czf "$ARCHIVE" -C "$OUTPUT_DIR" index.yaml "${package_files[@]}"
        ;;
    offline-tree)
        tar -czf "$ARCHIVE" -C "$STAGING/tree" deploy/pre-nereidforge/charts
        ;;
esac
printf '=== Pre chart packaging complete; nothing uploaded ===\n'
printf 'Packaged charts: %s\n' "$OUTPUT_DIR"
printf 'Chart archive (%s): %s\n' "$ARCHIVE_LAYOUT" "$ARCHIVE"
