#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Build and independently reparse a versioned ONLINE source installer. This
# entry point reads no registry, accesses no cluster, and creates no offline
# image bundle. Only tracked files below the pre module are eligible for staging.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$PRE_ROOT/scripts/lib/common.sh"

check_cmd git tar sha256sum python3
if [[ -z "$REPO_ROOT" ]]; then
    log_err "Packaging requires a Git worktree containing the pre module"
    exit 1
fi

VERSION="$(pre_version)"
if ! pre_validate_semver "$VERSION"; then
    log_err "Resolved pre version is not bare SemVer: $VERSION"
    exit 2
fi
# A Git package describes the actual checkout, never inherited metadata from a
# different archive or build environment.
SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"

OUTPUT_DIR="${PRE_OUTPUT_DIR:-$PRE_ROOT/dist}"
PREFIX="pre-nereidforge-installer-$VERSION"
ARCHIVE="${PRE_ARCHIVE:-$OUTPUT_DIR/$PREFIX.tar.gz}"
LOCK_FILE="${PRE_LOCK_FILE:-$OUTPUT_DIR/$PREFIX.lock.json}"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
TREE="$STAGING/tree/$PREFIX"
mkdir -p "$TREE" "$OUTPUT_DIR" "$(dirname "$ARCHIVE")" "$(dirname "$LOCK_FILE")"

if [[ "$PRE_ROOT" == "$REPO_ROOT" ]]; then
    MODULE_REL="."
else
    MODULE_REL="${PRE_ROOT#"$REPO_ROOT"/}"
fi
mkdir -p "$STAGING/repository"
(
    cd "$REPO_ROOT"
    git ls-files -z -- "$MODULE_REL" | tar --null -T - -cf -
) | tar -xf - -C "$STAGING/repository"
cp -a "$STAGING/repository/$MODULE_REL/." "$TREE/"

for required in \
    Makefile \
    scripts/lib/common.sh \
    scripts/lib/config.sh \
    scripts/lib/generate_nf_handoff.py \
    scripts/lib/pg_credentials.py \
    scripts/40_finalize-handoff.sh \
    scripts/40_check-nf-handoff.sh \
    scripts/lib/provision_s3_buckets.py \
    scripts/lib/s3_credentials.py \
    scripts/lib/publication.sh \
    scripts/package.sh \
    scripts/package-charts.sh \
    scripts/generate-lock.py \
    platform/kubeflow-manifests \
    platform/kubeflow-patches/apply.sh; do
    if [[ ! -e "$TREE/$required" ]]; then
        log_err "Staged pre module is missing $required (stage new source files before packaging)"
        exit 1
    fi
done
if [[ -e "$TREE/.git" || -e "$TREE/scripts/00_deploy-all.sh" || \
      -e "$TREE/values-prod.yaml" || -e "$TREE/secrets.yaml.template" || \
      -e "$TREE/images-manifest.txt" ]]; then
    log_err "Staged pre module contains repository-root or Git-only content"
    exit 1
fi

cat > "$TREE/release.env" <<EOF
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0
# Generated pre-NereidForge installer metadata; never tracked in source.
PRE_RELEASE_TAG=$VERSION
SOURCE_COMMIT=$SOURCE_COMMIT
EOF

python3 "$SCRIPT_DIR/generate-lock.py" "$TREE" "$VERSION" "$SOURCE_COMMIT" "$TREE/install-lock.json"
cp "$TREE/install-lock.json" "$LOCK_FILE"

rm -f "$ARCHIVE" "$ARCHIVE.sha256"
tar --sort=name --mtime="@${SOURCE_DATE_EPOCH:-0}" --owner=0 --group=0 \
    --numeric-owner -czf "$ARCHIVE" -C "$STAGING/tree" "$PREFIX"
(
    cd "$(dirname "$ARCHIVE")"
    sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256"
    sha256sum -c "$(basename "$ARCHIVE").sha256"
)

# Reparse the actual artifact in an empty environment outside the repository.
# Losing Git and both release-tag variables must not change the version, and
# sourcing config must resolve every runtime path back to the extracted module.
VERIFY_DIR="$STAGING/reparse"
mkdir -p "$VERIFY_DIR/home" "$VERIFY_DIR/operator"
tar -xzf "$ARCHIVE" -C "$VERIFY_DIR"
RESOLVED="$(env -i PATH="$PATH" HOME="$VERIFY_DIR/home" \
    DEPLOYMENT_ROOT="$VERIFY_DIR/operator" bash -c '
        set -euo pipefail
        module="$1"
        expected="$2"
        source "$module/scripts/lib/common.sh"
        actual="$(pre_version)"
        [[ "$actual" == "$expected" ]]
        source "$module/scripts/lib/config.sh"
        [[ "$PRE_ROOT" == "$module" ]]
        [[ "$CHARTS_DIR" == "$module/charts" ]]
        [[ "$MANIFESTS_DIR" == "$module/platform/kubeflow-manifests" ]]
        [[ -z "$REPO_ROOT" ]]
        printf "%s\n" "$actual"
    ' _ "$VERIFY_DIR/$PREFIX" "$VERSION")"
cmp "$TREE/install-lock.json" "$VERIFY_DIR/$PREFIX/install-lock.json"

printf 'Pre installer archive: %s\n' "$ARCHIVE"
printf 'SHA-256: %s.sha256\n' "$ARCHIVE"
printf 'Lock manifest: %s\n' "$LOCK_FILE"
printf 'Archived PRE_RELEASE_TAG: %s\n' "$RESOLVED"
