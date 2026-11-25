#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# build-images.sh builds the runtime plus three approved legacy definitions as
# local images, including the otherwise orphaned checkpoint-step definition.
# Every tag comes from the public resolver and the tag-sourced release version
# (root treaty §19.5). This helper has no publication path; customers
# independently choose GitHub-built artifacts or separate NereidOverview
# delivery tooling for a private hub.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Exact v* tag checkout -> bare SemVer; any other checkout -> 0.0.0-dev.
TAG="${VERSION:-$(ref=$(git -C "$DIR/../.." describe --tags --exact-match --exclude 'pre-nereidforge/v*' 2>/dev/null) && echo "${ref#v}" || echo 0.0.0-dev)}"
# Three-tag scheme (root treaty §19.2): VERSION plus UTC-timestamp and
# git-short-SHA traceability tags; no floating tag is ever created.
BUILD_TS="$(date -u +%Y%m%d-%H%M)"
GIT_SHA="${GIT_SHA:-$(git -C "$DIR/../.." rev-parse --short HEAD 2>/dev/null || echo unknown)}"
BUILD_DATE="${BUILD_DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"

# Tracked Dockerfiles are authoritative. This legacy helper must never overwrite
# them with divergent bases or dependency sets; local mirrors stay in the
# caller's environment and empty pip values preserve pristine upstream behavior.
BUILD_ARGS=(
  --build-arg "VERSION=$TAG"
  --build-arg "GIT_SHA=$GIT_SHA"
  --build-arg "BUILD_DATE=$BUILD_DATE"
  --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL-}"
  --build-arg "PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST-}"
)
if [[ -n "${PYTHON_BASE_IMAGE:-}" ]]; then
  BUILD_ARGS+=(--build-arg "PYTHON_BASE_IMAGE=$PYTHON_BASE_IMAGE")
fi
if [[ -n "${TENSORRT_BASE_IMAGE:-}" ]]; then
  BUILD_ARGS+=(--build-arg "TENSORRT_BASE_IMAGE=$TENSORRT_BASE_IMAGE")
fi

cd "$DIR"
# build_image builds one legacy image and returns a failed docker status to the
# aggregate loop instead of allowing a partial build to report success.
build_image() {
  local name="$1"
  local tag
  tag="$(bash "$DIR/../../scripts/image-ref.sh" "$name" "$TAG")"
  echo "===== building $name ====="
  if sudo docker build "${BUILD_ARGS[@]}" -t "$tag" -t "$tag-$BUILD_TS" -t "$tag-$GIT_SHA" \
      -f "$DIR/Dockerfile.$name" . > "$LOG_DIR/$name.log" 2>&1; then
    echo "  -> built locally (not pushed): $tag (+ $tag-$BUILD_TS, $tag-$GIT_SHA)"
  else
    echo "  -> FAILED (see $LOG_DIR/$name.log)" >&2
    return 1
  fi
}

failed=0
for name in runtime checkpoint-step eval opt-tensorrt; do
  if ! build_image "$name"; then failed=$((failed+1)); fi
done
echo "===== BUILD SUMMARY: failed=$failed / 4 ====="
exit "$failed"
