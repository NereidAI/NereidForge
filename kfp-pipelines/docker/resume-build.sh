#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# resume-build.sh resumes missing images from the build-images.sh aggregate.
# It forwards the same base/package overrides and preserves canonical public
# names and release tags. Any failed build makes the aggregate command fail;
# publication remains with GitHub Actions or separate NereidOverview tooling.
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

# Keep resume behavior identical to the aggregate helper's build inputs.
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
failed=0
for name in runtime checkpoint-step eval opt-tensorrt; do
  tag="$(bash "$DIR/../../scripts/image-ref.sh" "$name" "$TAG")"
  # The release tag presence means this legacy image was already built; the
  # traceability tags are recreated together with the release tag below.
  if sudo docker image inspect "$tag" >/dev/null 2>&1; then
    echo "== $name already exists locally, skipping build: $tag"
    continue
  fi
  echo "===== building $name ====="
  if sudo docker build --progress=plain "${BUILD_ARGS[@]}" -t "$tag" -t "$tag-$BUILD_TS" -t "$tag-$GIT_SHA" \
      -f "$DIR/Dockerfile.$name" . > "$LOG_DIR/$name.log" 2>&1; then
    echo "  -> built locally (not pushed): $tag (+ $tag-$BUILD_TS, $tag-$GIT_SHA)"
  else
    echo "  -> FAILED (see $LOG_DIR/$name.log)" >&2
    failed=$((failed+1))
  fi
done

echo "===== BUILD SUMMARY: failed=$failed / 4 ====="
exit "$failed"
