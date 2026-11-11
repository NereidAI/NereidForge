#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# build-optimizer.sh builds the legacy optimizer from its tracked Dockerfile.
# The local image uses the canonical public name and the tag-sourced release
# version (root treaty §19.5); publication is owned exclusively by the GitHub
# Actions workflow or the separately authorized NereidOverview delivery tooling.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Exact v* tag checkout -> bare SemVer; any other checkout -> 0.0.0-dev.
TAG="${VERSION:-$(ref=$(git -C "$DIR/../../.." describe --tags --exact-match 2>/dev/null) && echo "${ref#v}" || echo 0.0.0-dev)}"
IMAGE="$(bash "$DIR/../../../infra/scripts/image-ref.sh" train/optimizer "$TAG")"
# Three-tag scheme (root treaty §19.2): VERSION plus UTC-timestamp and
# git-short-SHA traceability tags; no floating tag is ever created.
BUILD_TS="$(date -u +%Y%m%d-%H%M)"
GIT_SHA="${GIT_SHA:-$(git -C "$DIR/../../.." rev-parse --short HEAD 2>/dev/null || echo unknown)}"
BUILD_DATE="${BUILD_DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
DF="$DIR/Dockerfile.optimizer"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"

# The tracked Dockerfile is authoritative; do not regenerate it during a build.
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

cd "$DIR"
echo "===== building optimizer $TAG (three local tags) ====="
sudo docker build "${BUILD_ARGS[@]}" -t "$IMAGE" -t "$IMAGE-$BUILD_TS" -t "$IMAGE-$GIT_SHA" \
  -f "$DF" . | tee "$LOG_DIR/optimizer-$TAG.log"
echo "===== built locally (not pushed): $IMAGE (+ $IMAGE-$BUILD_TS, $IMAGE-$GIT_SHA) ====="
