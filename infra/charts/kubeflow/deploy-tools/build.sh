#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Build the deploy-tools image used by the kubeflow chart install Job, tagged
# locally only. This script never uploads to a registry. Any GHCR publication
# must be defined explicitly in the tag-triggered GitHub Actions workflow; an
# operator-managed local hub is fed through NereidOverview harbor-sync.sh.
# Usage: ./build.sh [TAG]
#   TAG defaults to the tag-sourced release version (root treaty §19.5): bare
#   SemVer on an exact v* tag checkout, otherwise 0.0.0-dev. The canonical
#   public resolver owns the image coordinate; no registry override is accepted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
VERSION="$(ref=$(git -C "${REPO_ROOT}" describe --tags --exact-match 2>/dev/null) && echo "${ref#v}" || echo 0.0.0-dev)"
BUILD_TS="$(date -u +%Y%m%d-%H%M)"
GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"

TAG="${1:-$VERSION}"
IMAGE="$(bash "${REPO_ROOT}/infra/scripts/image-ref.sh" deploy-tools "$TAG")"

# Three-tag scheme (treaty §19.2): VERSION plus timestamp and git-sha tags.
echo ">> building ${IMAGE} (+ ${TAG}-${BUILD_TS}, ${TAG}-${GIT_SHA})"
docker build \
  -t "$IMAGE" \
  -t "${IMAGE}-${BUILD_TS}" \
  -t "${IMAGE}-${GIT_SHA}" \
  "$SCRIPT_DIR"

echo ">> built locally (not pushed): ${IMAGE}"
