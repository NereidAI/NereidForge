#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Delegate individual builder-directory requests to the canonical Make targets.
# The Makefile owns public image names, shared contexts, dependency bases,
# specialized wheel-index guards, and the three local tags. This script never
# pushes or accepts an alternate repository; private delivery belongs exclusively
# to the operator-side NereidOverview harbor-sync.sh workflow.
#
# Usage:
#   ./build.sh <builder-dir> <tag>
#   ./builders/build.sh trainer-torch 0.0.0-dev
#   NF_DOCKERFILE=Dockerfile.xpu ./build.sh trainer-torch 0.0.0-dev
#
# The tag must match the tag-sourced release version (root treaty §19.5): bare
# SemVer of the exact v* git tag, or 0.0.0-dev on a non-tag checkout (py3.12
# stays the notebook-base exception). This prevents the compatibility entry
# point from bypassing the single tag source.
set -euo pipefail

BUILDER="${1:?usage: build.sh <builder-dir> <tag>}"
TAG="${2:?usage: build.sh <builder-dir> <tag>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$HERE/.." && pwd)"
EXPECTED_TAG="$(ref=$(git -C "$ROOT_DIR" describe --tags --exact-match --exclude 'pre-nereidforge/v*' 2>/dev/null) && echo "${ref#v}" || echo 0.0.0-dev)"
MAKE="${MAKE:-make}"
DIR="$HERE/$BUILDER"

[[ -d "$DIR" ]] || { echo "No such builder: $DIR" >&2; exit 2; }
[[ $# -le 2 ]] || {
  echo "Repository overrides are not supported; use canonical local tags and harbor-sync for delivery." >&2
  exit 2
}
[[ -z "${NF_BUILD_CONTEXT:-}" ]] || {
  echo "NF_BUILD_CONTEXT is not supported; canonical Make targets own each build context." >&2
  exit 2
}

# select_vendor accepts a torch/paddle kind and reads NF_DOCKERFILE, prints the
# canonical cpu/cu/rocm/xpu selector for the Make mapping, and returns 2 for an
# unsupported or retired variant. It performs no build or filesystem mutation.
select_vendor() {
  local kind="$1" selector="${NF_DOCKERFILE:-Dockerfile}"
  selector="${selector##*/}"
  case "$kind-$selector" in
    torch-Dockerfile) printf 'cpu\n' ;;
    torch-Dockerfile.cu|torch-Dockerfile.rocm|torch-Dockerfile.xpu)
      printf '%s\n' "${selector##*.}" ;;
    paddle-Dockerfile) printf 'cpu\n' ;;
    paddle-Dockerfile.cu) printf 'cu\n' ;;
    *) echo "Unsupported or retired $kind Dockerfile selector: $selector" >&2; return 2 ;;
  esac
}

TARGET=""
EXTRA_ARGS=()
case "$BUILDER" in
  evaluator) TARGET="build-evaluator" ;;
  evaluator-xpu) TARGET="build-evaluator-xpu" ;;
  optimizer) TARGET="build-optimizer" ;;
  trainer-xpu) TARGET="build-xpu-base" ;;
  ml-backend) TARGET="build-ml-backend" ;;
  ml-backend-custom) TARGET="build-ml-backend-custom" ;;
  notebook-base) TARGET="build-jupyter-base"; EXPECTED_TAG="py3.12" ;;
  inference-model-server-openvino-xpu) TARGET="build-ovms-xpu" ;;
  trainer-torch|trainer-paddle)
    kind="${BUILDER#trainer-}"
    vendor="$(select_vendor "$kind")"
    if [[ "$vendor" == "cpu" ]]; then
      if [[ "$kind" == "torch" ]]; then TARGET="build-trainer"; else TARGET="build-trainer-paddle"; fi
    else
      TARGET="build-trainer-vendor"
      EXTRA_ARGS=("kind=$kind" "vendor=$vendor")
    fi
    ;;
  trainer-llm-cu|trainer-llm-rocm|trainer-llm-xpu)
    [[ -z "${NF_DOCKERFILE:-}" ]] || { echo "NF_DOCKERFILE is fixed for $BUILDER." >&2; exit 2; }
    TARGET="build-trainer-vendor"
    EXTRA_ARGS=("kind=llm" "vendor=${BUILDER##*-}")
    ;;
  *) echo "No canonical local Make target for builder: $BUILDER" >&2; exit 2 ;;
esac

[[ "$TAG" == "$EXPECTED_TAG" ]] || {
  echo "Tag $TAG does not match the required $EXPECTED_TAG for $BUILDER." >&2
  exit 2
}

echo "== Delegating $BUILDER to make $TARGET ${EXTRA_ARGS[*]} (local three-tag build; no push)"
exec "$MAKE" -C "$ROOT_DIR" "$TARGET" "${EXTRA_ARGS[@]}"
