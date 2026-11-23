#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# build-vendor-trainer.sh - Build one per-vendor trainer image locally.
# The image is tagged in the local docker daemon only; this script never
# pushes. GHCR publication is owned by the publish-images GitHub Actions
# workflow, and local/air-gapped publication is an operator-only manual
# docker tag/push or NereidOverview harbor-sync.sh.
#
# Usage: build-vendor-trainer.sh <kind:torch|paddle|llm> <vendor:cu|rocm|xpu>
#
#   torch   builders/trainer-torch/Dockerfile.<vendor> — torchvision + timm
#           (+ smp), every vision task in one image per vendor (ADR-014 D1).
#   paddle  builders/trainer-paddle/Dockerfile.<vendor> — PaddleX /
#           PP-YOLOE (the second supported framework); cu only: there is no
#           official ROCm wheel and zero xpu paddlepaddle builds on PyPI.
#   llm     builders/trainer-llm-<vendor>/ (context builders/)
#   cv      legacy alias for `torch`
#
# Every vendor FROM base is an ARG (BASE_IMAGE) resolved here to the public
# coordinate from lib/config.sh (GHCR flat refs for self-built bases, true
# upstream refs for third-party). Vendors stay independent images (ADR-012) —
# never build an all-in-one.
set -euo pipefail

KIND="${1:?usage: $0 <torch|paddle|llm> <cu|rocm|xpu>}"
VENDOR="${2:?usage: $0 <torch|paddle|llm> <cu|rocm|xpu>}"
[ "$KIND" = "cv" ] && KIND="torch"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/config.sh"

# Reject missing specialized wheel indexes before pulling large vendor bases.
# The Dockerfile checks remain as defense for direct docker-build callers.
case "$KIND-$VENDOR" in
    torch-xpu)
        : "${TORCH_XPU_INDEX:?set TORCH_XPU_INDEX for +xpu wheels}" ;;
    torch-rocm)
        : "${TORCH_ROCM_INDEX:?set TORCH_ROCM_INDEX for +rocm6.3 wheels}" ;;
    paddle-cu)
        : "${PADDLE_GPU_INDEX:?set PADDLE_GPU_INDEX for the CUDA Paddle wheel}" ;;
esac

# SCRIPT_DIR is infra/scripts; the repo root is two levels up (Dockerfiles
# live at the repository root, not under infra/ — a wrong level here makes the
# script print "skip (missing …)" and exit 0, silently building nothing).
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINER_CTX="$ROOT_DIR/kfp-pipelines/docker"
BUILDERS_DIR="$ROOT_DIR/builders"
# Three-tag scheme (root treaty §19.2): VERSION plus UTC-timestamp and
# git-short-SHA traceability tags; no floating tag is ever created.
BUILD_TS="$(date -u +%Y%m%d-%H%M)"
GIT_SHA="${GIT_SHA:-$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
BUILD_DATE="${BUILD_DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

case "$VENDOR" in
    cu)   base="$IMG_PYTORCH_CUDA" ;;
    rocm) base="$IMG_ROCM_PYTORCH" ;;
    xpu)  base="$IMG_XPU_BASE" ;;
    *) echo "unknown vendor: $VENDOR" >&2; exit 2 ;;
esac

if [ "$KIND" = "torch" ]; then
    df="$BUILDERS_DIR/trainer-torch/Dockerfile.$VENDOR"
    ctx="$BUILDERS_DIR"
elif [ "$KIND" = "paddle" ]; then
    if [ "$VENDOR" != "cu" ]; then
        echo "ERROR: paddle-$VENDOR does not exist: PaddleX ships cu only (no ROCm/XPU wheels, ADR-014 D-01)" >&2
        exit 2
    fi
    df="$BUILDERS_DIR/trainer-paddle/Dockerfile.$VENDOR"
    ctx="$BUILDERS_DIR"
elif [ "$KIND" = "llm" ]; then
    df="$BUILDERS_DIR/trainer-llm-$VENDOR/Dockerfile"
    ctx="$BUILDERS_DIR"
else
    echo "unknown kind: $KIND (expected torch|paddle|llm)" >&2; exit 2
fi

if [ ! -f "$df" ]; then
    # Hard failure (exit 2), not a silent skip: a requested vendor with no
    # Dockerfile is always a caller/path bug — 02_build's matrix loop wraps us
    # in `|| true` if it wants best-effort semantics.
    echo "ERROR: $KIND-$VENDOR Dockerfile not found: $df" >&2
    exit 2
fi

repo="$(nf_ref "trainer/$KIND-$VENDOR" "$RELEASE_TAG")"
echo "build: $repo (base $base)"
# The xpu variants need the self-built xpu-base present; build-xpu-base builds
# it. Failure (e.g. missing mirrored rocm base in public CI) is non-fatal for
# callers that iterate a vendor matrix — the caller decides via exit code 1.
# Generic endpoints are injected by the build environment and use pristine
# upstream defaults when empty; specialized indexes were validated above.
if docker build \
        --build-arg "VERSION=$RELEASE_TAG" \
        --build-arg "GIT_SHA=$GIT_SHA" \
        --build-arg "BUILD_DATE=$BUILD_DATE" \
        --build-arg "BASE_IMAGE=$base" \
        --build-arg "PIP_INDEX_URL=$PIP_INDEX_URL" \
        --build-arg "PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST" \
        --build-arg "APT_MIRROR=$APT_MIRROR" \
        --build-arg "TORCH_CPU_INDEX=$TORCH_CPU_INDEX" \
        --build-arg "TORCH_XPU_INDEX=$TORCH_XPU_INDEX" \
        --build-arg "TORCH_ROCM_INDEX=$TORCH_ROCM_INDEX" \
        --build-arg "PYPI_INDEX=$PIP_INDEX_URL" \
        --build-arg "HF_ENDPOINT=$HF_ENDPOINT" \
        --build-arg "PADDLE_GPU_INDEX=$PADDLE_GPU_INDEX" \
        -f "$df" \
        -t "$repo" -t "$repo-$BUILD_TS" -t "$repo-$GIT_SHA" \
        "$ctx"; then
    echo "built locally (not pushed): $repo (+ $repo-$BUILD_TS, $repo-$GIT_SHA)"
else
    echo "FAILED: $repo (base image unreachable or build error)" >&2
    exit 1
fi
