#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# build-images.sh
# Build the NF self-built images as local docker tags. This script never
# pushes: local make is build-only, GHCR publication is owned by the
# publish-images GitHub Actions workflow, and local/air-gapped publication is
# an operator-only manual docker tag/push or NereidOverview harbor-sync.sh.
# Every image uses the single public scheme (root treaty section 18):
# ghcr.io/nereidai/<flat-name>:<VERSION>, resolved by nf_ref() in
# lib/common.sh. Air-gapped sites get the same refs through the node
# containerd registry mirror; there is no profile switch.
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# No VERSION file (root treaty §19.5): the exact v* git tag supplies bare
# SemVer (v0.0.1 -> 0.0.1); any other checkout builds the rolling 0.0.0-dev
# channel. An exported VERSION still overrides the derivation.
VERSION="${VERSION:-$(ref=$(git -C "$WORKDIR" describe --tags --exact-match --exclude 'pre-nereidforge/v*' 2>/dev/null) && echo "${ref#v}" || echo 0.0.0-dev)}"
# Three-tag scheme (root treaty §19.2): every local build carries the moving
# VERSION tag plus UTC-timestamp and git-short-SHA traceability tags. No
# floating tag is ever created. Local builds stay native/single-arch.
BUILD_TS="$(date -u +%Y%m%d-%H%M)"
GIT_SHA="${GIT_SHA:-$(git -C "$WORKDIR" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
BUILD_DATE="${BUILD_DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

# shellcheck source=lib/common.sh
source "$WORKDIR/scripts/lib/common.sh"
# shellcheck source=lib/config.sh
source "$WORKDIR/scripts/lib/config.sh"

echo "=== Build NF self-built images ==="
echo "Registry root: ghcr.io/nereidai (flat layout)"
echo "Version: $VERSION"
echo "pip: $PIP_INDEX_URL  apt mirror: ${APT_MIRROR:-<upstream>}"
echo ""

# Generic endpoints are injected by the build environment; empty values select
# pristine upstream sources. Specialized vendor wheel indexes are validated
# below before any image is built, because generic PyPI is not a substitute.
# Empty values stay as "--build-arg X=", overriding Dockerfile ARG defaults.
MIRROR_BUILD_ARGS=(
    --build-arg "PIP_INDEX_URL=$PIP_INDEX_URL"
    --build-arg "PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST"
    --build-arg "APT_MIRROR=$APT_MIRROR"
    --build-arg "TORCH_CPU_INDEX=$TORCH_CPU_INDEX"
    --build-arg "TORCH_XPU_INDEX=$TORCH_XPU_INDEX"
    --build-arg "TORCH_ROCM_INDEX=$TORCH_ROCM_INDEX"
    --build-arg "PYPI_INDEX=$PIP_INDEX_URL"
    --build-arg "HF_ENDPOINT=$HF_ENDPOINT"
    --build-arg "PADDLE_GPU_INDEX=$PADDLE_GPU_INDEX"
)
# OCI labels use the same version/revision as the three local tags; BUILD_DATE
# remains an RFC3339 value rather than the compact timestamp used in tag names.
IMAGE_METADATA_ARGS=(
    --build-arg "VERSION=$VERSION"
    --build-arg "GIT_SHA=$GIT_SHA"
    --build-arg "BUILD_DATE=$BUILD_DATE"
)

# Validate the selected matrix before building any core image. A
# specialized vendor failure is fatal: a partial matrix is not a valid release.
TRAINER_VENDORS="${TRAINER_VENDORS-xpu}"
for v in $TRAINER_VENDORS; do
    case "$v" in
        xpu)  : "${TORCH_XPU_INDEX:?set TORCH_XPU_INDEX for the XPU matrix}" ;;
        rocm) : "${TORCH_ROCM_INDEX:?set TORCH_ROCM_INDEX for the ROCm matrix}" ;;
        cu)   : "${PADDLE_GPU_INDEX:?set PADDLE_GPU_INDEX for the CUDA matrix}" ;;
        *)    echo "ERROR: unknown trainer vendor: $v (expected cu|rocm|xpu)" >&2; exit 2 ;;
    esac
done

build_image() {
    local repo="$1"
    local dockerfile="$2"
    local context="$3"
    # Optional extra arguments forwarded to `docker build` (e.g. --build-arg).
    local extra_args="${4:-}"
    local tag
    tag="$(nf_ref "$repo" "$VERSION")"

    echo "build: $tag"
    # shellcheck disable=SC2086 # extra_args is intentionally word-split
    docker build "${IMAGE_METADATA_ARGS[@]}" "${MIRROR_BUILD_ARGS[@]}" $extra_args -f "$dockerfile" \
        -t "$tag" -t "$tag-$BUILD_TS" -t "$tag-$GIT_SHA" "$context"
    echo "built locally (not pushed): $tag (+ $tag-$BUILD_TS, $tag-$GIT_SHA)"
    echo ""
}

# Backend and frontend default to local verification builds. The dedicated
# Actions application job sets NF_BUILD_APP_IMAGES=0 because it builds and
# publishes their real AMD64+ARM64 indexes directly instead of using these
# host-architecture Docker images.
if [[ "${NF_BUILD_APP_IMAGES:-1}" == "1" ]]; then
    # Backend needs the repo-root context so its Dockerfile can COPY configs/.
    build_image "nereidforge-backend" \
        "$WORKDIR/python-backend/Dockerfile" \
        "$WORKDIR"

    # Frontend bakes release and upload settings into the static bundle.
    build_image "nereidforge-frontend" \
        "$WORKDIR/frontend/Dockerfile" \
        "$WORKDIR/frontend" \
        "--build-arg VITE_APP_VERSION=$VERSION --build-arg VITE_IMAGE_TAG=$VERSION --build-arg VITE_DATASET_UPLOAD_MAX_FILES=${VITE_DATASET_UPLOAD_MAX_FILES:-200} --build-arg VITE_DATASET_UPLOAD_MAX_BYTES=${VITE_DATASET_UPLOAD_MAX_BYTES:-1073741824}"
fi

# 3. Trainer images — self-built training runtimes (tag follows VERSION).
#    Flat GHCR repo trainer-<name>, e.g. trainer/torch-cpu -> trainer-torch-cpu.
#    torch  = torchvision + timm (all vision tasks), paddle = PaddleDetection,
#    llm    = LLaMA-Factory LLM/VLM.
TRAINER_CTX="$WORKDIR/kfp-pipelines/docker"
trainer_ref() { nf_ref "trainer/$1" "$VERSION"; }

# 3a. Vision trainers — no weights are baked into any image (ADR-014 D5);
#   `base_weights_uri` (S3) or the permissive upstream backbone download is
#   used at run time, so there is no weight-preseeding step here.
#
#   trainer/torch-cpu: torchvision + timm + smp, every vision task in one
#   image. GPU variants are per-vendor images built below — vendors are NEVER
#   merged; each accelerator vendor has an independent runtime stack.
#   The build context is builders: the trainer sources and the shared
#   Trainer SDK package ship together (ADR-016).
docker build "${IMAGE_METADATA_ARGS[@]}" "${MIRROR_BUILD_ARGS[@]}" \
    -f "$WORKDIR/builders/trainer-torch/Dockerfile" \
    -t "$(trainer_ref torch-cpu)" \
    -t "$(trainer_ref torch-cpu)-$BUILD_TS" \
    -t "$(trainer_ref torch-cpu)-$GIT_SHA" \
    "$WORKDIR/builders"
echo "built locally (not pushed): $(trainer_ref torch-cpu) (+ timestamp and git-SHA tags)"

#   trainer/paddle-cpu: PaddleX / PP-YOLOE (second supported framework).
#   Context is builders like the torch trainer: the component sources and
#   the shared Trainer SDK ship together (ADR-016).
docker build "${IMAGE_METADATA_ARGS[@]}" "${MIRROR_BUILD_ARGS[@]}" --build-arg PYTHON_BASE_IMAGE="$IMG_PYTHON_BASE" \
    -f "$WORKDIR/builders/trainer-paddle/Dockerfile" \
    -t "$(trainer_ref paddle-cpu)" \
    -t "$(trainer_ref paddle-cpu)-$BUILD_TS" \
    -t "$(trainer_ref paddle-cpu)-$GIT_SHA" \
    "$WORKDIR/builders"
echo "built locally (not pushed): $(trainer_ref paddle-cpu) (+ timestamp and git-SHA tags)"

# 3a-ter. Torch/custom conversion uses the Python 3.14 default. The removed
#   AGPL legacy YOLO trees are not packaged; Paddle and TensorFlow artifacts
#   use the dedicated converters below. Same tag convention as the trainers.
docker build "${IMAGE_METADATA_ARGS[@]}" "${MIRROR_BUILD_ARGS[@]}" --build-arg PYTHON_BASE_IMAGE="$IMG_PYTHON_314_BASE" \
    -f "$TRAINER_CTX/Dockerfile.convert" \
    -t "$(trainer_ref convert)" \
    -t "$(trainer_ref convert)-$BUILD_TS" \
    -t "$(trainer_ref convert)-$GIT_SHA" \
    "$TRAINER_CTX"
echo "built locally (not pushed): $(trainer_ref convert) (+ timestamp and git-SHA tags)"

# 3a-ter-2. Per-family converter images (per-vendor separation rule): TF/Keras
#   and Paddle artifacts get their own images so the torch convert image never
#   carries tensorflow/paddle. Their user-required Python 3.11 pins remain until
#   compatible cp314 wheels exist and the suite treaty exception is registered.
docker build "${IMAGE_METADATA_ARGS[@]}" "${MIRROR_BUILD_ARGS[@]}" --build-arg PYTHON_BASE_IMAGE="$IMG_PYTHON_BASE" \
    -f "$TRAINER_CTX/Dockerfile.convert-tf" \
    -t "$(trainer_ref convert-tf)" \
    -t "$(trainer_ref convert-tf)-$BUILD_TS" \
    -t "$(trainer_ref convert-tf)-$GIT_SHA" \
    "$TRAINER_CTX"
echo "built locally (not pushed): $(trainer_ref convert-tf) (+ timestamp and git-SHA tags)"

docker build "${IMAGE_METADATA_ARGS[@]}" "${MIRROR_BUILD_ARGS[@]}" --build-arg PYTHON_BASE_IMAGE="$IMG_PYTHON_BASE" \
    -f "$TRAINER_CTX/Dockerfile.convert-pp" \
    -t "$(trainer_ref convert-pp)" \
    -t "$(trainer_ref convert-pp)-$BUILD_TS" \
    -t "$(trainer_ref convert-pp)-$GIT_SHA" \
    "$TRAINER_CTX"
echo "built locally (not pushed): $(trainer_ref convert-pp) (+ timestamp and git-SHA tags)"

# 3a-bis. Per-vendor GPU trainer images, each a fully independent image/tag/
#   dependency stack (cu=CUDA/LLaMA-Factory, rocm=ROCm, xpu=native torch.xpu).
#   Select vendors with TRAINER_VENDORS="cu rocm xpu" (empty = skip).
#   Default is xpu only: NVIDIA/AMD are paused as of 2026-10-21 (re-enable by
#   overriding TRAINER_VENDORS, e.g. TRAINER_VENDORS="cu xpu"; rocm needs its
#   ~30 GB upstream base mirrored/pullable). Build logic: build-vendor-trainer.sh.
#   The selected matrix and required indexes were validated before core builds.

# build_vendor <kind:torch|paddle|llm> <vendor:cu|rocm|xpu>
build_vendor() {
    RELEASE_TAG="$VERSION" GIT_SHA="$GIT_SHA" BUILD_DATE="$BUILD_DATE" \
        bash "$WORKDIR/scripts/build-vendor-trainer.sh" "$1" "$2"
}

for v in $TRAINER_VENDORS; do
    build_vendor torch "$v"
    # Paddle ships cu only: no ROCm wheels and zero xpu paddlepaddle builds
    # exist on PyPI (ADR-014 D-01, corrected 2026-10).
    if [ "$v" = "cu" ]; then
        build_vendor paddle "$v"
    fi
    build_vendor llm "$v"
done

# 4. ML Backends (enable on demand)
# build_image "ml-backend-det" \
#     "$WORKDIR/ml-backend/Dockerfile.det" \
#     "$WORKDIR/ml-backend"
#
# build_image "ml-backend-seg" \
#     "$WORKDIR/ml-backend/Dockerfile.seg" \
#     "$WORKDIR/ml-backend"
#
# build_image "ml-backend-ocr" \
#     "$WORKDIR/ml-backend/Dockerfile.ocr" \
#     "$WORKDIR/ml-backend"
#
# build_image "ml-backend-vllm" \
#     "$WORKDIR/ml-backend/Dockerfile.vllm" \
#     "$WORKDIR/ml-backend"

echo "=== NF self-built image build complete (local tags only, nothing pushed) ==="
echo "Built images:"
# NOTE: --format belongs to `docker images`, never chain it onto grep (grep
# exits 2 on an unknown option and, under set -e, fails the whole script after
# every image was already built — pit 138 CI follow-up).
docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" \
    | grep -F "ghcr.io/nereidai" | grep -F ":$VERSION" || true
