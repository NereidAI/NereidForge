# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# NereidForge Paddle vision trainer — NVIDIA CUDA (per-vendor image, ADR-012/014;
# Blackwell sm_120 baseline per ADR-023).
#
# Context MUST be builders (the trainer sources plus the shared SDK package).
#
# GPU wheels come from the official Paddle cu129 index, not from PyPI: `pip
# install paddlepaddle` on a GPU host installs the CPU build and silently trains
# on CPU. Resolve the exact paddlepaddle-gpu version from that package index;
# its links select the manylinux x86_64 cp311 build for this Python 3.11 base.
# Do not construct a wheel URL below the index root: the official repository
# nests files under /paddlepaddle-gpu/ and redirects them to its CDN. Paddle 3.3.x is
# the first line whose cu129 wheels embed sm_120 (Blackwell, e.g. RTX 5060 Ti)
# cubins; the former cu126 / paddle 3.0.0 combination stops at sm_90 and fails
# on Blackwell with an architecture-mismatch error. The wheel bundles its own
# CUDA 12.9 runtime libraries, so it runs on the CUDA 12.8 PyTorch base as long
# as the host driver supports CUDA 12.9 (Linux driver >= 575, R580 recommended).
#
# Base: the official PyTorch devel image (Ubuntu 22.04 / Python 3.11), used only
# for its CUDA userspace and interpreter; torch is present in the base but is
# NOT a dependency of this trainer — nothing here installs or relies on torch.
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='trainer-paddle-cu' \
      org.opencontainers.image.description='NVIDIA CUDA PaddleDetection training and model export for NereidForge.' \
      org.opencontainers.image.vendor='YIQISOFT' \
      org.opencontainers.image.licenses='Apache-2.0' \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source='https://github.com/NereidAI/NereidForge' \
      org.opencontainers.image.url='https://github.com/NereidAI/NereidForge' \
      org.opencontainers.image.documentation='https://github.com/NereidAI/NereidForge#readme' \
      org.opencontainers.image.authors='NereidAI (https://github.com/NereidAI)'
COPY LICENSE NOTICE /licenses/

ARG PIP_INDEX_URL=""
ARG PIP_TRUSTED_HOST=""
ARG APT_MIRROR=""
# Vendor-built Paddle GPU wheel index, injected at build time. It is mandatory:
# the default package index must never substitute a CPU or unverified GPU wheel.
# Paddle is pinned to 3.3.1 because sm_120 Blackwell support first ships in the
# cu129 wheel line (the cu126 / 3.0.0 line stops at sm_90). The earlier 3.1+
# PIR export regression against paddle2onnx 2.1.0 on PicoDet was documented on
# the old stack, so paddle2onnx stays at the verified 2.0.2rc3 and the full
# train -> PIR export -> ONNX chain on 3.3.1 is a required on-device release
# gate (ADR-023): it cannot be exercised without an sm_120 GPU.
ARG PADDLE_GPU_INDEX=""
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}
ENV PYTHONUNBUFFERED=1
ENV NF_DEVICE=cuda

# Switch Ubuntu apt to the injected regional mirror, then install headless CV
# system libs; an empty APT_MIRROR keeps upstream apt sources.
RUN if [ -n "$APT_MIRROR" ]; then sed -i "s@//.*archive.ubuntu.com@//$APT_MIRROR@g; s@//security.ubuntu.com@//$APT_MIRROR@g" \
        /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true; fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN if [ -z "$PADDLE_GPU_INDEX" ]; then \
        echo "ERROR: PADDLE_GPU_INDEX is required for the CUDA Paddle wheel" >&2; exit 1; \
    fi; \
    pip install --no-cache-dir --timeout 600 --retries 20 \
        --index-url "${PADDLE_GPU_INDEX}" \
        "paddlepaddle-gpu==3.3.1"

# The GPU-only distribution name and exact version prevent the generic package
# index from substituting the CPU paddlepaddle build during resolution.
# Fail fast if the base image ever moves off Python 3.11: a later layer writes
# into a hardcoded python3.11 site-packages path for the PaddleX repo tree.
RUN python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version" \
    && python -c "import paddle; assert paddle.__version__ == '3.3.1', paddle.__version__; assert paddle.device.is_compiled_with_cuda(), 'CPU-only paddle in the GPU image'; print('paddle', paddle.__version__, 'cuda', paddle.version.cuda())"

RUN pip install --no-cache-dir --timeout 300 --retries 20 \
        "paddlex==3.7.2" "paddle2onnx==2.0.2rc3" \
        "onnx>=1.14,<2" "onnxruntime>=1.17" \
        "opencv-contrib-python==4.10.0.84" "pillow>=10.0" "pyyaml>=6.0" \
        "numpy==1.26.4" "boto3>=1.34"

# PaddleX 3.7.2 ships the repo-API training configs only in the sdist (see the
# CPU Dockerfile for the full why); restore the detection config tree into the
# installed package from the matching sdist.
RUN pip download --no-deps --no-binary :all: "paddlex==3.7.2" -d /tmp/pdx-sdist \
    && api_dir="$(python -c 'import os,paddlex;print(os.path.join(os.path.dirname(paddlex.__file__),"repo_apis"))')" \
    && tar xzf /tmp/pdx-sdist/paddlex-*.tar.gz --strip-components=3 -C "$api_dir" \
        --wildcards "*/paddlex/repo_apis/PaddleDetection_api/configs" \
    && rm -rf /tmp/pdx-sdist

# Bake the PaddleDetection repo the repo API trains against (see the CPU
# Dockerfile for the why). On CUDA, PaddleX's gcc probe does NOT skip the
# rotated-Det custom-op build (cuda + gcc>=8.2); leave gcc in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && mkdir -p /usr/local/lib/python3.11/site-packages/paddlex/repo_manager/repos \
    && python -c "from paddlex.repo_manager import setup; setup(['PaddleDetection'])" \
    && python -c "from paddlex.repo_apis.base import PaddleModel; \
m = PaddleModel(model_name='PicoDet-S'); print('repo API OK:', type(m).__name__)"

COPY sdk /opt/nf-trainer-sdk
RUN pip install --no-cache-dir /opt/nf-trainer-sdk && rm -rf /opt/nf-trainer-sdk

RUN mkdir -p /root/.paddlex/PaddleX && chmod -R 777 /root/.paddlex

COPY trainer-paddle/common.py trainer-paddle/dataset.py trainer-paddle/models.py \
     trainer-paddle/trainer_metrics.py trainer-paddle/train.py trainer-paddle/evaluate.py \
     trainer-paddle/export.py trainer-paddle/catalog.py trainer-paddle/nf_profile.py \
     trainer-paddle/trainer-profile.json /app/
WORKDIR /app

ENTRYPOINT ["sh", "-c", "case \"${SCRIPT:-train}\" in evaluate) exec python /app/evaluate.py;; export) exec python /app/export.py;; catalog) exec python /app/catalog.py;; profile) exec python /app/nf_profile.py;; *) exec python /app/train.py;; esac"]
