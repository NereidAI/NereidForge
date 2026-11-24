# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# NereidForge vision trainer — NVIDIA CUDA (per-vendor image, ADR-012/014;
# Blackwell sm_120 baseline per ADR-023).
#
# Base: the official PyTorch devel image carrying torch 2.8.0 + CUDA 12.8 on
# Ubuntu 22.04 / Python 3.11. Its prebuilt cubins cover sm_120 (RTX 50-series
# Blackwell such as the RTX 5060 Ti) while retaining sm_80/86/89/90, so Ampere
# and Ada cards keep working from the same image. The previous
# hiyouga/llamafactory:0.9.4 base shipped torch 2.6+cu124, whose arch list ends
# at sm_90 and fails on Blackwell with "no kernel image is available".
# torch/torchvision come FROM the base on purpose — reinstalling would let pip
# swap in a CPU build. Host requirement: NVIDIA driver >= 570 (R580 recommended
# once the CUDA 12.9 Paddle stack runs on the same node).
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='trainer-torch-cu' \
      org.opencontainers.image.description='NVIDIA CUDA PyTorch vision training and model export for NereidForge.' \
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
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}
ENV PYTHONUNBUFFERED=1

RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s@//.*archive.ubuntu.com@//$APT_MIRROR@g; s@//security.ubuntu.com@//$APT_MIRROR@g" \
            /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# torch.cuda.get_arch_list() returns [] without a GPU in torch 2.8. Query the
# compiled CUDA_ARCH_FLAGS directly so this build-time assertion works on CI.
RUN python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda); \
    arches=torch._C._cuda_getArchFlags().split(); print('arch list:', arches); \
    assert any('sm_120' in a for a in arches), 'Blackwell sm_120 cubin missing from the torch build'" \
    && pip install --no-cache-dir --timeout 120 --retries 10 \
        "timm>=1.0" "segmentation-models-pytorch>=0.3" \
        "numpy<2.3" "pillow>=10.0" "pyyaml>=6.0" "boto3>=1.34" \
        "onnx>=1.14,<2" "onnxruntime>=1.17"

ENV NF_DEVICE=cuda
# The SDK is the only shared contract (ADR-016); the build context is
# builders so the package and the trainer sources ship together.
COPY sdk /opt/nf-trainer-sdk
RUN pip install --no-cache-dir /opt/nf-trainer-sdk && rm -rf /opt/nf-trainer-sdk
COPY trainer-torch/common.py trainer-torch/dataset.py trainer-torch/models.py \
     trainer-torch/metrics.py trainer-torch/train.py trainer-torch/evaluate.py \
     trainer-torch/export.py trainer-torch/catalog.py trainer-torch/nf_profile.py \
     trainer-torch/trainer-profile.json /app/
WORKDIR /app

ENTRYPOINT ["sh", "-c", "case \"${SCRIPT:-train}\" in evaluate) exec python /app/evaluate.py;; export) exec python /app/export.py;; catalog) exec python /app/catalog.py;; profile) exec python /app/nf_profile.py;; *) exec python /app/train.py;; esac"]
