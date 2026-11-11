# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# NereidForge vision trainer — NVIDIA CUDA (per-vendor image, ADR-012/014).
#
# Base: the already-mirrored LLaMA-Factory image, which ships torch 2.6+cu124
# and torchvision (avoids re-pulling the ~5 GB CUDA wheel set on slow links).
# torch/torchvision come FROM the base on purpose — reinstalling would let pip
# swap in a CPU build.
ARG BASE_IMAGE=hiyouga/llamafactory:0.9.4
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

RUN python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
    && pip install --no-cache-dir --timeout 120 --retries 10 \
        "timm>=1.0" "segmentation-models-pytorch>=0.3" \
        "numpy<2.3" "pillow>=10.0" "pyyaml>=6.0" "boto3>=1.34" \
        "onnx>=1.14,<2" "onnxruntime>=1.17"

ENV NF_DEVICE=cuda
# The SDK is the only shared contract (ADR-016); the build context is
# app/builders so the package and the trainer sources ship together.
COPY sdk /opt/nf-trainer-sdk
RUN pip install --no-cache-dir /opt/nf-trainer-sdk && rm -rf /opt/nf-trainer-sdk
COPY trainer-torch/common.py trainer-torch/dataset.py trainer-torch/models.py \
     trainer-torch/metrics.py trainer-torch/train.py trainer-torch/evaluate.py \
     trainer-torch/export.py trainer-torch/catalog.py trainer-torch/nf_profile.py \
     trainer-torch/trainer-profile.json /app/
WORKDIR /app

ENTRYPOINT ["sh", "-c", "case \"${SCRIPT:-train}\" in evaluate) exec python /app/evaluate.py;; export) exec python /app/export.py;; catalog) exec python /app/catalog.py;; profile) exec python /app/nf_profile.py;; *) exec python /app/train.py;; esac"]
