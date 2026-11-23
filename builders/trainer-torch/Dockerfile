# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# NereidForge vision trainer — CPU (torchvision + timm + smp).
#
# Context MUST be builders (the trainer sources plus the shared
# SDK package). One image per vendor; never an all-in-one (ADR-012/014).
#
# No weights are baked in: `base_weights_uri` or the permissive upstream
# backbone download provides transfer-learning initialisation at runtime.
ARG BASE_IMAGE=python:3.14-slim
FROM ${BASE_IMAGE}
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='trainer-torch-cpu' \
      org.opencontainers.image.description='CPU PyTorch vision training and model export for NereidForge.' \
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

# Build network endpoints injected at build time for regional networks;
# empty values reproduce pristine upstream behavior (default PyPI, upstream
# apt sources, and torch wheels from the package's default public index).
ARG PIP_INDEX_URL=""
ARG PIP_TRUSTED_HOST=""
ARG APT_MIRROR=""
ARG TORCH_CPU_INDEX=""
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1

ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=8

RUN if [ -n "$APT_MIRROR" ] && [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s@//deb.debian.org@//$APT_MIRROR@g; s@//security.debian.org@//$APT_MIRROR@g" \
            /etc/apt/sources.list.d/debian.sources; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*

RUN if [ -n "$TORCH_CPU_INDEX" ]; then \
        pip install --no-cache-dir --timeout 120 --retries 10 \
            --index-url "${TORCH_CPU_INDEX}" \
            "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; \
    else \
        pip install --no-cache-dir --timeout 120 --retries 10 \
            "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; \
    fi \
    && pip install --no-cache-dir --timeout 120 --retries 10 \
        "timm>=1.0" "segmentation-models-pytorch>=0.3" \
        "numpy<2.4" "pillow>=10.0" "pyyaml>=6.0" "boto3>=1.34" \
        "onnx>=1.14,<2" "onnxruntime>=1.17"

ENV NF_DEVICE=cpu
# The SDK is the only shared contract (ADR-016); the build context is
# builders so the package and the trainer sources ship together.
COPY sdk /opt/nf-trainer-sdk
RUN pip install --no-cache-dir /opt/nf-trainer-sdk && rm -rf /opt/nf-trainer-sdk
COPY trainer-torch/common.py trainer-torch/dataset.py trainer-torch/models.py \
     trainer-torch/metrics.py trainer-torch/train.py trainer-torch/evaluate.py \
     trainer-torch/export.py trainer-torch/catalog.py trainer-torch/nf_profile.py \
     trainer-torch/trainer-profile.json /app/
WORKDIR /app

# SCRIPT selects the entrypoint (train is the default). The KFP TrainJob sets
# the command explicitly, so a shell entrypoint cannot swallow it.
ENTRYPOINT ["sh", "-c", "case \"${SCRIPT:-train}\" in evaluate) exec python /app/evaluate.py;; export) exec python /app/export.py;; catalog) exec python /app/catalog.py;; profile) exec python /app/nf_profile.py;; *) exec python /app/train.py;; esac"]
