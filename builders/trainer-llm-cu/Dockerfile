# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# NVIDIA CUDA variant of the LLM/VLM trainer (separate per-vendor image).
# The official PyTorch CUDA 12.8 image supplies torch 2.8.0 with Blackwell
# sm_120 cubins (ADR-023); LLaMA-Factory itself is pip-installed here, mirroring
# the ROCm builder pattern. This layer adds SeaweedFS S3 glue, the pinned
# LLaMA-Factory release and the NereidForge entrypoint. Do not merge vendors
# into one image; each accelerator keeps its own builder/tag (ADR-012).
# Host requirement: NVIDIA driver >= 570.
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='trainer-llm-cu' \
      org.opencontainers.image.description='NVIDIA CUDA LLM and VLM fine-tuning trainer for NereidForge.' \
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
# Injected at build time for regional networks; empty uses pip's default index.
ENV PIP_INDEX_URL=${PIP_INDEX_URL} PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}
# LLaMA-Factory 0.9.5 (torch>=2.4, transformers>=4.55). The base already
# satisfies torch 2.8.0+cu128, so pip must not swap the CUDA torch build.
# Check compiled CUDA_ARCH_FLAGS directly: torch 2.8's public get_arch_list()
# returns [] on a CI builder without a GPU even when sm_120 cubins are present.
COPY sdk /opt/nf-trainer-sdk
RUN pip install --no-cache-dir "llamafactory[torch,metrics]==0.9.5" \
        "boto3>=1.34" "pyyaml>=6.0" /opt/nf-trainer-sdk \
    && rm -rf /opt/nf-trainer-sdk \
    && python -c "import torch; arches=torch._C._cuda_getArchFlags().split(); print('arch list:', arches); \
assert any('sm_120' in a for a in arches), 'Blackwell sm_120 cubin missing from the torch build'" \
    && python -c "from importlib.metadata import version; print('llamafactory', version('llamafactory')); assert version('llamafactory').startswith('0.9.5')"
COPY trainer-llm-cu/llm_train.py /app/llm_train.py
WORKDIR /app
ENTRYPOINT ["python", "/app/llm_train.py"]
