# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# NVIDIA CUDA variant of the LLM/VLM trainer (separate per-vendor image).
# LLaMA-Factory official base ships a CUDA build of PyTorch. This layer only
# adds SeaweedFS S3 glue + the NereidForge entrypoint. Do not merge vendors into
# one image; each accelerator keeps its own builder/tag (ADR-012).
ARG BASE_IMAGE=hiyouga/llamafactory:0.9.4
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
COPY sdk /opt/nf-trainer-sdk
RUN pip install --no-cache-dir "boto3>=1.34" "pyyaml>=6.0" /opt/nf-trainer-sdk \
    && rm -rf /opt/nf-trainer-sdk
COPY trainer-llm-cu/llm_train.py /app/llm_train.py
WORKDIR /app
ENTRYPOINT ["python", "/app/llm_train.py"]
