# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Evaluator with Intel XPU support: runs IR edge-package NUMERICAL verification
# on Arc GPUs (model_package.py --spec-uri, "edge-verified" profile). FP16/INT8
# IR packages are checked on BOTH CPU and GPU here. Direct ONNX FP32 structural
# packages use the CPU evaluator profile and never claim this GPU execution.
#
# Why not FROM the regular evaluator: that image is Debian-based and the
# validated Battlemage compute userspace (NEO >= 26, kobuk PPA) is Ubuntu-only
# (see builders/inference-model-server-openvino-xpu). trainer-xpu-base is the
# validated Ubuntu 24.04 + NEO + xe-KMD stack. Build context MUST be
# app/builders so the same shared packaging script shipped in the CPU
# evaluator is copied into this image.
#
# The container runs as root: NEO enumerates zero GPUs for arbitrary
# non-root uids (nf-backend pit 174), and KFP pods default to root anyway.

# Tag-sourced model (root treaty §19.5): release builds pass the bare SemVer
# explicitly; the 0.0.0-dev default covers non-tag local builds.
ARG XPU_BASE_IMAGE=ghcr.io/nereidai/trainer-xpu-base:0.0.0-dev
FROM ${XPU_BASE_IMAGE}
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='train-evaluator-xpu' \
      org.opencontainers.image.description='Intel XPU model packaging and numerical evaluation for NereidForge.' \
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
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

# openvino must track the CPU evaluator's supported range so IR handling is
# identical; opencv implements the letterbox preprocess, boto3 reads/writes S3.
RUN /opt/xpu/bin/pip install --no-cache-dir \
        "boto3>=1.34.0" \
        "onnx>=1.14,<2" \
        "opencv-python-headless>=4.8.0" \
        "openvino>=2024.3.0" \
        "PyYAML>=6.0"

COPY common/ /app/common/
COPY common/model_package.py /app/model_package.py

WORKDIR /app
ENTRYPOINT ["sh", "-c", "exec /opt/xpu/bin/python /app/model_package.py"]
