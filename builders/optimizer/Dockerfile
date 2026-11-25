# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Optimizer image: ONNX quantization (int8 dynamic/static, fp16), graph opt,
# OpenVINO IR conversion + benchmarking. No torch needed — everything works
# on the ONNX graph / ORT / OpenVINO level, keeping the image lean.
# TensorRT path is code-ready but requires a GPU node (fails fast there).
# Self-built bases share the tag-sourced release version (root treaty §19.5);
# build scripts pass --build-arg VERSION=<bare-semver> from the Makefile-derived
# VERSION. The 0.0.0-dev default only helps non-tag local builds.
ARG VERSION=0.0.0-dev
# Self-built runtime base; harbor builds override with the nested ref
ARG RUNTIME_IMAGE=ghcr.io/nereidai/nf-runtime:${VERSION}
FROM ${RUNTIME_IMAGE}
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='train-optimizer' \
      org.opencontainers.image.description='ONNX quantization and OpenVINO optimization for NereidForge pipelines.' \
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
# NNCF provides static accuracy-aware PTQ for the OpenVINO edge path
# (quantize_with_accuracy_control, gate-1); it constrains the compatible
# openvino release, so install both in one resolution.
RUN pip install --no-cache-dir --timeout 120 --retries 20 \
    "onnxconverter-common>=1.14.0" \
    "openvino>=2024.3.0" \
    "nncf>=2.14.0" \
    "PyYAML>=6.0"
COPY common/ /app/common/
COPY common/nseval.py /app/nseval.py
COPY optimizer/optimize.py /app/optimize.py
WORKDIR /app
# SCRIPT=optimize runs the optimizer entrypoint
ENTRYPOINT ["sh", "-c", "case \"${SCRIPT:-optimize}\" in optimize) exec python /app/optimize.py;; *) exec python /app/optimize.py;; esac"]
