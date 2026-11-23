# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Evaluator image: real detection/classification metrics over ONNX models
# with datasets from seaweedfs. Also hosts the gate/baseline logic used by
# the optimization pipeline (same evaluate.py entrypoint).
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
      org.opencontainers.image.title='train-evaluator' \
      org.opencontainers.image.description='Model evaluation and optimization gates for NereidForge pipelines.' \
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
RUN pip install --no-cache-dir "onnx>=1.14,<2" "openvino>=2024.3.0" "PyYAML>=6.0"
COPY common/ /app/common/
COPY common/model_package.py /app/model_package.py
COPY common/nseval.py /app/nseval.py
COPY evaluator/evaluate.py /app/evaluate.py
WORKDIR /app
# SCRIPT=evaluate runs the evaluator entrypoint; package jobs invoke
# `python /app/model_package.py --spec-uri ...` explicitly.
ENTRYPOINT ["sh", "-c", "exec python /app/evaluate.py"]
