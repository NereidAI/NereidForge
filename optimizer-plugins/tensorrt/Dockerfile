# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/tensorrt:24.06-py3
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='opt-tensorrt' \
      org.opencontainers.image.description='TensorRT optimizer plugin for NereidForge model conversion and benchmarking.' \
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

# Install additional dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    onnx \
    onnxruntime \
    boto3 \
    pyyaml \
    pytest

# Set working directory
WORKDIR /workspace

# Copy plugin files
COPY manifest.yaml /workspace/
COPY plugin.py /workspace/
COPY converter.py /workspace/
COPY precision.py /workspace/
COPY benchmark.py /workspace/
COPY tests/ /workspace/tests/

# Set entrypoint
ENTRYPOINT ["python3", "plugin.py"]
