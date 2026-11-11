# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# The PaddlePaddle dependency ceiling requires Python 3.11; callers may inject
# a mirrored Python base without changing the tracked public coordinate.
ARG PYTHON_BASE_IMAGE=python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='ml-backend-custom' \
      org.opencontainers.image.description='Custom Label Studio ML backend runtime for NereidForge annotation workflows.' \
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
# CPU torch wheel index injected at build time; empty leaves no extra index.
ARG TORCH_CPU_INDEX=""
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    WORKERS=1 \
    THREADS=4 \
    MODEL_ROOT=/app/models \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Regional pip index injected at build time (empty = default PyPI); the CPU
# torch wheel index is exposed through PIP_EXTRA_INDEX_URL for the torch step.
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    PIP_EXTRA_INDEX_URL=${TORCH_CPU_INDEX} \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=50

COPY requirements-base.txt .
# Split install: first install label-studio packages with longer timeout, then others.
# rq is explicit: the PyPI 1.0.9 wheel omits it from metadata but api.py imports rq.exceptions
RUN pip install --no-cache-dir --timeout 1800 --retries 50 \
    "gunicorn==23.0.0" "joblib==1.4.2" "rq==2.12.0" "label-studio-ml==1.0.9" "label-studio-sdk==2.1.2"

# Install CPU torch (via the extra index when injected), then the remaining
# packages from the main pip index.
RUN pip install --no-cache-dir --timeout 1800 --retries 50 \
    torch torchvision torchaudio

# Install ONNX Runtime, PaddleOCR, and their supporting dependencies.
RUN pip install --no-cache-dir --timeout 1800 --retries 50 \
    "onnxruntime>=1.17" "opencv-python-headless>=4.8" \
    "paddlepaddle>=3.0,<4" "paddleocr>=3.7,<4"

# Pre-warm the PP-OCR model artifacts.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='ch')" || true

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 1800 --retries 50 -r requirements.txt

COPY . ./

CMD ["/app/start.sh"]
