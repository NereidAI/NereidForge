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
      org.opencontainers.image.title='ml-backend' \
      org.opencontainers.image.description='PP-YOLOE and ONNX pre-annotation backend for NereidForge and Label Studio.' \
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
# CPU torch wheel index injected at build time for regional networks; empty
# makes pip resolve torch from the default public index.
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
# torch wheel index is added only in the torch install step below, so a global
# extra index never slows down or stalls the other packages.
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=50

COPY requirements-base.txt .
# label-studio-ml 2.x (label_studio_ml.response.ModelResponse, Flask 2
# `app.post` API our _wsgi.py targets) is NOT on PyPI: newest PyPI 1.0.9
# predates those APIs. Install its declared deps from PyPI, then install the
# vendored git-master snapshot --no-deps (vendor/label-studio-ml). rq rides
# along: the proven legacy image had it and 1.0.9's api.py needs rq.exceptions.
COPY vendor/label-studio-ml /tmp/vendor-label-studio-ml
RUN pip install --no-cache-dir --timeout 120 --retries 50 \
    "gunicorn==23.0.0" "joblib==1.4.2" "rq==2.12.0" \
    "Flask>2.3" "colorama~=0.4" "requests~=2.31" "semver~=3.0.2" \
    "label-studio-sdk==2.1.2" \
    && pip install --no-cache-dir --no-deps /tmp/vendor-label-studio-ml \
    && rm -rf /tmp/vendor-label-studio-ml

# CPU torch/torchvision/torchaudio: the injected CPU wheel index is added as
# an extra index when set (plain torch on the main index is the CUDA build);
# empty resolves the packages from the default public index only.
RUN if [ -n "$TORCH_CPU_INDEX" ]; then \
        pip install --no-cache-dir --timeout 300 --retries 50 \
            --extra-index-url "${TORCH_CPU_INDEX}" \
            torch torchvision torchaudio; \
    else \
        pip install --no-cache-dir --timeout 300 --retries 50 \
            torch torchvision torchaudio; \
    fi
# onnxruntime powers the generic ONNX branch (DETR/FRCNN/U-Net exports);
# paddle 3.x + paddleocr power the model_ocr branch (all-in-one image)
RUN pip install --no-cache-dir --timeout 300 --retries 50 \
    "onnxruntime>=1.17" "opencv-python-headless>=4.8" \
    "paddlepaddle>=3.0,<4" "paddleocr>=3.7,<4"
# prewarm the default PP-OCR mobile models so offline clusters never
# fetch them at runtime
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='ch')" || true
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 50 -r requirements.txt

COPY . ./

# No model weights are baked into the image. The pod starts with zero models
# loaded; every detector/segmenter/classifier/SAM is pulled on demand from S3
# (s3://bucket/models/...) the first time a request selects it, then held in a
# bounded LRU cache. Users upload and choose models in the NF UI.
# MODEL_ROOT (/app/models) is the writable runtime cache for those downloads.

CMD ["/app/start.sh"]
