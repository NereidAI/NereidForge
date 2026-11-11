# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# NereidForge Paddle vision trainer — CPU (PaddleX detection, Apache-2.0).
#
# Context MUST be app/builders (the trainer sources plus the shared SDK package).
# One image per vendor; never an all-in-one (ADR-012/014).
#
# No torch: this is the Paddle stack, and the earlier pin
# (torch==2.14.0+cpu / torchvision==0.29.0+cpu) does not exist on the mirror —
# torch is not a PaddleX dependency, so installing it only created a broken
# build and a silent CPU fallback. See ADR-014 D-01 and pit 149.
#
# No weights are baked in: PaddleX fetches permissive (Apache-2.0) pretrained
# backbones at run time, or the job passes `base_weights_uri` (S3).
ARG PYTHON_BASE_IMAGE=python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='trainer-paddle-cpu' \
      org.opencontainers.image.description='CPU PaddleDetection training and model export for NereidForge.' \
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
ARG PADDLE_VERSION=3.0.0
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}
ENV PYTHONUNBUFFERED=1
ENV NF_DEVICE=cpu

# Switch Debian apt to the injected regional mirror, then install headless CV
# system libs; an empty APT_MIRROR keeps upstream apt sources.
RUN if [ -n "$APT_MIRROR" ]; then sed -i "s@//deb.debian.org@//$APT_MIRROR@g; s@//security.debian.org@//$APT_MIRROR@g" \
        /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*

# PaddleX needs a PaddlePaddle runtime but never installed one itself, which is
# why the old image could not import `paddlex`. CPU wheel comes from PyPI.
# Paddle is pinned to 3.0.0 on purpose: 3.1+ ships the new PIR serializer by
# default and the PaddleX3.0 PaddleDetection repo's dy2static export crashes on
# it (InvalidType in lcnet.py), while paddle2onnx 2.1.0 fails to parse the PIR
# output of PicoDet ("elementwise MUST be ir::FloatAttribute"). The 3.0.0
# combination with paddle2onnx 2.0.2rc3 (PaddleX's own pin in requirements)
# is the verified train -> PIR export -> ONNX chain.
# numpy/opencv pins follow PaddleDetection's own requirements (its setup pins
# numpy 1.26.x and opencv-contrib-python 4.10; headless + contrib both ship a
# `cv2`, so we install only contrib to avoid a two-cv2 conflict).
RUN pip install --no-cache-dir --timeout 300 --retries 20 \
        "paddlepaddle==${PADDLE_VERSION}" \
    && pip install --no-cache-dir --timeout 300 --retries 20 \
        "paddlex==3.7.2" "paddle2onnx==2.0.2rc3" \
        "onnx>=1.14,<2" "onnxruntime>=1.17" \
        "opencv-contrib-python==4.10.0.84" "pillow>=10.0" "pyyaml>=6.0" \
        "numpy==1.26.4" "boto3>=1.34"

# PaddleX 3.7.2 ships the repo-API training configs only in the sdist; the
# wheel's register.py points at repo_apis/PaddleDetection_api/configs/*.yaml
# which the wheel does not contain, so PaddleModel(model_name=...) fails with
# FileNotFoundError on a plain pip install. Restore just that tree from the
# matching sdist (same index/mirror policy as every other pip call).
RUN pip download --no-deps --no-binary :all: "paddlex==3.7.2" -d /tmp/pdx-sdist \
    && api_dir="$(python -c 'import os,paddlex;print(os.path.join(os.path.dirname(paddlex.__file__),"repo_apis"))')" \
    && tar xzf /tmp/pdx-sdist/paddlex-*.tar.gz --strip-components=3 -C "$api_dir" \
        --wildcards "*/paddlex/repo_apis/PaddleDetection_api/configs" \
    && rm -rf /tmp/pdx-sdist

# PaddleX training/evaluate/export run against the PaddleDetection repository
# (repo API, `paddlex.repo_apis.base.PaddleModel` — `paddlex.create_model`
# builds an inference predictor only). Baking the repo at build time keeps the
# image self-contained for offline jobs; it is Apache-2.0 code installed via
# PaddleX's own repo manager (tar from the PaddleX CDN + its pip package).
# gcc is required only because PaddleX probes gcc unconditionally before
# deciding to skip the CUDA rotated-Det custom ops, so purge it in the same
# layer once the repo is installed.
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && mkdir -p /usr/local/lib/python3.11/site-packages/paddlex/repo_manager/repos \
    && python -c "from paddlex.repo_manager import setup; setup(['PaddleDetection'])" \
    && apt-get purge -y gcc && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && python -c "from paddlex.repo_apis.base import PaddleModel; \
m = PaddleModel(model_name='PicoDet-S'); print('repo API OK:', type(m).__name__)"

# The SDK is the only shared contract (ADR-016); the build context is
# app/builders so the package and the trainer sources ship together.
COPY sdk /opt/nf-trainer-sdk
RUN pip install --no-cache-dir /opt/nf-trainer-sdk && rm -rf /opt/nf-trainer-sdk

# PaddleX stores derived artifacts under /root/.paddlex; make it writable so the
# image also works with a read-only-ish root or a non-root runAsUser.
RUN mkdir -p /root/.paddlex/PaddleX && chmod -R 777 /root/.paddlex

COPY trainer-paddle/common.py trainer-paddle/dataset.py trainer-paddle/models.py \
     trainer-paddle/trainer_metrics.py trainer-paddle/train.py trainer-paddle/evaluate.py \
     trainer-paddle/export.py trainer-paddle/catalog.py trainer-paddle/nf_profile.py \
     trainer-paddle/trainer-profile.json /app/
WORKDIR /app

# SCRIPT selects the entrypoint (train is the default). The KFP TrainJob sets
# the command explicitly, so a shell entrypoint cannot swallow it.
ENTRYPOINT ["sh", "-c", "case \"${SCRIPT:-train}\" in evaluate) exec python /app/evaluate.py;; export) exec python /app/export.py;; catalog) exec python /app/catalog.py;; profile) exec python /app/nf_profile.py;; *) exec python /app/train.py;; esac"]
