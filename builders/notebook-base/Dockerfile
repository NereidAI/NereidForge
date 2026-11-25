# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# NereidForge notebook base image: JupyterLab workbench + a self-contained ML
# stack so Kale/KFP notebook cells and the interactive workbench can import
# the common data-science / CV packages without any runtime network access.
#
# Builds on the mirrored official Jupyter image (conda, Ubuntu 24.04, user
# jovyan) so the jupyter launch args used by the notebook service keep working
# unchanged; we only add the Python packages on top.
# NO ultralytics/YOLO - AGPL-3.0 removed
ARG JUPYTER_BASE_IMAGE=quay.io/jupyter/base-notebook:python-3.12.9
FROM ${JUPYTER_BASE_IMAGE}
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='prediction-jupyter-base' \
      org.opencontainers.image.description='JupyterLab workbench and Kale runtime for NereidForge notebooks.' \
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
# CPU torch wheel index injected at build time; empty uses the default index.
ARG TORCH_CPU_INDEX=""
ARG DEBIAN_FRONTEND=noninteractive

USER root

# Regional apt mirror injected at build time; empty keeps the base image's
# upstream apt sources.
ARG APT_MIRROR=""
RUN if [ -n "$APT_MIRROR" ]; then sed -i "s|//.*archive.ubuntu.com|//$APT_MIRROR|g;s|//security.ubuntu.com|//$APT_MIRROR|g" \
    /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true; fi \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
     libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 git curl \
  && rm -rf /var/lib/apt/lists/*

ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=20

# Core data-science stack + S3 access (read/write model artifacts).
RUN /opt/conda/bin/pip install --no-cache-dir --timeout 600 --retries 20 \
    "numpy>=1.26" \
    "pandas>=2.1" \
    "scipy>=1.11" \
    "matplotlib>=3.8" \
    "seaborn>=0.13" \
    "scikit-learn>=1.3" \
    "pillow>=10.0" \
    "boto3>=1.34"

# CPU PyTorch (kept separate from the pip index bucket on purpose: a regional
# PyPI mirror may not carry torch wheels reliably; install from the injected
# CPU wheel index exactly like the trainer images do). Empty TORCH_CPU_INDEX
# falls back to a plain pip install from the default index.
RUN if [ -n "$TORCH_CPU_INDEX" ]; then \
        /opt/conda/bin/pip install --no-cache-dir --timeout 600 --retries 20 \
            --index-url "${TORCH_CPU_INDEX}" \
            torch torchvision; \
    else \
        /opt/conda/bin/pip install --no-cache-dir --timeout 600 --retries 20 \
            torch torchvision; \
    fi

# CV stack: opencv headless (no libGL dependency beyond what we apt-installed)
# + PP-YOLOE (PaddleDetection) - Apache-2.0
# NO ultralytics/YOLO - AGPL-3.0 removed
RUN /opt/conda/bin/pip install --no-cache-dir --timeout 600 --retries 20 \
    "opencv-python-headless>=4.8" \
    "paddlepaddle>=3.0,<4" "paddlex>=3.0.0"

# Kale runtime engine for NF Notebook tasks (ADR-013 Case 2): this single
# dual-use image is both the JupyterLab workbench AND the Kale KFP step base.
# kubeflow-kale (requires_python >=3.11, ok on this 3.12 base) compiles/swims
# the runs; kfp is pinned to the cluster's KFP 2.17 so the executor bootstrap
# (pip-clone) is a no-op after our submit-time strip -> steps never need
# run-time network.
# backports.tarfile repairs the broken conda setuptools: the base ships
# backports.functools-lru-cache which shadows the `backports` namespace, so
# setuptools cannot import backports.tarfile and legacy-setup.py packages
# (kale/kfp metadata) fail to build without it.
RUN /opt/conda/bin/pip install --no-cache-dir --timeout 600 --retries 20 "backports.tarfile>=1.0.0" \
 && /opt/conda/bin/pip install --no-cache-dir --timeout 600 --retries 20 \
    "kubeflow-kale==2.2.0" \
    "kfp==2.17.0"

# Back to the non-root notebook user (jupyter image convention).
USER jovyan
WORKDIR /home/jovyan
