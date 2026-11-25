# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Self-built PyTorch XPU runtime for Intel discrete GPUs (Battlemage / xe KMD).
# Modeled on pytorch .ci/docker/common/install_xpu.sh, but a *runtime* image:
# host provides the xe/i915 KMD; the image provides userspace NEO + Level Zero
# plus the prebuilt torch XPU wheel (no from-source torch build).
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='trainer-xpu-base' \
      org.opencontainers.image.description='Intel XPU PyTorch runtime base for NereidForge trainers.' \
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

ENV DEBIAN_FRONTEND=noninteractive
# Prebuilt torch XPU wheels + Intel userspace stack use a pinned build contract;
# an unversioned install can silently select a newer upstream torch and break
# every dependent image with a different compatibility constraint.
# Generic network endpoints use pristine upstream defaults when empty. The
# specialized XPU index is mandatory because generic PyPI torch is not an XPU
# build; callers inject its official or mirrored endpoint at build time.
ARG TORCH_XPU_INDEX=""
ARG TORCH_VERSION=2.14.0
ARG PYPI_INDEX=""
ARG PIP_TRUSTED_HOST=""

# Regional apt mirror for the base Ubuntu packages; empty keeps upstream apt.
ARG APT_MIRROR=""
RUN if [ -n "$APT_MIRROR" ]; then sed -i "s@//.*archive.ubuntu.com@//$APT_MIRROR@g; s@//security.ubuntu.com@//$APT_MIRROR@g" \
        /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true; fi

# --- Intel GPU userspace compute stack (client / kobuk PPA) ---
# Battlemage (Xe2) needs the newer client stack (NEO >= 26.x); the noble
# "unified" repo (25.18) aborts in the Level-Zero command encoder on B580.
# This mirrors pytorch .ci/docker/common/install_xpu.sh XPU_DRIVER_TYPE=client
# and matches the host stack (26.18). No regional mirror carries these debs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gnupg wget software-properties-common \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libze-intel-gpu1 libze1 intel-opencl-icd clinfo intel-ocloc \
        intel-metrics-discovery intel-gsc xpu-smi \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools \
        libva-glx2 va-driver-all vainfo \
    && rm -rf /var/lib/apt/lists/*

# --- Python 3.12 + prebuilt torch XPU (indexes injected at build time) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/xpu \
    && if [ -n "$PYPI_INDEX" ]; then \
           /opt/xpu/bin/pip install --no-cache-dir \
               --index-url "${PYPI_INDEX}" ${PIP_TRUSTED_HOST:+--trusted-host "${PIP_TRUSTED_HOST}"} --upgrade pip; \
       else \
           /opt/xpu/bin/pip install --no-cache-dir --upgrade pip; \
       fi
# The XPU wheels only exist on the specialized wheel index. Install and verify
# the local-version marker in one layer so a wrong wheel can never be published.
RUN if [ -z "$TORCH_XPU_INDEX" ]; then \
        echo "ERROR: TORCH_XPU_INDEX is required for the XPU base" >&2; exit 1; \
    fi; \
    /opt/xpu/bin/pip install --no-cache-dir --timeout 120 --retries 10 \
        --index-url "${TORCH_XPU_INDEX}" ${PYPI_INDEX:+--extra-index-url "${PYPI_INDEX}"} \
        "torch==${TORCH_VERSION}+xpu" \
    && /opt/xpu/bin/python -c "import importlib.metadata, torch; \
        expected = '${TORCH_VERSION}+xpu'; \
        assert torch.__version__ == expected, torch.__version__; \
        assert importlib.metadata.version('torch') == expected, importlib.metadata.version('torch'); \
        print('XPU torch verified:', torch.__version__)"

# Toolchain + Level Zero headers: triton-XPU JIT-compiles kernels at *runtime*
# (bitsandbytes XPU 4-bit / 8-bit optimizers go through triton) and needs a C/C++
# compiler plus level_zero/ze_api.h inside the container, not only at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make python3-dev libze-dev libze-intel-gpu-dev \
    && rm -rf /var/lib/apt/lists/*
ENV CC=gcc CXX=g++

ENV PATH=/opt/xpu/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
CMD ["bash"]
