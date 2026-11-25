# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# NereidForge OpenVINO Model Server — Intel XPU variant (Arc / Battlemage).
#
# The standard NF OVMS image mirrors the upstream Ubuntu-24.04 based
# openvino/model_server and stays CPU-only (no compute userspace). This layer
# adds the client/kobuk Intel GPU compute stack (NEO >= 26) so the same OVMS
# binary can run inference on Arc dGPUs via the OpenVINO GPU plugin.
#
# Device is selected at deploy time with `ovms --target_device GPU` (the ISVC
# manifest adds it when the user picks Intel GPU); the image itself stays
# usable on CPU hosts as well. Artifacts are unchanged: one ONNX/IR serves
# both devices.
#
# The kobuk-team PPA has no regional mirror: it is always used directly.
# APT_MIRROR only rewrites the Ubuntu base repos; an empty value keeps
# pristine upstream sources.
ARG BASE_IMAGE=openvino/model_server:2025.1
FROM ${BASE_IMAGE}
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='inference-model-server-openvino-xpu' \
      org.opencontainers.image.description='OpenVINO Model Server with Intel XPU userspace for NereidForge inference.' \
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

ARG APT_MIRROR=""
USER root
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s@//.*archive.ubuntu.com@//$APT_MIRROR@g; s@//security.ubuntu.com@//$APT_MIRROR@g" \
            /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true; \
    fi

# Minimal compute userspace for Battlemage (Xe2): Level Zero + OpenCL ICD.
# NEO >= 26 is mandatory — the noble "unified" repo (25.18) aborts in the
# Level-Zero command encoder on B580 (same stack choice as trainer-xpu).
# libdrm-intel1 from noble-updates (>= 2.4.122) aligns with the validated
# trainer-xpu userspace on the new "xe" KMD (base image ships stock 2.4.120).
# Note: it is NOT what makes non-root device enumeration work — see README
# (uid 5000 still enumerates zero; GPU deploys run as root).
# NEO >= 26 is mandatory for Battlemage regardless; the kobuk PPA keeps only
# the newest 26.x build, so do not pin an exact NEO version.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gnupg wget software-properties-common \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libdrm-intel1 \
        libze-intel-gpu1 libze1 intel-opencl-icd clinfo intel-ocloc \
    && rm -rf /var/lib/apt/lists/* \
    && test -f /usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so

# Default user stays the upstream non-root ovms uid; the KServe manifest runs
# GPU deployments as root (runAsUser 0) because NEO enumerates zero devices
# for uid 5000 — see README and nf-backend pit 174.
USER 5000
# No ENTRYPOINT/CMD override: the ISVC manifest supplies the model download +
# ovms command, identical to the CPU image except --target_device GPU.
