# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# AMD ROCm variant of the LLM/VLM trainer (separate per-vendor image, ADR-012).
#
# Targets RDNA consumer GPUs officially supported by ROCm (RX 7900 gfx1100,
# 7800 XT gfx1101, 7600 gfx1102; RX 9000 gfx1200/1201). ROCm PyTorch exposes
# the CUDA API through HIP, so torch.cuda.* and LLaMA-Factory run unchanged
# (NCCL calls become RCCL).
#
# The ROCm userspace base cannot come from a pip index (ROCm system libs); on
# an air-gapped site the node containerd mirror resolves the upstream ref, or
# the operator can pre-stage it under a site registry and override BASE_IMAGE
# at build time (root treaty section 18; no internal host is committed here):
#   docker pull rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0
#   docker tag  rocm/pytorch:rocm6.3_..._2.4.0 <registry>/rocm/pytorch:rocm6.3
#   docker push <registry>/rocm/pytorch:rocm6.3
# That base ships torch 2.4.0 + ROCm 6.3 (py3.12, ubuntu24.04). LLaMA-Factory
# 0.9.5 runs on that torch (its floor is torch>=2.4); torch is NOT reinstalled
# here. All other pip packages use the injected pip index (empty = default
# PyPI).
# Cards without prebuilt gfx kernels may need HSA_OVERRIDE_GFX_VERSION=11.0.0
# (set it in the AMD accelerator profile env).
# The full upstream tag is the only public default (no short rocm6.3 tag
# exists upstream); site builds may override BASE_IMAGE with a staged copy.
ARG BASE_IMAGE=rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0
FROM ${BASE_IMAGE}
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='trainer-llm-rocm' \
      org.opencontainers.image.description='AMD ROCm LLM and VLM fine-tuning trainer for NereidForge.' \
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
COPY sdk /opt/nf-trainer-sdk
RUN pip install --no-cache-dir "llamafactory[torch,metrics]==0.9.5" \
    && pip install --no-cache-dir \
        "boto3>=1.34" "pyyaml>=6.0" "sentencepiece" "tiktoken" "protobuf" \
        /opt/nf-trainer-sdk && rm -rf /opt/nf-trainer-sdk
COPY trainer-llm-rocm/llm_train.py /app/llm_train.py
WORKDIR /app
ENTRYPOINT ["python", "/app/llm_train.py"]
