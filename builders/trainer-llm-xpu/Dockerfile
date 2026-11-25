# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Intel XPU variant of the LLM/VLM trainer (separate per-vendor image, ADR-012).
# Native torch.xpu stack validated on Arc B580 (Battlemage): transformers +
# peft + bitsandbytes XPU 4-bit QLoRA, bf16, single card. LLaMA-Factory 0.9.4
# does not support torch.xpu, so this vendor uses its own entrypoint
# (llm_train.py here), NOT the LLaMA-Factory one used by cu/rocm.
#
# xpu-base already provides torch==2.14.0+xpu (the specialized XPU wheel
# installed from the injected wheel index), torchvision-xpu is not needed for
# LLM, plus the gcc/g++/Level Zero headers for triton XPU JIT and NEO >= 26
# userspace. All pip packages below use the injected pip index (empty = PyPI).
# Tag-sourced model (root treaty §19.5): release builds pass the bare SemVer
# explicitly; the 0.0.0-dev default covers non-tag local builds.
ARG BASE_IMAGE=ghcr.io/nereidai/trainer-xpu-base:0.0.0-dev
FROM ${BASE_IMAGE}
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='trainer-llm-xpu' \
      org.opencontainers.image.description='Intel XPU LLM and VLM fine-tuning trainer for NereidForge.' \
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

# Build network endpoints injected at build time for regional networks; empty
# values reproduce pristine upstream behavior (default PyPI / Hugging Face).
ARG PIP_INDEX_URL=""
ARG PIP_TRUSTED_HOST=""
ARG HF_ENDPOINT=""
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}
ENV HF_ENDPOINT=${HF_ENDPOINT}
# --no-deps is NOT used: bitsandbytes/peft pull compatible deps but torch is
# already satisfied by the base so pip keeps the +xpu build. transformers is
# pinned to the validated 4.x line (5.x changed Trainer/tokenizer APIs and has
# not been certified on torch.xpu + the B580 QLoRA stack yet).
COPY sdk /opt/nf-trainer-sdk
RUN pip install --no-cache-dir --timeout 120 --retries 10 \
        "boto3>=1.34" "pyyaml>=6.0" \
        "transformers>=4.46,<5" "peft>=0.13" "accelerate>=0.34" "datasets>=3.0,<4" \
        "bitsandbytes==0.50.2" "numpy<2.3" \
        "sentencepiece" "tiktoken" "protobuf" "safetensors" "einops" \
        "pillow" "requests" /opt/nf-trainer-sdk && rm -rf /opt/nf-trainer-sdk
COPY trainer-llm-xpu/llm_train.py /app/llm_train.py
COPY trainer-llm-xpu/vlm_train.py /app/vlm_train.py
WORKDIR /app
ENTRYPOINT ["python", "/app/llm_train.py"]
