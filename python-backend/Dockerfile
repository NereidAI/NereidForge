# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Build stage
FROM python:3.14-slim AS builder
ARG PIP_INDEX_URL=""
ARG PIP_TRUSTED_HOST=""
ARG APT_MIRROR=""

WORKDIR /app

# Regional apt mirror injected at build time; empty keeps upstream apt sources.
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources \
        && sed -i "s|security.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources; \
    fi

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Injected at build time for regional networks; empty uses pip's default index.
ENV PIP_INDEX_URL=${PIP_INDEX_URL} PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

# Install Python dependencies
COPY python-backend/pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# Production stage
FROM python:3.14-slim AS production
ARG PIP_INDEX_URL=""
ARG PIP_TRUSTED_HOST=""
ARG APT_MIRROR=""

# Release tag shared by every NF self-built image (root treaty §19.5). The
# tag-sourced model bakes the bare SemVer at build time (--build-arg
# VERSION=0.0.1 on a release tag); non-tag builds keep the rolling 0.0.0-dev
# default. Used by app/core/images.py so catalog/job/KFP defaults never drift.
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
ENV NF_IMAGE_TAG=${VERSION}

LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='nereidforge-backend' \
      org.opencontainers.image.description='FastAPI backend and worker for the NereidForge model factory.' \
      org.opencontainers.image.vendor='YIQISOFT' \
      org.opencontainers.image.licenses='Apache-2.0' \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source='https://github.com/NereidAI/NereidForge' \
      org.opencontainers.image.url='https://github.com/NereidAI/NereidForge' \
      org.opencontainers.image.documentation='https://github.com/NereidAI/NereidForge#readme' \
      org.opencontainers.image.authors='NereidAI (https://github.com/NereidAI)'
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md /licenses/

WORKDIR /app

# Regional apt mirror injected at build time; empty keeps upstream apt sources.
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources \
        && sed -i "s|security.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources; \
    fi

# Injected at build time for regional networks; empty uses pip's default index.
ENV PIP_INDEX_URL=${PIP_INDEX_URL} PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and the component announcements consumed by built-in
# image seeding. trainer_profiles resolves these to /app/builders in this image.
COPY python-backend/app/ ./app/
COPY kfp-pipelines/ /app/kfp-pipelines/
COPY builders/trainer-torch/trainer-profile.json /app/builders/trainer-torch/trainer-profile.json
COPY builders/trainer-paddle/trainer-profile.json /app/builders/trainer-paddle/trainer-profile.json
COPY configs/unified/config.rendered.yaml /etc/nereidforge/config.yaml

# Create non-root user; HOME must be writable so the KFP SDK cache (kfp
# cache .cache dir under $HOME) and temp pipeline packages do not fail.
RUN useradd --no-create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
ENV HOME=/tmp
USER appuser

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Default command (overridden by Helm for worker)
# --loop asyncio is REQUIRED: the model-registry SDK 0.3.14 calls
# nest_asyncio2.apply() on init, which cannot patch a uvloop event loop
# ("Can't patch loop of type uvloop.Loop") and makes hub.enabled()=False.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--loop", "asyncio"]
