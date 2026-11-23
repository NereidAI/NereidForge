# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Build stage
FROM node:24-alpine AS builder

# Tag-sourced version model (root treaty §19.5): release builds bake the bare
# SemVer via --build-arg; non-tag builds keep the rolling 0.0.0-dev default.
ARG VITE_APP_VERSION=0.0.0-dev
ENV VITE_APP_VERSION=${VITE_APP_VERSION}

# Release tag shared by every self-built image; drives the UI-selected image
# refs so they can never drift from the backend catalog. Image refs themselves
# are hardwired to public ghcr.io/nereidai flat names (root treaty §18);
# air-gapped pulls resolve via the node containerd mirror.
ARG VITE_IMAGE_TAG=0.0.0-dev
ENV VITE_IMAGE_TAG=${VITE_IMAGE_TAG}

# Dataset upload guards are build-time configuration because Vite emits a
# static bundle. Local development falls back to 200 files and 1 GiB when these
# arguments are unset; deployment builds can override both independently.
ARG VITE_DATASET_UPLOAD_MAX_FILES
ENV VITE_DATASET_UPLOAD_MAX_FILES=${VITE_DATASET_UPLOAD_MAX_FILES}

ARG VITE_DATASET_UPLOAD_MAX_BYTES
ENV VITE_DATASET_UPLOAD_MAX_BYTES=${VITE_DATASET_UPLOAD_MAX_BYTES}

WORKDIR /app

COPY package*.json ./
RUN npm ci --ignore-scripts

COPY . .
RUN npx vite build --mode production

# Production stage
FROM nginx:alpine AS production
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""
LABEL license='SPDX-License-Identifier: Apache-2.0' \
      copyright='Copyright (c) 2026: YIQISOFT' \
      org.opencontainers.image.title='nereidforge-frontend' \
      org.opencontainers.image.description='Web interface for the NereidForge model factory.' \
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

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
