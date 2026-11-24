#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${IMMUTABLE:?IMMUTABLE is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"
: "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"
: "${PLATFORMS:?PLATFORMS is required}"

extra="${EXTRA_BUILD_ARGS:-}"
extra="${extra//@IMMUTABLE@/$IMMUTABLE}"
annotation_levels="manifest"
if [[ "$PLATFORMS" == *,* ]]; then
  annotation_levels="index,manifest"
fi

{
  echo "build_args<<BUILD_ARGS_EOF"
  echo "VERSION=$IMMUTABLE"
  echo "GIT_SHA=$GITHUB_SHA"
  echo "BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ -n "$extra" ]]; then printf '%s\n' "$extra"; fi
  echo "BUILD_ARGS_EOF"

  echo "annotations<<ANNOTATIONS_EOF"
  printf '%s:org.opencontainers.image.source=https://github.com/NereidAI/NereidForge\n' "$annotation_levels"
  printf '%s:org.opencontainers.image.version=%s\n' "$annotation_levels" "$IMMUTABLE"
  printf '%s:org.opencontainers.image.revision=%s\n' "$annotation_levels" "$GITHUB_SHA"
  printf '%s:org.opencontainers.image.vendor=YIQISOFT\n' "$annotation_levels"
  printf '%s:org.opencontainers.image.licenses=Apache-2.0\n' "$annotation_levels"
  echo "ANNOTATIONS_EOF"
} >> "$GITHUB_OUTPUT"
