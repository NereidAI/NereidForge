#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# check-legal-copies.sh - verify that every non-root Docker build context
# carries byte-identical copies of the repository's LICENSE and NOTICE.
# These copies make the legal texts available to COPY without changing any
# Docker build context. Missing files and content drift fail the gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONTEXTS=(
  app/frontend
  app/kfp-pipelines/docker
  app/builders
  app/builders/inference-model-server-openvino-xpu
  app/builders/ml-backend
  app/builders/ml-backend-custom
  app/builders/notebook-base
  app/builders/trainer-xpu
  app/optimizer-plugins/tensorrt
)

status=0
checked=0
for context in "${CONTEXTS[@]}"; do
  for name in LICENSE NOTICE; do
    source_file="$REPO_ROOT/$name"
    copy="$REPO_ROOT/$context/$name"
    checked=$((checked + 1))
    if [[ ! -f "$copy" ]]; then
      printf 'MISSING  %s\n' "$context/$name"
      status=1
      continue
    fi
    if ! diff -q "$source_file" "$copy"; then
      printf 'DRIFT  %s\n' "$context/$name"
      status=1
    fi
  done
done

if [[ "$status" -eq 0 ]]; then
  printf 'Legal copies: %d files match the repository LICENSE and NOTICE\n' "$checked"
fi
exit "$status"
