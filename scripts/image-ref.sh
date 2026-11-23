#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# image-ref.sh - print the full public reference of a self-built image.
# Used by the Makefile so image paths are defined once. All references use
# the single public scheme (root treaty section 18):
#   ghcr.io/nereidai/<flat-name>:<tag>
#
# Usage: image-ref.sh <category/name> [tag]
#   image-ref.sh trainer/torch-cpu
#     -> ghcr.io/nereidai/trainer-torch-cpu:0.0.0-dev (off-tag; bare SemVer on a v* tag checkout)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/config.sh"
case "${1:-}" in
  --env) echo "${!2:?usage: image-ref.sh --env VAR_NAME}" ;;
  *) nf_ref "$@" ;;
esac
