#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Apply the NereidForge customization patch set (patches/) and local-only
# additions (additions/) on top of a pristine kubeflow/community-distribution
# checkout pinned to the 26.03.1 tag.
#
# Usage:
#   ./apply.sh [MANIFEST_DIR]      # default: ../kubeflow-manifests
#
# The result must equal infra/third_party/kubeflow-manifests.
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_DIR="${1:-$SCRIPT_DIR/../kubeflow-manifests}"

cd "$MANIFEST_DIR"

echo "==> applying ${SCRIPT_DIR}/patches/*/patch.diff"
for pf in "$SCRIPT_DIR"/patches/*/patch.diff; do
  printf '  %-24s ' "$(basename "$(dirname "$pf")")"
  if patch -p1 -s -N < "$pf" > /dev/null 2>&1; then
    echo "applied"
  else
    echo "already-applied-or-skipped"
  fi
done

# Replace upstream static credentials without recording their values in the
# patch ledger. Installers substitute these placeholders only in temporary or
# rendered output; neither upstream nor lab credentials become NF public data.
DEX_PASSWORDS="$MANIFEST_DIR/common/dex/base/dex-passwords.yaml"
if [ -f "$DEX_PASSWORDS" ]; then
  sed -i -E \
    's|^([[:space:]]*DEX_USER_PASSWORD:).*$|\1 "${DEX_STATIC_PASSWORD_HASH}"|' \
    "$DEX_PASSWORDS"
fi
KFP_ARTIFACT_SECRET="$MANIFEST_DIR/applications/pipeline/upstream/third-party/seaweedfs/base/seaweedfs/mlpipeline-minio-artifact-secret.yaml"
if [ -f "$KFP_ARTIFACT_SECRET" ]; then
  sed -i -E \
    -e 's|^([[:space:]]*accesskey:).*$|\1 "${SEAWEEDFS_S3_ACCESS_KEY}"|' \
    -e 's|^([[:space:]]*secretkey:).*$|\1 "${SEAWEEDFS_S3_SECRET_KEY}"|' \
    "$KFP_ARTIFACT_SECRET"
fi

echo "==> copying additions/"
cd "$SCRIPT_DIR/additions"
find . -type f | while IFS= read -r f; do
  dst="$MANIFEST_DIR/$f"
  mkdir -p "$(dirname "$dst")"
  cp -a "$f" "$dst"
done

echo "==> removals/"
if [ -f "$SCRIPT_DIR/removals.txt" ]; then
  while IFS= read -r rel; do
    [ -z "$rel" ] && continue
    rm -rf "$MANIFEST_DIR/$rel"
  done < "$SCRIPT_DIR/removals.txt"
fi
echo "==> done"