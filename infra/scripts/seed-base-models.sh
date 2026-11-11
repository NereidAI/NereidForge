#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Seed the S3 base-model pool with stock checkpoints for offline installations.
#
# Trainer images do not bake in base weights: training jobs download them from
# s3://<bucket>/base-models/cv/<name>. This script uploads every .pt/.pth file
# from a local directory to that prefix. Pool rows are backfilled automatically
# by the API startup reconcile (or restart the API deployments after uploading).
#
# Usage:
#   S3_ENDPOINT=https://object-store.example.com:8333 \
#   S3_BUCKET=my-models S3_ACCESS_KEY=<access-key> S3_SECRET_KEY=<secret-key> \
#     seed-base-models.sh ./weights
set -euo pipefail

SRC_DIR="${1:?usage: seed-base-models.sh <dir-with-.pt-files>}"
: "${S3_ENDPOINT:?S3_ENDPOINT required, e.g. https://object-store.example.com:8333}"
: "${S3_BUCKET:=nereidforge}"
: "${S3_ACCESS_KEY:?S3_ACCESS_KEY required}"
: "${S3_SECRET_KEY:?S3_SECRET_KEY required}"

command -v aws >/dev/null 2>&1 || { echo "aws CLI is required" >&2; exit 1; }

export AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY"
export AWS_DEFAULT_REGION="us-east-1"

shopt -s nullglob
files=("$SRC_DIR"/*.pt "$SRC_DIR"/*.pth)
[ ${#files[@]} -gt 0 ] || { echo "no .pt/.pth files under $SRC_DIR" >&2; exit 1; }

for f in "${files[@]}"; do
  echo "uploading $(basename "$f") -> s3://$S3_BUCKET/base-models/cv/"
  aws --endpoint-url "$S3_ENDPOINT" s3 cp "$f" "s3://$S3_BUCKET/base-models/cv/"
done

echo "done. Restart the API so pool rows are backfilled:"
echo "  kubectl rollout restart deploy/nereidforge-api -n nereidforge"
