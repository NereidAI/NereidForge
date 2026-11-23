#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Dev worker: consumes async tasks (dataset_import, preannotate, ...) from the same
# Postgres + S3 as the API. Must run on the host alongside run-dev.sh (the in-cluster
# worker pod connects to a different DB/S3 and will NOT process dev tasks).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/infra/scripts/lib/common.sh"

if [[ -f "$NF_DEV_ENV_FILE" ]]; then
  set -a
  source "$NF_DEV_ENV_FILE"
  set +a
fi

export NEREIDFORGE_ENV=dev
# Dev shares the prod data plane (same bucket + DB as the API).
export NEREIDFORGE_STORAGE_BUCKET="${NEREIDFORGE_STORAGE_BUCKET:-nereidforge}"
export NEREIDFORGE_DATABASE_NAME="${NEREIDFORGE_DATABASE_NAME:-nereidforge}"
ensure_development_jwt_secret
if [[ -z "${NEREIDFORGE_DATABASE_PASSWORD:-}" && -n "${POSTGRES_PASSWORD:-}" ]]; then
  export NEREIDFORGE_DATABASE_PASSWORD="$POSTGRES_PASSWORD"
fi

# Same host->cluster port-forwards the API needs (S3 :8833, KFP :18888, Hub :8889).
bash infra/scripts/dev-port-forward.sh || echo "[run-worker] port-forward bootstrap skipped"

exec env PYTHONPATH=python-backend ./.venv/bin/python -m app.worker
