#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Start the NereidForge Python API server (dev).
# Loads the centralized NF_DEV_ENV_FILE and selects the dev configuration.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/python-backend"
source "$REPO_ROOT/infra/scripts/lib/common.sh"

if [[ -f "$NF_DEV_ENV_FILE" ]]; then
  set -a
  source "$NF_DEV_ENV_FILE"
  set +a
fi

export NEREIDFORGE_ENV="${NEREIDFORGE_ENV:-dev}"
export NEREIDFORGE_AUTH_ENABLED="${NEREIDFORGE_AUTH_ENABLED:-true}"
ensure_development_jwt_secret
if [[ -z "${NEREIDFORGE_DATABASE_PASSWORD:-}" && -n "${POSTGRES_PASSWORD:-}" ]]; then
  export NEREIDFORGE_DATABASE_PASSWORD="$POSTGRES_PASSWORD"
fi

exec "$REPO_ROOT/.venv/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8080 --workers 1 --loop asyncio
