#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Start the NF API for development (python-backend / FastAPI + uvicorn): load
# config.dev.yaml + local env; data domain = dev (nereidforge_dev).
# Usage: LOG_LEVEL=debug ./scripts/run-dev.sh       # Debug mode
#        LOG_LEVEL=warning ./scripts/run-dev.sh     # Warnings and errors only
#        LOG_FORMAT=plain ./scripts/run-dev.sh      # Plain text (not JSON)
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
export NEREIDFORGE_AUTH_ENABLED=true
# Never start with a shared signing key. Pin the secret in NF_DEV_ENV_FILE when
# development tokens must survive a restart; otherwise use an ephemeral key.
ensure_development_jwt_secret
# Reuse the Compose database credential when the operator supplied one, without
# overriding a password loaded from the development environment or rendered config.
if [[ -z "${NEREIDFORGE_DATABASE_PASSWORD:-}" && -n "${POSTGRES_PASSWORD:-}" ]]; then
  export NEREIDFORGE_DATABASE_PASSWORD="$POSTGRES_PASSWORD"
fi
# Dev shares the SAME data plane as prod (single SeaweedFS bucket + Postgres DB);
# no separate nereidforge-dev / nereidforge_dev namespace. Override the rendered
# /etc config (or any yaml) explicitly so dev == prod data.
export NEREIDFORGE_STORAGE_BUCKET="${NEREIDFORGE_STORAGE_BUCKET:-nereidforge}"
export NEREIDFORGE_DATABASE_NAME="${NEREIDFORGE_DATABASE_NAME:-nereidforge}"
# Label Studio: the external ingress host 404s on this host, so reach LS via the
# dev port-forward (8890 -> labelstudio svc:8080); token comes from NF_DEV_ENV_FILE.
export NEREIDFORGE_LABELSTUDIO_HOST="${NEREIDFORGE_LABELSTUDIO_HOST:-http://localhost:8890}"

# Bring up host-side port-forwards the on-host API needs to reach in-cluster
# services (seaweedfs S3 :8833, KFP :18888, Model Registry :8889). Idempotent and
# self-healing; no-ops if kubectl/sudo unavailable or a port is already listening.
bash "$(dirname "$0")/dev-port-forward.sh" || echo "[run-dev] port-forward bootstrap skipped"

# Logging: default to info for uvicorn access, app logging handled by _setup_logging()
LOG_LEVEL="${LOG_LEVEL:-info}"
UVICORN_LOG_LEVEL="${UVICORN_LOG_LEVEL:-info}"

exec ./.venv/bin/uvicorn app.main:app --app-dir app/python-backend \
  --host 0.0.0.0 --port 8080 --reload --timeout-keep-alive 65 \
  --loop asyncio \
  --log-level "$UVICORN_LOG_LEVEL"
