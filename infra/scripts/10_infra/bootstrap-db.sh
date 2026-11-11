#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# bootstrap-db.sh - create business databases inside the CNPG nereidforge cluster.
# External-PG mode: LS/CVAT/NF each connect to the same CNPG cluster using a
# distinct database; the charts only consume an existingSecret and never create
# databases themselves, so a blank-cluster deployment must pre-create them here.
# Must run AFTER postgresql.sh (needs the cluster up).
# Connects as the local `postgres` superuser via the instance socket (peer auth),
# so no superuser secret is required.

set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)"
source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

# Resolve the current CNPG primary pod (label-driven, no hardcoded ordinal).
PG_POD=$(kubectl get pods -n nereidforge -l cnpg.io/cluster="$PG_CLUSTER_NAME",cnpg.io/instanceRole=primary \
    -o jsonpath='{.items[0].metadata.name}')
[[ -n "$PG_POD" ]] || { log_err "CNPG primary pod not found"; exit 1; }

log_info "========== Pre-creating databases (CNPG nereidforge, primary=$PG_POD) =========="

# Wait for the primary to accept connections before creating databases.
kubectl wait --for=condition=Ready pod/"$PG_POD" -n nereidforge --timeout=300s >/dev/null 2>&1 \
    || { log_err "CNPG primary pod not Ready"; exit 1; }
retry 30 10 kubectl exec -n nereidforge "$PG_POD" -c postgres -- pg_isready -U postgres >/dev/null 2>&1 \
    || { log_err "PostgreSQL primary is not ready"; exit 1; }

for db in nereidforge labelstudio cvat; do
    if kubectl exec -n nereidforge "$PG_POD" -c postgres -- \
        psql -U postgres -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='$db'" 2>/dev/null | grep -q 1; then
        log_info "Database $db already exists; skipping"
    else
        log_info "Creating database $db (owner nereidforge)"
        kubectl exec -n nereidforge "$PG_POD" -c postgres -- \
            psql -U postgres -d postgres -c "CREATE DATABASE \"$db\" OWNER nereidforge"
    fi
done

log_ok "Databases pre-created (nereidforge / labelstudio / cvat)"
