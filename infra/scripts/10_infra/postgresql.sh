#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# postgresql.sh - Deploy CloudNativePG PostgreSQL (Helm chart).

set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)"
source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

ROOT_DIR="$(cd "$LIB_DIR/../../.." && pwd)"
CHARTS_DIR="$ROOT_DIR/infra/charts"

log_info "========== Deploying PostgreSQL - CloudNativePG (Helm) =========="

# Read configuration from environment variables.
: "${PG_CLUSTER_NAME:=nereidforge-postgresql}"
: "${PG_INSTANCES:=3}"
: "${PG_STORAGE_SIZE:=50Gi}"
: "${PG_DB_NAME:=nereidforge}"
: "${PG_DB_USER:=nereidforge}"

read_password() {
    local prompt="$1"
    local var_name="$2"
    local value
    if [[ -n "${!var_name:-}" ]]; then
        value="${!var_name}"
        log_info "$prompt: ******** (from environment variable)"
    else
        read -rsp "$prompt: " value
        echo
    fi
    export "$var_name"="$value"
    echo "$value"
}

gen_password() {
    local len="${1:-32}"
    openssl rand -base64 $len | tr -d "=+/" | cut -c1-$len
}

export PG_ADMIN_PASSWORD="${PG_ADMIN_PASSWORD:-}"

# Ensure a password is generated.
if [[ -z "$PG_ADMIN_PASSWORD" ]]; then
    PG_ADMIN_PASSWORD=$(gen_password 32)
    log_info "Generated a random PG password"
fi

# CNPG CRDs live in charts/postgresql/crds/ and are installed automatically by
# Helm (server-side by default for crds/). The bootstrap secret is rendered by
# the chart (templates/00-bootstrap-secret.yaml).

# Reuse a pre-existing cluster-wide CNPG operator (e.g. installed by another
# team in a different namespace): skip the bundled operator and adopt its CRDs
# for this release so Helm's ownership check passes.
OPERATOR_EXTRA=()
if kubectl get deploy -A --no-headers 2>/dev/null | awk '{print $2}' | grep -qx 'cloudnative-pg'; then
    log_info "Existing cloudnative-pg Operator detected; reusing it and disabling the chart's bundled Operator"
    OPERATOR_EXTRA+=(--set cnpg.enabled=false)
    for crd in $(kubectl get crd -o name 2>/dev/null | grep 'postgresql.cnpg.io'); do
        ann_ns=$(kubectl get "$crd" -o jsonpath='{.metadata.annotations.meta\.helm\.sh/release-namespace}' 2>/dev/null || true)
        if [[ "$ann_ns" != "nereidforge" ]]; then
            kubectl annotate "$crd" --overwrite \
                meta.helm.sh/release-name=postgresql \
                meta.helm.sh/release-namespace=nereidforge >/dev/null
        fi
    done
fi

# Deploy the cnpg operator + Cluster through the postgresql chart.
cd "$CHARTS_DIR/postgresql"
ls charts/*.tgz >/dev/null 2>&1 || helm dependency build

helm upgrade --install postgresql . -n nereidforge --create-namespace \
    --set cnpg.cluster.name="$PG_CLUSTER_NAME" \
    --set cnpg.cluster.instances="$PG_INSTANCES" \
    --set cnpg.cluster.imageName="$IMG_POSTGRESQL" \
    --set cnpg.cluster.storage.size="$PG_STORAGE_SIZE" \
    --set cnpg.cluster.storage.storageClass="$STORAGE_CLASS" \
    --set cnpg.cluster.bootstrap.initdb.database="$PG_DB_NAME" \
    --set cnpg.cluster.bootstrap.initdb.owner="$PG_DB_USER" \
    --set cnpg.cluster.bootstrap.initdb.secret.name=pg-${PG_CLUSTER_NAME}-credentials \
    --set bootstrap.password="$PG_ADMIN_PASSWORD" \
    "${OPERATOR_EXTRA[@]}" \
    --set "cnpg.cluster.postgresql.parameters.max_connections=200" \
    --set "cnpg.cluster.postgresql.parameters.shared_buffers=256MB" \
    --set "cnpg.cluster.postgresql.parameters.effective_cache_size=1GB" \
    --set "cnpg.cluster.postgresql.parameters.work_mem=16MB" \
    --set "cnpg.cluster.postgresql.parameters.maintenance_work_mem=128MB" \
    --wait --timeout 10m

# Wait for the Cluster to become ready.
log_info "Waiting for the PostgreSQL Cluster to become ready..."
retry 60 10 kubectl wait --for=condition=Ready cluster.postgresql.cnpg.io/$PG_CLUSTER_NAME -n nereidforge --timeout=300s

# Verify connectivity.
log_info "Verifying the PostgreSQL connection..."
kubectl run -n nereidforge pg-test-$$ --rm -i --restart=Never \
    --image="$IMG_POSTGRESQL" \
    --env=PGPASSWORD="$PG_ADMIN_PASSWORD" -- \
    psql -h ${PG_CLUSTER_NAME}-rw -U "$PG_DB_USER" -d "$PG_DB_NAME" -c "SELECT version();"

log_ok "PostgreSQL deployment completed"
log_info "Connection information:"
echo "  Host: ${PG_CLUSTER_NAME}-rw.nereidforge.svc.cluster.local"
echo "  Port: 5432"
echo "  Database: $PG_DB_NAME"
echo "  User: $PG_DB_USER"
echo "  Password: $PG_ADMIN_PASSWORD"
