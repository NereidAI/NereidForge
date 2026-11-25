#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# postgresql.sh - Deploy CloudNativePG PostgreSQL (Helm chart).

set -euo pipefail

PRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB_DIR="$PRE_ROOT/scripts/lib"
source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

CHARTS_DIR="$PRE_ROOT/charts"

log_info "========== Deploying PostgreSQL - CloudNativePG (Helm) =========="

# Read configuration from environment variables.
: "${PG_CLUSTER_NAME:=nereidforge-postgresql}"
: "${PG_INSTANCES:=3}"
: "${PG_STORAGE_SIZE:=50Gi}"
: "${PG_DB_NAME:=nereidforge}"
: "${PG_DB_USER:=nereidforge}"

export PG_ADMIN_PASSWORD="${PG_ADMIN_PASSWORD:-}"
export PG_DB_USER
PG_SECRET_NAME="pg-${PG_CLUSTER_NAME}-credentials"
SITE_PG_ADMIN_PASSWORD="$PG_ADMIN_PASSWORD"

load_pg_credentials() {
    local resource="$1" exports
    if ! exports="$(printf '%s' "$resource" | python3 "$LIB_DIR/pg_credentials.py" exports)"; then
        log_err "Existing PostgreSQL credentials Secret is malformed; refusing site fallback"
        return 1
    fi
    eval "$exports"
    if [[ "$PG_SECRET_USERNAME" != "$PG_DB_USER" ]]; then
        log_err "Existing PostgreSQL Secret username differs from PG_DB_USER"
        return 1
    fi
    PG_ADMIN_PASSWORD="$PG_SECRET_PASSWORD"
    export PG_ADMIN_PASSWORD
    if [[ -n "$SITE_PG_ADMIN_PASSWORD" && "$SITE_PG_ADMIN_PASSWORD" != "$PG_ADMIN_PASSWORD" ]]; then
        log_err "PG_ADMIN_PASSWORD differs from the existing PostgreSQL Secret; restore the retained site value instead of rotating the Secret"
        return 1
    fi
}

# The canonical Secret owns the identity after first creation. Read it before
# any namespace, CRD, dependency, or Helm mutation so reruns cannot rotate only
# the Kubernetes side of an already initialized CNPG role.
PG_SECRET_JSON=""
if PG_SECRET_JSON="$($KUBECTL get secret "$PG_SECRET_NAME" -n nereidforge -o json 2>/dev/null)"; then
    load_pg_credentials "$PG_SECRET_JSON" || exit 1
    log_info "Reusing the existing PostgreSQL credentials Secret (credentials not logged)"
    $KUBECTL annotate secret "$PG_SECRET_NAME" -n nereidforge --overwrite \
        helm.sh/resource-policy=keep >/dev/null
else
    PG_LOOKUP_ERROR="$($KUBECTL get secret "$PG_SECRET_NAME" -n nereidforge -o name 2>&1 || true)"
    if [[ "$PG_LOOKUP_ERROR" != *"NotFound"* && "$PG_LOOKUP_ERROR" != *"not found"* ]]; then
        log_err "Could not query the existing PostgreSQL credentials Secret"
        exit 1
    fi
    validate_required PG_ADMIN_PASSWORD || exit 1
    ensure_ns "nereidforge"

    # Create-only semantics preserve a concurrent winner. Diagnostics are
    # captured but never printed because a rejected API response could echo the
    # candidate manifest. JSON carries every credential byte over stdin only.
    if create_error="$(
        {
            python3 "$LIB_DIR/pg_credentials.py" manifest "$PG_SECRET_NAME" nereidforge \
                | $KUBECTL create -f - >/dev/null
        } 2>&1
    )"; then
        log_info "Created the PostgreSQL credentials Secret (credentials not logged)"
    elif [[ "$create_error" == *"AlreadyExists"* ]]; then
        log_info "Reusing the concurrently created PostgreSQL credentials Secret"
    else
        log_err "Could not create the PostgreSQL credentials Secret"
        exit 1
    fi
    if ! PG_SECRET_JSON="$($KUBECTL get secret "$PG_SECRET_NAME" -n nereidforge -o json)"; then
        log_err "Could not reload the PostgreSQL credentials Secret"
        exit 1
    fi
    load_pg_credentials "$PG_SECRET_JSON" || exit 1
fi
unset PG_SECRET_JSON PG_SECRET_USERNAME PG_SECRET_PASSWORD SITE_PG_ADMIN_PASSWORD

# CNPG CRDs live in charts/postgresql/crds/. The Cluster references the
# externally retained Secret by name; Helm never receives its credential data.

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
    --set cnpg.cluster.bootstrap.initdb.secret.name="$PG_SECRET_NAME" \
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

# Verify connectivity with the retained identity. The password travels over
# stdin rather than process arguments, Pod spec environment, or deploy logs.
log_info "Verifying the PostgreSQL connection..."
printf '%s' "$PG_ADMIN_PASSWORD" |
    kubectl run -n nereidforge pg-test-$$ --rm -i --restart=Never \
        --image="$IMG_POSTGRESQL" -- \
        sh -c '
            # The suffix prevents command substitution from stripping trailing
            # newlines; removing it preserves every retained password byte.
            PGPASSWORD="$(cat; printf .)"
            PGPASSWORD=${PGPASSWORD%.}
            export PGPASSWORD
            psql -h "$3" -U "$1" -d "$2" -c "SELECT version();"
        ' sh "$PG_DB_USER" "$PG_DB_NAME" "${PG_CLUSTER_NAME}-rw"

log_ok "PostgreSQL deployment completed"
log_info "Connection information:"
echo "  Host: ${PG_CLUSTER_NAME}-rw.nereidforge.svc.cluster.local"
echo "  Port: 5432"
echo "  Database: $PG_DB_NAME"
echo "  User: $PG_DB_USER"
echo "  Credentials: Kubernetes Secret $PG_SECRET_NAME (not logged)"
