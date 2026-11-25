#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Read-only, fail-closed preflight for direct NF OCI Helm installation.
# This script never installs/upgrades charts and never prints Secret data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PRE_ROOT/scripts/lib/common.sh"
source "$PRE_ROOT/scripts/lib/config.sh"

TENANT="${1:-nereidforge}"
if [[ "$TENANT" != "nereidforge" ]]; then
    log_err "The initial Helm handoff supports only the namespace-fixed tenant 'nereidforge'"
    exit 1
fi
VALUES_FILE="${NF_VALUES_FILE:-$PRE_HANDOFF_DIR/values.generated.yaml}"
OPERATOR_VALUES_FILE="${NF_OPERATOR_VALUES_FILE:-$PRE_HANDOFF_DIR/operator-values.yaml}"
SSO_MODE="${SSO_MODE:-ns}"
case "$SSO_MODE" in
    ns|oidc|local) ;;
    *) log_err "SSO_MODE must be ns, oidc, or local"; exit 1 ;;
esac
check_cmd python3 helm || exit 1
if ! command -v "$KUBECTL" >/dev/null 2>&1; then
    log_err "Missing required command: $KUBECTL"
    exit 1
fi

python3 "$PRE_ROOT/scripts/lib/generate_nf_handoff.py" validate \
    --values "$VALUES_FILE" --operator-values "$OPERATOR_VALUES_FILE"

require_resource() {
    local kind="$1" name="$2" namespace="${3:-}"
    local -a scope=()
    if [[ -n "$namespace" ]]; then
        scope=(-n "$namespace")
    fi
    if ! "$KUBECTL" get "$kind" "$name" "${scope[@]}" >/dev/null 2>&1; then
        log_err "Missing required resource: $kind/$name ${namespace:+in $namespace}"
        return 1
    fi
}

require_secret_keys() {
    local namespace="$1" name="$2"
    shift 2
    local key jsonpath_key value
    for key in "$@"; do
        # Secret keys such as ca.crt are literal field names. Without escaping,
        # JSONPath treats the dot as traversal and falsely reports an empty key.
        jsonpath_key="${key//./\\.}"
        if ! value="$("$KUBECTL" get secret "$name" -n "$namespace" \
            -o "jsonpath={.data.$jsonpath_key}" 2>/dev/null)" || [[ -z "$value" ]]; then
            log_err "Missing required Secret key: $namespace/$name [$key]"
            return 1
        fi
    done
}

require_ready_deployment() {
    local namespace="$1" name="$2" ready
    if ! ready="$("$KUBECTL" get deployment "$name" -n "$namespace" \
        -o jsonpath='{.status.readyReplicas}' 2>/dev/null)" || \
        [[ ! "$ready" =~ ^[1-9][0-9]*$ ]]; then
        log_err "Deployment is not ready: $namespace/$name"
        return 1
    fi
}

log_info "Checking read-only NF Helm prerequisites for $TENANT"
require_resource namespace "$TENANT"
require_resource namespace kubeflow
require_resource storageclass "$STORAGE_CLASS"
require_resource clusterissuer internal-ca
require_resource service seaweedfs-s3 "$TENANT"
require_resource service labelstudio "$TENANT"
require_resource service cvat-backend-service "$TENANT"
require_resource service ml-pipeline kubeflow
require_ready_deployment "$TENANT" labelstudio
require_ready_deployment "$TENANT" cvat-backend-server
require_ready_deployment kubeflow ml-pipeline
require_resource cluster.postgresql.cnpg.io "$PG_CLUSTER_NAME" "$TENANT"

require_secret_keys "$TENANT" seaweedfs-s3-credentials accessKey secretKey
require_secret_keys "$TENANT" "pg-$PG_CLUSTER_NAME-credentials" username password
require_secret_keys "$TENANT" mlpipeline-minio-artifact accesskey secretkey
require_secret_keys "$TENANT" nereid-internal-ca ca.crt
require_secret_keys "$TENANT" nereidforge-admin-credentials password
RUNTIME_KEYS=(
    NEREIDFORGE_LABELSTUDIO_TOKEN
    NEREIDFORGE_CVAT_TOKEN
    NEREIDFORGE_AUTH_JWT_SECRET
)
case "$SSO_MODE" in
    ns) RUNTIME_KEYS+=(NEREIDFORGE_SSO_NS_CLIENT_SECRET) ;;
    oidc) RUNTIME_KEYS+=(NEREIDFORGE_SSO_OIDC_CLIENT_SECRET) ;;
esac
require_secret_keys "$TENANT" nereidforge-service-credentials "${RUNTIME_KEYS[@]}"

log_ok "NF Helm preflight passed; no cluster resources were changed"
log_info "Install backend, then frontend, then ml-backend with both validated values files"
