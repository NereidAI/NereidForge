#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# common.sh - Shared function library for all deployment scripts.
#
# Merged from the former lib/common.sh + lib/k8s.sh:
#   - k8s-specific helpers (apply/patch/wait/secret/svc) now live here.
#   - Duplicated definitions were collapsed into a single implementation.
#   - Naming is *_ready suffixed (wait_deployment_ready / wait_statefulset_ready),
#     matching what the active scripts actually call.

set -euo pipefail

# ========== Repository and centralized operator paths ==========
# Keep sensitive operator state outside every product checkout while allowing
# each product to own a separate values/log directory. All defaults derive
# from the repository parent so sibling checkouts share one deployment root.
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$LIB_DIR/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
: "${DEPLOYMENT_ROOT:=$WORKSPACE_ROOT/.deployment}"
: "${DEPLOY_VALUES_DIR:=$DEPLOYMENT_ROOT/deploy-values/nereidforge}"
: "${DEPLOY_LOG_DIR:=$DEPLOYMENT_ROOT/deploy-logs/nereidforge}"
: "${NF_ENV_FILE:=$DEPLOY_VALUES_DIR/site.env}"
: "${NF_DEV_ENV_FILE:=$DEPLOY_VALUES_DIR/dev.env}"
: "${NF_HANDOFF_DIR:=$DEPLOY_VALUES_DIR}"
export DEPLOYMENT_ROOT DEPLOY_VALUES_DIR DEPLOY_LOG_DIR
export NF_ENV_FILE NF_DEV_ENV_FILE NF_HANDOFF_DIR

# ========== Logging ==========

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_err() { echo -e "${RED}[ERR]${NC} $*"; }

# ensure_private_dir creates a sensitive directory and any missing parents with
# owner-only permissions. Existing parents are left untouched; the target is
# tightened to 0700 so credentials and deployment output cannot leak locally.
ensure_private_dir() {
    local dir="$1"
    (umask 077; mkdir -p -- "$dir")
    chmod 700 "$dir"
}

# ensure_development_jwt_secret supplies a unique signing key for a host-run
# development process when the operator has not pinned one. An ephemeral key
# invalidates existing tokens on restart; independent API and worker launches
# that must share a key therefore need NEREIDFORGE_AUTH_JWT_SECRET in dev.env.
ensure_development_jwt_secret() {
    if [[ -z "${NEREIDFORGE_AUTH_JWT_SECRET:-}" ]]; then
        NEREIDFORGE_AUTH_JWT_SECRET="$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | xxd -p -c 64)"
        export NEREIDFORGE_AUTH_JWT_SECRET
        log_warn "Generated an ephemeral JWT signing key; set NEREIDFORGE_AUTH_JWT_SECRET for stable sessions"
    fi
    : "${NEREIDFORGE_AUTH_JWT_SECRET:?JWT secret generation failed}"
}

# ========== Deployment logging ==========
# init_deploy_log <stage> — mirror all subsequent stdout/stderr into a
# timestamped mode-0600 file under DEPLOY_LOG_DIR. latest-<stage>.log always
# points at the most recent run. Call once per entry-point script after sourcing
# lib/config.sh so a site-specific DEPLOY_LOG_DIR override has been loaded.
init_deploy_log() {
    local stage="$1"
    local dir="$DEPLOY_LOG_DIR"
    ensure_private_dir "$dir"
    local ts
    ts="$(date +%Y%m%d-%H%M%S)"
    DEPLOY_LOG_FILE="$dir/${ts}-${stage}.log"
    (umask 077; touch "$DEPLOY_LOG_FILE")
    chmod 600 "$DEPLOY_LOG_FILE"
    ln -sfn "$(basename "$DEPLOY_LOG_FILE")" "$dir/latest-${stage}.log"
    export DEPLOY_LOG_FILE
    exec > >(tee -a "$DEPLOY_LOG_FILE") 2>&1
    log_info "Deploy log: $DEPLOY_LOG_FILE"
}

# ========== Validation / misc helpers ==========

# Validate required variables
validate_required() {
    local missing=()
    for var in "$@"; do
        if [[ -z "${!var:-}" ]]; then
            missing+=("$var")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_err "Missing required environment variables: ${missing[*]}"
        return 1
    fi
    return 0
}

# Retry a command until it succeeds or the retry budget is exhausted.
# Usage: retry <max> <delay> <cmd...>
retry() {
    local max="${1:-3}"
    local delay="${2:-5}"
    shift 2
    local n=1
    while true; do
        if "$@"; then
            return 0
        fi
        if [[ $n -ge $max ]]; then
            log_err "Command failed after $max retries: $*"
            return 1
        fi
        log_warn "Attempt $n failed, retrying in ${delay}s: $*"
        sleep $delay
        ((n++))
    done
}

# Check that required commands exist
check_cmd() {
    for cmd in "$@"; do
        if ! command -v "$cmd" &>/dev/null; then
            log_err "Missing command: $cmd"
            return 1
        fi
    done
}

# Generate a random password
gen_password() {
    local len="${1:-32}"
    openssl rand -base64 $len | tr -d "=+/" | cut -c1-$len
}

# Remove lines matching a pattern from a kustomization file (keeps a .bak)
patch_kustomization() {
    local file="$1"
    local pattern="$2"
    if [[ -f "$file" ]]; then
        cp "$file" "$file.bak"
        sed -i "/$pattern/d" "$file"
        log_info "Patched $file: removed $pattern"
    fi
}

# Interactive fallback for required config (referenced by 00_deploy-all.sh when
# the values are not supplied via environment). Reads unset variables.
interactive_config() {
    local var value
    for var in BASE_DOMAIN NFS_SERVER NFS_SHARE; do
        if [[ -z "${!var:-}" ]]; then
            read -rp "Missing config ${var}, please enter (Enter to skip): " value
            export "$var=${value}"
        fi
    done
}

# ========== Namespace / registry / secret helpers ==========

# nf_version derives the bare SemVer image version (root treaty §19.5). There
# is no VERSION file: an exported RELEASE_TAG wins (operator/CI override),
# otherwise the exact v* git tag at this checkout is used with its leading v
# stripped (v0.0.1 -> 0.0.1), and every non-tag checkout builds the rolling
# 0.0.0-dev dev channel.
nf_version() {
    if [[ -n "${RELEASE_TAG:-}" ]]; then
        printf '%s\n' "$RELEASE_TAG"
        return
    fi
    local ref
    if ref="$(git -C "$ROOT_DIR" describe --tags --exact-match 2>/dev/null)"; then
        printf '%s\n' "${ref#v}"
    else
        printf '%s\n' "0.0.0-dev"
    fi
}

# nf_ref resolves a self-built image reference (class A, release-tagged) under
# the single public scheme (root treaty section 18):
#   ghcr.io/nereidai/<flat-name>:<tag>
# The flat name keeps the category as a prefix and collapses duplicate
# prefixes via the alias map below. There is no registry/layout switch:
# air-gapped sites resolve the same reference through the node containerd
# mirror rendered by NereidOverview harbor-sync.sh.
# Usage: nf_ref <category/name> [tag]   (tag defaults to nf_version)
NF_FLAT_ALIASES_RAW='
ml-backend/ml-backend=ml-backend
ml-backend/ml-backend-custom=ml-backend-custom
runtime=nf-runtime
'
nf_ref() {
    local repo="$1" tag="${2:-}"
    [[ -z "$tag" ]] && tag="$(nf_version)"
    local root="ghcr.io/nereidai"
    local flat
    flat="$(printf '%s' "$NF_FLAT_ALIASES_RAW" | awk -F= -v r="$repo" '$1==r{print $2; exit}')"
    [[ -z "$flat" ]] && flat="${repo//\//-}"
    printf '%s/%s:%s\n' "$root" "$flat" "$tag"
}

# docker_ref_parts splits a full image reference into REGISTRY / REPOSITORY / TAG.
# It sets the globals _REF_REGISTRY (may be empty for docker.io shorthand),
# _REF_REPOSITORY, _REF_TAG.
docker_ref_parts() {
    local ref="$1" rest tag registry=""
    tag="${ref##*:}"
    rest="${ref%:*}"
    # The first path element is a registry only when it contains "." or ":"
    # or is "localhost".
    local first="${rest%%/*}"
    if [[ "$first" == *.* || "$first" == *:* || "$first" == "localhost" ]]; then
        registry="$first"
        rest="${rest#*/}"
    fi
    _REF_REGISTRY="$registry"
    _REF_REPOSITORY="$rest"
    _REF_TAG="$tag"
}

ensure_ns() {
    local ns="$1"
    if ! kubectl get ns "$ns" &>/dev/null; then
        log_info "Creating namespace: $ns"
        kubectl create ns "$ns"
    fi
}

# create_registry_secret creates a Docker registry Secret unless it already
# exists. Registry coordinates are never committed (root treaty section 18):
# the caller must pass server/user/pass explicitly. Without complete credentials
# this is a no-op, so public/anonymous deployment renders no pull secret.
create_registry_secret() {
    local ns="$1"
    local secret_name="${2:-harbor-registry-secret}"
    local server="${3:-}"
    local user="${4:-}"
    local pass="${5:-}"
    # Docker registry email is optional metadata; never invent an organization identity.
    local email="${6:-}"

    if [[ -z "$server" || -z "$user" || -z "$pass" ]]; then
        log_info "No registry credentials supplied; skip pull secret $secret_name in $ns"
        return 0
    fi

    if kubectl get secret "$secret_name" -n "$ns" &>/dev/null; then
        log_info "Secret $secret_name already exists in $ns"
        return 0
    fi
    log_info "Creating registry secret $secret_name in $ns"
    kubectl create secret docker-registry "$secret_name" \
        --docker-server="$server" \
        --docker-username="$user" \
        --docker-password="$pass" \
        --docker-email="$email" \
        -n "$ns"
}

# Read a (base64) key out of a secret
get_secret() {
    local name="$1"
    local namespace="$2"
    local key="$3"
    kubectl get secret "$name" -n "$namespace" -o jsonpath="{.data.$key}" | base64 -d
}

# Get a service cluster IP
get_svc_ip() {
    local name="$1"
    local namespace="$2"
    kubectl get svc "$name" -n "$namespace" -o jsonpath='{.spec.clusterIP}'
}

# ========== Apply / delete / patch helpers ==========

# Delete resource (ignore not found)
delete_resource() {
    local resource="$1"
    local namespace="${2:-}"
    local ns_flag=""
    [[ -n "$namespace" ]] && ns_flag="-n $namespace"
    kubectl delete $ns_flag "$resource" --ignore-not-found=true
}

# Patch resource (json patch)
patch_json() {
    local resource="$1"
    local namespace="$2"
    local patch="$3"
    local ns_flag=""
    [[ -n "$namespace" ]] && ns_flag="-n $namespace"
    kubectl patch $ns_flag "$resource" --type=json -p="$patch"
}

# Patch resource (merge patch)
patch_merge() {
    local resource="$1"
    local namespace="$2"
    local patch="$3"
    local ns_flag=""
    [[ -n "$namespace" ]] && ns_flag="-n $namespace"
    kubectl patch $ns_flag "$resource" --patch="$patch"
}

# Patch labels
patch_labels() {
    local resource="$1"
    local namespace="$2"
    local labels="$3"
    local ns_flag=""
    [[ -n "$namespace" ]] && ns_flag="-n $namespace"
    kubectl label $ns_flag "$resource" $labels --overwrite
}

# Patch annotations
patch_annotations() {
    local resource="$1"
    local namespace="$2"
    local annotations="$3"
    local ns_flag=""
    [[ -n "$namespace" ]] && ns_flag="-n $namespace"
    kubectl annotate $ns_flag "$resource" $annotations --overwrite
}

# ========== Wait helpers ==========

# Wait for a Deployment to finish rolling out
wait_deployment_ready() {
    local name="$1"
    local namespace="$2"
    local timeout="${3:-300}"
    log_info "Waiting for Deployment $name to be ready..."
    retry 1 5 kubectl rollout status deployment/$name -n $namespace --timeout=${timeout}s
}

# Wait until StatefulSet readyReplicas == desired replicas
wait_statefulset_ready() {
    local name="$1"
    local namespace="$2"
    local timeout="${3:-300}"
    local start=$(date +%s)
    log_info "Waiting for StatefulSet $name to be ready..."
    while true; do
        local ready=$(kubectl get sts $name -n $namespace -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
        local desired=$(kubectl get sts $name -n $namespace -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)
        if [[ "$ready" == "$desired" && "$desired" -gt 0 ]]; then
            log_ok "StatefulSet $name ready ($ready/$desired)"
            return 0
        fi
        local now=$(date +%s)
        if [[ $((now - start)) -gt $timeout ]]; then
            log_err "StatefulSet $name timed out ($ready/$desired)"
            return 1
        fi
        sleep 5
    done
}

# Wait until at least $expected pods matching a selector are Running
wait_pods_running() {
    local selector="$1"
    local namespace="$2"
    local expected="${3:-1}"
    local timeout="${4:-300}"
    local start=$(date +%s)
    log_info "Waiting for Pods ($selector) to be running..."
    while true; do
        local running=$(kubectl get pods -n "$namespace" -l "$selector" --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
        if [[ $running -ge $expected ]]; then
            log_ok "Pods ready ($running/$expected)"
            return 0
        fi
        local now=$(date +%s)
        if [[ $((now - start)) -gt $timeout ]]; then
            log_err "Pods timed out ($running/$expected)"
            kubectl get pods -n "$namespace" -l "$selector"
            return 1
        fi
        sleep 5
    done
}

# Wait for a CRD to be established
wait_crd() {
    local crd="$1"
    local timeout="${2:-60}"
    log_info "Waiting for CRD $crd to be established..."
    kubectl wait --for=condition=Established crd/$crd --timeout=${timeout}s
}

# Wait until a Service has ready endpoints
wait_endpoints() {
    local svc="$1"
    local namespace="$2"
    local timeout="${3:-60}"
    local start=$(date +%s)
    log_info "Waiting for Service $svc endpoints to be ready..."
    while true; do
        local eps=$(kubectl get endpoints "$svc" -n "$namespace" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null | wc -w)
        if [[ $eps -gt 0 ]]; then
            log_ok "Endpoints ready ($eps)"
            return 0
        fi
        local now=$(date +%s)
        if [[ $((now - start)) -gt $timeout ]]; then
            log_err "Endpoints timed out"
            return 1
        fi
        sleep 3
    done
}
