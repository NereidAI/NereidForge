#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Shared helpers for the self-contained pre-NereidForge module.

set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRE_ROOT="$(cd "$LIB_DIR/../.." && pwd)"

# A generated installer carries pre-specific metadata. Preserve an explicit
# operator/CI override while loading it; NF's RELEASE_TAG is deliberately not
# consulted anywhere in this module.
_PRE_RELEASE_TAG_OVERRIDE="${PRE_RELEASE_TAG:-}"
if [[ -f "$PRE_ROOT/release.env" ]]; then
    # shellcheck source=/dev/null
    source "$PRE_ROOT/release.env"
    if [[ -n "$_PRE_RELEASE_TAG_OVERRIDE" ]]; then
        PRE_RELEASE_TAG="$_PRE_RELEASE_TAG_OVERRIDE"
    fi
fi
unset _PRE_RELEASE_TAG_OVERRIDE
: "${SOURCE_COMMIT:=}"

# Git metadata belongs to the enclosing repository, not the copied module.
# A normal worktree therefore works even though only REPO_ROOT/.git exists.
REPO_ROOT="$(git -C "$PRE_ROOT" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -n "$REPO_ROOT" && "$PRE_ROOT" != "$REPO_ROOT" && "$PRE_ROOT" != "$REPO_ROOT"/* ]]; then
    REPO_ROOT=""
fi

# Raw source copies have neither generated metadata nor a repository that owns
# this module. Checking a tracked module marker avoids mistaking an unrelated
# parent checkout for the module's source repository.
if [[ ! -f "$PRE_ROOT/release.env" ]]; then
    _PRE_MODULE_PREFIX=""
    if [[ -n "$REPO_ROOT" && "$PRE_ROOT" != "$REPO_ROOT" ]]; then
        _PRE_MODULE_PREFIX="${PRE_ROOT#"$REPO_ROOT"/}/"
    fi
    if [[ -z "$REPO_ROOT" ]] || ! git -C "$REPO_ROOT" ls-files --error-unmatch \
        -- "${_PRE_MODULE_PREFIX}scripts/10_infra.sh" >/dev/null 2>&1; then
        echo "Unsupported unversioned source archive: use pre-nereidforge-installer-<version>.tar.gz or a Git checkout." >&2
        exit 1
    fi
    unset _PRE_MODULE_PREFIX
fi

# pre_validate_semver accepts bare SemVer, including prerelease/build metadata,
# and rejects numeric prerelease identifiers with forbidden leading zeroes.
pre_validate_semver() {
    local version="$1"
    [[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$ ]] || return 1
    if [[ "$version" == *-* ]]; then
        local prerelease="${version#*-}"
        prerelease="${prerelease%%+*}"
        local -a identifiers
        IFS='.' read -ra identifiers <<< "$prerelease"
        local identifier
        for identifier in "${identifiers[@]}"; do
            if [[ "$identifier" =~ ^[0-9]+$ && "${#identifier}" -gt 1 && "$identifier" == 0* ]]; then
                return 1
            fi
        done
    fi
}

# pre_version resolves only the pre release channel. Exact NF v* tags and the
# root RELEASE_TAG compatibility variable never participate in this decision.
pre_version() {
    if [[ -n "${PRE_RELEASE_TAG:-}" ]]; then
        if ! pre_validate_semver "$PRE_RELEASE_TAG"; then
            echo "PRE_RELEASE_TAG must be bare SemVer: $PRE_RELEASE_TAG" >&2
            return 2
        fi
        printf '%s\n' "$PRE_RELEASE_TAG"
        return 0
    fi
    if [[ -z "$REPO_ROOT" ]]; then
        echo "Generated release.env must set PRE_RELEASE_TAG for a source archive." >&2
        return 1
    fi
    local ref
    if ref="$(git -C "$REPO_ROOT" describe --tags --exact-match \
        --match 'pre-nereidforge/v*' HEAD 2>/dev/null)"; then
        local candidate="${ref#pre-nereidforge/v}"
        if [[ "$ref" == pre-nereidforge/v* ]] && pre_validate_semver "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi
    printf '%s\n' '0.0.0-dev'
}

# ========== Logging and local operator state ==========

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_err() { echo -e "${RED}[ERR]${NC} $*"; }

ensure_private_dir() {
    local dir="$1"
    (umask 077; mkdir -p -- "$dir")
    chmod 700 "$dir"
}

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

# ========== Validation and command helpers ==========

validate_required() {
    local missing=()
    local var
    for var in "$@"; do
        [[ -n "${!var:-}" ]] || missing+=("$var")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_err "Missing required environment variables: ${missing[*]}"
        return 1
    fi
}

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
        sleep "$delay"
        ((n++))
    done
}

check_cmd() {
    local cmd
    for cmd in "$@"; do
        if ! command -v "$cmd" &>/dev/null; then
            log_err "Missing command: $cmd"
            return 1
        fi
    done
}

# Split an upstream image into registry/repository/tag for chart values.
docker_ref_parts() {
    local ref="$1" rest tag registry=""
    tag="${ref##*:}"
    rest="${ref%:*}"
    local first="${rest%%/*}"
    if [[ "$first" == *.* || "$first" == *:* || "$first" == "localhost" ]]; then
        registry="$first"
        rest="${rest#*/}"
    fi
    _REF_REGISTRY="$registry"
    _REF_REPOSITORY="$rest"
    _REF_TAG="$tag"
}

# ========== Kubernetes helpers used by pre deployment and verification ==========

# Load the canonical object-store identity directly from an existing Kubernetes
# Secret, or from the legacy SeaweedFS config during the one-time migration.
# The Python helper shell-quotes both values; no credential file or handoff is
# created and command output contains only the exports consumed by this shell.
load_seaweedfs_s3_credentials() {
    local resource_kind="$1"
    local resource_name="$2"
    local source_kind="$3"
    local kubectl_cmd="${KUBECTL:-kubectl}"
    local exports
    if ! exports="$($kubectl_cmd get "$resource_kind" "$resource_name" -n nereidforge -o json \
        | python3 "$LIB_DIR/s3_credentials.py" exports "$source_kind")"; then
        log_err "Could not load the existing SeaweedFS credentials from $resource_kind/$resource_name"
        return 1
    fi
    eval "$exports"
}

ensure_ns() {
    local ns="$1"
    if ! kubectl get ns "$ns" &>/dev/null; then
        log_info "Creating namespace: $ns"
        kubectl create ns "$ns"
    fi
}

wait_deployment_ready() {
    local name="$1"
    local namespace="$2"
    local timeout="${3:-300}"
    log_info "Waiting for Deployment $name to be ready..."
    retry 1 5 kubectl rollout status "deployment/$name" -n "$namespace" --timeout="${timeout}s"
}

wait_statefulset_ready() {
    local name="$1"
    local namespace="$2"
    local timeout="${3:-300}"
    local start
    start="$(date +%s)"
    log_info "Waiting for StatefulSet $name to be ready..."
    while true; do
        local ready desired now
        ready="$(kubectl get sts "$name" -n "$namespace" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)"
        desired="$(kubectl get sts "$name" -n "$namespace" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)"
        if [[ "$ready" == "$desired" && "$desired" -gt 0 ]]; then
            log_ok "StatefulSet $name ready ($ready/$desired)"
            return 0
        fi
        now="$(date +%s)"
        if [[ $((now - start)) -gt $timeout ]]; then
            log_err "StatefulSet $name timed out ($ready/$desired)"
            return 1
        fi
        sleep 5
    done
}

# Wait for Dex to serve OIDC keys before any RequestAuthentication naming its
# issuer is applied. When istiod's first JWKS fetch for a new rule fails, it
# caches a built-in public-only JWKS whose private key was discarded and
# whose key has no kid; every Dex-issued JWT is then rejected with
# "Jwt verification fails" until the exponential-backoff refresher happens
# to recover. The bootstrap race (rule applied while Dex is still starting
# or being rescheduled) is prevented by gating the oauth2 target on this
# check. The parse runs on the operator host; the Dex container only needs
# wget. Tune bounds with DEX_READY_TIMEOUT / DEX_READY_POLL.
wait_for_dex_readiness() {
    local kubectl_cmd="${KUBECTL:-kubectl}"
    local namespace="${DEX_NAMESPACE:-auth}"
    local deployment="${DEX_DEPLOYMENT_NAME:-dex}"
    local container="${DEX_CONTAINER_NAME:-dex}"
    local port="${DEX_HTTP_PORT:-5556}"
    local timeout="${DEX_READY_TIMEOUT:-300}"
    local poll="${DEX_READY_POLL:-5}"
    local start now
    start="$(date +%s)"
    log_info "Waiting for Dex OIDC readiness ($namespace/$deployment) before applying the JWT RequestAuthentication..."
    while true; do
        if "$kubectl_cmd" rollout status "deployment/$deployment" -n "$namespace" \
                --timeout=10s >/dev/null 2>&1 \
            && "$kubectl_cmd" exec -n "$namespace" "deployment/$deployment" -c "$container" -- \
                wget -q -O - "http://127.0.0.1:$port/dex/keys" 2>/dev/null \
                | python3 -c '
import json
import sys
try:
    doc = json.load(sys.stdin)
except (json.JSONDecodeError, ValueError):
    sys.exit(1)
keys = doc.get("keys") or []
sys.exit(0 if keys and all(key.get("kid") for key in keys) else 1)
'; then
            log_ok "Dex is serving OIDC keys; the JWT policy may be applied"
            return 0
        fi
        now="$(date +%s)"
        if (( now - start >= timeout )); then
            log_err "Dex did not become OIDC-ready within ${timeout}s"
            log_err "Refusing to apply the dex-jwt RequestAuthentication against an unreachable issuer; fix Dex and rerun"
            return 1
        fi
        sleep "$poll"
    done
}

# Verify the gateway cached the real Dex JWKS after the dex-jwt
# RequestAuthentication was applied, rather than istiod's public-only
# fallback. The fallback key carries no kid and cannot verify any token;
# real Dex keys always have one. The gateway proxy exposes its xDS-served
# config dump on the loopback admin port, so the check stays inside the
# cluster and needs no registry or browser call. Tune with
# ISTIO_JWKS_VERIFY_TIMEOUT / ISTIO_JWKS_VERIFY_POLL.
verify_istiod_dex_jwks() {
    local kubectl_cmd="${KUBECTL:-kubectl}"
    local gateway_ns="${ISTIO_GATEWAY_NAMESPACE:-istio-system}"
    local gateway_deploy="${ISTIO_GATEWAY_DEPLOYMENT:-istio-ingressgateway}"
    local proxy_container="${ISTIO_PROXY_CONTAINER:-istio-proxy}"
    local timeout="${ISTIO_JWKS_VERIFY_TIMEOUT:-180}"
    local poll="${ISTIO_JWKS_VERIFY_POLL:-5}"
    local start now dump
    start="$(date +%s)"
    log_info "Verifying the gateway cached the real Dex JWKS (not the public-only fallback)..."
    while true; do
        dump="$("$kubectl_cmd" exec -n "$gateway_ns" "deployment/$gateway_deploy" \
            -c "$proxy_container" -- curl -s localhost:15000/config_dump 2>/dev/null || true)"
        if [[ -n "$dump" ]] && printf '%s' "$dump" | python3 -c '
import json
import sys
try:
    doc = json.load(sys.stdin)
except (json.JSONDecodeError, ValueError):
    sys.exit(1)
kids = []
def walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "local_jwks" and isinstance(value, dict):
                try:
                    inline = json.loads(value.get("inline_string") or "{}")
                except (json.JSONDecodeError, ValueError):
                    continue
                kids.extend(
                    key_id for key_id in
                    (item.get("kid") for item in inline.get("keys", []))
                    if key_id
                )
            walk(value)
    elif isinstance(node, list):
        for item in node:
            walk(item)
walk(doc)
sys.exit(0 if kids else 1)
'; then
            log_ok "Gateway cached the real Dex JWKS"
            return 0
        fi
        now="$(date +%s)"
        if (( now - start >= timeout )); then
            log_err "Gateway did not cache a real Dex JWKS within ${timeout}s; istiod may hold the public-only fallback"
            log_err "Inspect: kubectl logs -n istio-system deploy/istiod | grep -i jwks"
            log_err "Once Dex is healthy, restart istiod and rerun this stage to re-fetch the JWKS"
            return 1
        fi
        sleep "$poll"
    done
}

export PRE_ROOT REPO_ROOT SOURCE_COMMIT
