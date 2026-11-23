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

export PRE_ROOT REPO_ROOT SOURCE_COMMIT
