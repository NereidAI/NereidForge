#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# nf-check.sh - NereidForge business-layer verification (L1 liveness + L2 real login)

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/config.sh"

TENANT="${1:-nereidforge}"
FAIL=0

fail_check() {
    log_err "$1"
    FAIL=$((FAIL+1))
}

# Accept curl's exact status or its legacy doubled form, never an arbitrary
# substring or 000. HTTP probes below use kubectl exec so cleanup messages from
# kubectl run --rm cannot be interleaved with or replace curl's output.
http_code_is() {
    local actual="$1" expected="$2" normalized
    normalized="$(printf '%s' "$actual" | tr -d '[:space:]')"
    [[ "$normalized" == "$expected" || "$normalized" == "$expected$expected" ]]
}

log_info "========== Verify NereidForge business layer ($TENANT) =========="

log_info "Check API..."
if ! wait_deployment_ready "nereidforge-api" "$TENANT"; then
    log_err "API deployment is not ready"
    exit 1
fi

API_URL="http://nereidforge-api.$TENANT:8080"
FE_URL="http://nereidforge-frontend.$TENANT:80"

# Reuse an API container as the curl client for the login and HTTP contract
# probes. The requests still traverse the API Service, while kubectl exec
# avoids the attach/log fallback and deletion-message races of kubectl run --rm.
API_POD=""
if ! API_POD=$(kubectl get pods -n "$TENANT" -l app.kubernetes.io/name=nereidforge-backend,component=api \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); then
    API_POD=""
fi

api_http_code() {
    local url="$1"
    [[ -n "$API_POD" ]] || return 1
    kubectl exec -n "$TENANT" "$API_POD" -c api -- \
        curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null
}

# 1. healthz / ready (L1)
if HEALTH=$(kubectl run -n "$TENANT" "api-verify-$$" --rm -i --restart=Never --image="$IMG_BUSYBOX" -- \
    wget -q -O - "$API_URL/healthz" 2>/dev/null); then
    if printf '%s\n' "$HEALTH" | grep -q '"status":"ok"'; then
        log_ok "API healthz OK"
    else
        fail_check "API healthz returned an unexpected response"
    fi
else
    fail_check "API healthz probe command failed"
fi
if READY=$(kubectl run -n "$TENANT" "ready-verify-$$" --rm -i --restart=Never --image="$IMG_BUSYBOX" -- \
    wget -q -O - "$API_URL/ready" 2>/dev/null); then
    if printf '%s\n' "$READY" | grep -q '"status":"ready"'; then
        log_ok "API ready OK"
    else
        fail_check "API ready returned an unexpected response"
    fi
else
    fail_check "API ready probe command failed"
fi

# 2. Real admin login to get a token (L2)
# NOTE: the captcha store is per-pod in-memory, so GET + POST must hit the SAME
# api pod. Run the pair inside one api pod against localhost:8080 (istio denies
# direct pod-IP cross-tenant traffic, so exec-in-pod is the reliable path).
log_info "Test real admin login..."
# Read the operator-retained administrator credential without falling back to a
# retired built-in password. An explicit NF_ADMIN_PASSWORD wins, followed by the
# legacy private file and then the direct-Helm existing-Secret contract. Secret
# material is never printed by this script.
NF_ADMIN_PW="${NF_ADMIN_PASSWORD:-}"
CRED_FILE="$NF_HANDOFF_DIR/admin-credentials.env"
if [[ -z "$NF_ADMIN_PW" && -f "$CRED_FILE" ]]; then
    # shellcheck disable=SC1091
    source "$CRED_FILE"
    NF_ADMIN_PW="${NF_ADMIN_PASSWORD:-}"
fi
if [[ -z "$NF_ADMIN_PW" ]]; then
    ADMIN_PASSWORD_B64=""
    if ADMIN_PASSWORD_B64=$(kubectl get secret nereidforge-admin-credentials -n "$TENANT" \
        -o jsonpath='{.data.password}' 2>/dev/null); then
        NF_ADMIN_PW=$(printf '%s' "$ADMIN_PASSWORD_B64" | base64 -d 2>/dev/null) || NF_ADMIN_PW=""
    fi
fi
LOGIN_SCRIPT='
CAP=$(curl -s http://127.0.0.1:8080/api/v1/auth/captcha)
CID=$(echo "$CAP" | sed -n "s/.*\"captcha_id\":\"\([^\"]*\)\".*/\1/p")
EXPR=$(echo "$CAP" | sed -n "s/.*\"expression\":\"\([^\"]*\)\".*/\1/p")
set -- ${EXPR:-}
ans=0
if [ $# -ge 3 ]; then
    op=$2
    [ "$op" = "x" ] && op="*"
    ans=$(( $1 $op $3 ))
fi
RESP=$(curl -s -w '\n%{http_code}' -X POST http://127.0.0.1:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"captcha_id\":\"${CID}\",\"captcha_answer\":${ans},\"username\":\"admin\",\"password\":\"${NF_ADMIN_PASSWORD}\"}")
HTTP_CODE=$(printf "%s\n" "$RESP" | tail -n 1)
RESP_BODY=$(printf "%s\n" "$RESP" | sed "\$d")
# Require both a successful HTTP response and a non-empty token. Return only an
# exit status; neither response output nor bearer credentials cross the exec
# stream, and acceptance does not depend on a short-lived stdout marker.
if [ "$HTTP_CODE" = 200 ] && printf "%s\n" "$RESP_BODY" | grep -Eq "\"access_token\"[[:space:]]*:[[:space:]]*\"[^\"]+\""; then
    exit 0
else
    exit 1
fi
'
if [[ -z "$NF_ADMIN_PW" ]]; then
    fail_check "admin login credential unavailable: set NF_ADMIN_PASSWORD, provide $CRED_FILE, or retain $TENANT/nereidforge-admin-credentials [password]"
elif [[ -z "$API_POD" ]]; then
    fail_check "admin login failed: no API pod is available for the login probe"
elif kubectl exec -n "$TENANT" "$API_POD" -c api -- \
    env NF_ADMIN_PASSWORD="$NF_ADMIN_PW" sh -c "$LOGIN_SCRIPT" >/dev/null 2>&1; then
    log_ok "admin login OK (got access_token)"
else
    fail_check "admin login probe failed or did not return a non-empty access token"
fi

# 2b. SSO status endpoint (verifies SSO config + suite reachability)
if SSO_STATUS=$(kubectl run -n "$TENANT" "sso-verify-$$" --rm -i --restart=Never --image="$IMG_BUSYBOX" -- \
    wget -q -O - "$API_URL/api/v1/auth/sso/status" 2>/dev/null); then
    # Accept either suite mode; NS is the standard default, oidc remains opt-in.
    if printf '%s\n' "$SSO_STATUS" | grep -q '"enabled":true' \
        && printf '%s\n' "$SSO_STATUS" | grep -qE '"mode":"(ns|oidc)"' \
        && printf '%s\n' "$SSO_STATUS" | grep -q '"logout_url":"http'; then
        SSO_MODE=$(printf '%s\n' "$SSO_STATUS" | sed -n 's/.*"mode":"\([a-z]*\)".*/\1/p')
        log_ok "SSO status OK (mode=$SSO_MODE, enabled=true, logout_url present)"
    else
        fail_check "SSO status contract failed (want enabled=true, mode=ns|oidc, logout_url)"
    fi
else
    fail_check "SSO status probe command failed"
fi

# 3. Training jobs endpoint reachability. The production route requires a
# bearer token, so an unauthenticated 401 proves reachability just as a 200 does
# in an explicitly auth-disabled development deployment. Other statuses fail.
log_info "Check training jobs endpoint..."
if JOBS_CODE=$(api_http_code "$API_URL/api/v1/training/jobs"); then
    if http_code_is "$JOBS_CODE" 200 || http_code_is "$JOBS_CODE" 401; then
        log_ok "Training jobs endpoint reachable (HTTP 200 or authenticated 401)"
    else
        fail_check "Training jobs endpoint HTTP contract failed (want 200 or 401)"
    fi
else
    fail_check "Training jobs endpoint probe command failed"
fi

# 2c. NS integration tenant-list endpoint lives OUTSIDE /api and must be
# secret-gated: with no X-NS-Integration-Secret it must answer 401 even though
# the route is mounted without the normal bearer-token protection.
if INT_CODE=$(api_http_code "$API_URL/v1/integration/ns/tenants"); then
    if http_code_is "$INT_CODE" 401; then
        log_ok "NS integration tenant-list secret-gated (401 without secret)"
    else
        fail_check "NS integration tenant-list secret gate failed (want HTTP 401 without secret)"
    fi
else
    fail_check "NS integration tenant-list probe command failed"
fi

# 4. Frontend
log_info "Check Frontend..."
if ! wait_deployment_ready "nereidforge-frontend" "$TENANT"; then
    log_err "Frontend deployment is not ready"
    exit 1
fi
if FE_RESP=$(kubectl run -n "$TENANT" "fe-verify-$$" --rm -i --restart=Never --image="$IMG_BUSYBOX" -- \
    wget -q -O - "$FE_URL/" 2>/dev/null); then
    if printf '%s\n' "$FE_RESP" | grep -qiE "nereidforge|<!DOCTYPE html>"; then
        log_ok "Frontend responding"
    else
        fail_check "Frontend returned an unexpected response"
    fi
else
    fail_check "Frontend probe command failed"
fi

if (( FAIL > 0 )); then
    log_err "NereidForge business layer verification failed: $FAIL core check(s) failed"
    exit 1
fi
log_ok "NereidForge business layer verification done"
