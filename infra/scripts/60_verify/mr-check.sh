#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# mr-check.sh - Model Registry verification against the REAL deployed topology
# (and the real user path through the Kubeflow gateway).
#
# Live topology (upstream overlays pin namespaces; do NOT assume everything is
# in nereidforge):
#   registry server + db -> default ns     (model-registry-service:8080)
#   registry UI         -> kubeflow ns     (model-registry-ui-service:80)
#   gateway route       -> /model-registry/* -> UI, /api/model_registry/* -> server
#
# The old version of this script hard-coded MR_NS=nereidforge (where no
# registry resource exists) and suffixed every probe with `|| true`, so it
# always printed "verified" even when the dashboard page returned 500.

set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/config.sh"

# lib/config.sh enables set -e; this script deliberately runs failing probes
# and reports them itself, so disable errexit after sourcing.
set +e

# The authenticated probe must use the same operator-managed identity as Dex.
# Never turn a missing site credential into a known public fallback.
validate_required DEX_STATIC_EMAIL DEX_STATIC_PASSWORD || exit 1

MR_SERVER_NS="default"
MR_UI_NS="kubeflow"
GW_NS="istio-system"
fail=0

# check <description> <0=ok|nonzero=fail from $?>
check() {
    local desc="$1" rc="$2"
    if [ "$rc" = "0" ]; then
        log_ok "PASS: $desc"
    else
        log_info "FAIL: $desc"
        fail=1
    fi
}

log_info "========== Verify Model Registry (deployed topology) =========="

# 1. Workloads exist in their real namespaces.
kubectl wait --for=condition=available deployment/model-registry-deployment \
    -n "$MR_SERVER_NS" --timeout=120s >/dev/null 2>&1
check "registry server deployment available in $MR_SERVER_NS" $?
kubectl wait --for=condition=available deployment/model-registry-db \
    -n "$MR_SERVER_NS" --timeout=120s >/dev/null 2>&1
check "registry db deployment available in $MR_SERVER_NS" $?
kubectl wait --for=condition=available deployment/model-registry-ui \
    -n "$MR_UI_NS" --timeout=120s >/dev/null 2>&1
check "registry UI deployment available in $MR_UI_NS" $?

# 2. RBAC: the UI service account (kubeflow ns) must list clusterrolebindings,
#    otherwise its /api/v1/user BFF returns 500 and the dashboard page breaks.
kubectl auth can-i list clusterrolebindings \
    --as system:serviceaccount:"$MR_UI_NS":model-registry-ui 2>/dev/null \
    | grep -qx yes
check "model-registry-ui SA can list clusterrolebindings" $?
for crb in model-registry-ui-services-reader-binding \
           model-registry-retrieve-clusterrolebindings-binding \
           model-registry-create-sars-binding; do
    kubectl get clusterrolebinding "$crb" -o jsonpath='{.subjects[*].namespace}' 2>/dev/null \
        | grep -qw "$MR_UI_NS"
    check "ClusterRoleBinding $crb subject namespace == $MR_UI_NS" $?
done

# 3. The API VirtualService must point at the server in default ns, not the
#    non-existent model-registry-service.kubeflow host (upstream YAML bug).
kubectl get virtualservice model-registry -n "$MR_SERVER_NS" \
    -o jsonpath='{.spec.http[0].route[0].destination.host}' 2>/dev/null \
    | grep -q "model-registry-service.${MR_SERVER_NS}.svc"
check "API VirtualService host points at $MR_SERVER_NS service" $?

# 3b. Hub BFF per-namespace discovery wiring. The BFF lists
#     Services with label component=model-registry inside the dashboard
#     namespace; the server lives in default. The tenant ns gets a
#     selector-less alias Service whose external-address-rest annotation (used
#     in federated mode) points at the real default-ns DNS name.
TENANT_NS="${MR_TENANT_NS:-nereidforge}"
kubectl get svc -n "$TENANT_NS" model-registry-service \
    -o jsonpath='{.metadata.labels.component}' 2>/dev/null | grep -qx model-registry
check "registry alias Service exists in $TENANT_NS (component=model-registry)" $?
kubectl get svc -n "$TENANT_NS" model-registry-service \
    -o jsonpath='{.metadata.annotations.modelregistry\.kubeflow\.org/external-address-rest}' 2>/dev/null \
    | grep -q "model-registry-service.${MR_SERVER_NS}.svc.cluster.local:8080"
check "alias external-address-rest annotation -> $MR_SERVER_NS DNS" $?
kubectl get deploy model-registry-ui -n "$MR_UI_NS" \
    -o jsonpath='{.spec.template.spec.containers[0].args}' 2>/dev/null \
    | grep -q -- "--deployment-mode=federated"
check "model-registry-ui BFF runs in federated mode" $?
# The alias must remain selector-less / endpoint-less: a manual Endpoints
# pointing at another Service ClusterIP double-DNATs and times out. A
# not-found Endpoints resource or an empty subsets array are both healthy.
ep_subsets="$(kubectl get endpoints model-registry-service -n "$TENANT_NS" \
    -o jsonpath='{.subsets}' 2>/dev/null || true)"
[ -z "$ep_subsets" ]
check "alias Service has no static Endpoints (DNS resolution, not VIP)" $?

# 4. User-path probes THROUGH THE GATEWAY (the path browsers actually use:
#    node/hostNetwork -> istio-ingressgateway -> VirtualServices). Cross-ns
#    pods are blocked by the gateway NetworkPolicy, so use kubectl
#    port-forward from the deploy host, exactly like an operator does.
#    401 on the API without a bearer token is a PASS: the request reached the
#    registry service. 503 (bad upstream host) / 500 (UI RBAC broken) fail.
PF_PORT="${MR_VERIFY_PF_PORT:-18099}"
kubectl port-forward -n "$GW_NS" svc/istio-ingressgateway "${PF_PORT}:80" \
    >/tmp/mr-verify-pf.log 2>&1 &
PF_PID=$!
cleanup_pf() { kill "$PF_PID" 2>/dev/null || true; wait "$PF_PID" 2>/dev/null || true; }
trap cleanup_pf EXIT
for _ in $(seq 1 30); do
    if curl -sf -m 2 -o /dev/null "http://127.0.0.1:${PF_PORT}/" 2>/dev/null \
       || curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PF_PORT}/" 2>/dev/null | grep -qE '^[0-9]{3}$'; then
        break
    fi
    sleep 1
done

curl_gateway() {
    # curl_gateway <path> -> HTTP status code
    curl -s -m 10 -o /dev/null -w '%{http_code}' \
        -H "kubeflow-userid: ${DEX_STATIC_EMAIL}" \
        "http://127.0.0.1:${PF_PORT}$1"
}
# 401 without a browser oidc cookie is a HEALTHY response (service is up, only
# unauthenticated); the RBAC breakage produced 500 and a bad VS host 503, so
# healthy here means any non-5xx status.
curl_gateway "/model-registry/api/v1/user" | grep -qE '^(200|401|403)$'
check "UI BFF /model-registry/api/v1/user via gateway -> non-5xx (healthy)" $?
curl_gateway "/api/model_registry/v1alpha3/registered_models" | grep -qE '^(200|401)$'
check "registry API /api/model_registry via gateway -> 200/401 (not 503)" $?

# Authenticated, browser-identical check: Dex OIDC login with the static user,
# then exercise BFF discovery and the registry proxy. This is the only probe
# that catches a missing per-namespace alias (empty registry list in the UI).
PF_PORT="$PF_PORT" TENANT_NS="$TENANT_NS" \
    DEX_EMAIL="$DEX_STATIC_EMAIL" \
    DEX_PASSWORD="$DEX_STATIC_PASSWORD" \
    python3 - <<'PYEOF'
import html
import http.cookiejar
import json
import os
import re
import sys
import urllib.parse
import urllib.request

base = "http://127.0.0.1:%s" % os.environ["PF_PORT"]
ns = os.environ["TENANT_NS"]
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(
    urllib.request.HTTPRedirectHandler(),
    urllib.request.HTTPCookieProcessor(cj),
)
opener.addheaders = [("User-Agent", "mr-check/1.0")]

resp = opener.open(base + "/oauth2/start?rd=%2Fmodel-registry%2F", timeout=30)
page = resp.read().decode()
m = re.search(r'action="([^"]+)"', page)
if not m:
    print("dex login form not found: %s" % resp.geturl())
    sys.exit(1)
action = html.unescape(m.group(1))
if action.startswith("/"):
    action = base + action
form = urllib.parse.urlencode({
    "login": os.environ["DEX_EMAIL"],
    "password": os.environ["DEX_PASSWORD"],
}).encode()
opener.open(action, data=form, timeout=30).read()

resp = opener.open(
    base + "/model-registry/api/v1/model_registry?namespace=" + ns, timeout=30
)
regs = json.loads(resp.read().decode()).get("data") or []
if not regs:
    print("BFF discovered no registry in ns %s" % ns)
    sys.exit(1)
reg = regs[0]
if "model-registry-service.default.svc" not in (reg.get("serverAddress") or ""):
    print("registry not resolved through default-ns DNS: %s" % reg.get("serverAddress"))
    sys.exit(1)
resp = opener.open(
    base + "/model-registry/api/v1/model_registry/%s/registered_models?namespace=%s"
    % (reg["name"], ns),
    timeout=30,
)
payload = json.loads(resp.read().decode()) or {}
# The BFF wraps proxied MR responses in its own envelope: {"data": {"items": ...}}
items = payload.get("items")
if items is None and isinstance(payload.get("data"), dict):
    items = payload["data"].get("items")
if items is None:
    print("BFF proxy returned an unexpected envelope (no items key)")
    sys.exit(1)
# A fresh registry legitimately has zero models; the failure this probe exists
# to catch is a missing per-namespace alias (empty discovery above) or a 5xx
# from the proxy, not an empty model list. 2026-10-29 live install check.
print("  authenticated BFF discovery OK: %s, %d registered model(s)"
      % (reg["name"], len(items)))
PYEOF
check "authenticated BFF discovery + proxy via gateway (Dex login, models listed)" $?

cleanup_pf
trap - EXIT

if [ "$fail" != "0" ]; then
    log_info "Model Registry verification failed; review the FAIL items above"
    exit 1
fi
log_ok "Model Registry verification passed"
