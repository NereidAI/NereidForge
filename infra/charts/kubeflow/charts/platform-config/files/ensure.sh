#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# platform-config ensure hook: Kubeflow platform-level wiring that must run AFTER
# all upstream component subcharts are applied. Migrated from the original
# 30_install-kubeflow.sh ensure_* functions; config comes from env (chart globals).
set -euo pipefail

MANIFESTS_DIR="${MANIFESTS_DIR:-/manifests}"
DEX_STATIC_EMAIL="${DEX_STATIC_EMAIL:-admin@nereidai.local}"
DEX_STATIC_PASSWORD_HASH="${DEX_STATIC_PASSWORD_HASH:-}"
TENANT="${TENANT:-nereidforge-admin}"
POD_CIDR="${POD_CIDR:-10.42.0.0/16}"

WFP_DIR="$MANIFESTS_DIR/.wfp-mr"
WFP_KUSTOMIZATION="$WFP_DIR/kustomization.yaml"

log_info() { echo "[INFO]  $*"; }
log_warn() { echo "[WARN]  $*" >&2; }
log_err()  { echo "[ERROR] $*" >&2; }
log_ok()   { echo "[OK]    $*"; }
retry() {
    local n=$1 s=$2; shift 2
    local i
    for ((i=0; i<n; i++)); do "$@" && return 0; sleep "$s"; done
    return 1
}

# Backfill istiod caBundle on sidecar-injector / validator webhooks.
ensure_istio_webhook_cabundle() {
    local cert64=""
    if ! retry 18 10 kubectl get secret istio-ca-secret -n istio-system >/dev/null 2>&1; then
        log_warn "istio-ca-secret never appeared, skip webhook caBundle backfill"
        return 0
    fi
    cert64=$(kubectl get secret istio-ca-secret -n istio-system -o jsonpath='{.data.ca-cert\.pem}')
    [[ -n "$cert64" ]] || { log_warn "istio-ca-secret has no ca-cert.pem, skip"; return 0; }
    for cfg in istio-sidecar-injector istio-validator-istio-system; do
        kubectl get mutatingwebhookconfiguration "$cfg" &>/dev/null || kubectl get validatingwebhookconfiguration "$cfg" &>/dev/null || continue
        python3 - "$cfg" "$cert64" <<'PYEOF'
import sys, json, subprocess
cfg, cert64 = sys.argv[1], sys.argv[2]
kind = "mutatingwebhookconfiguration"
get = subprocess.run(["kubectl","get",kind,cfg,"-o","json"],capture_output=True,text=True)
if get.returncode != 0:
    kind = "validatingwebhookconfiguration"
    get = subprocess.run(["kubectl","get",kind,cfg,"-o","json"],capture_output=True,text=True)
d = json.loads(get.stdout)
ops = []
for i, h in enumerate(d["webhooks"]):
    if not h.get("clientConfig", {}).get("caBundle"):
        ops.append({"op":"add","path":f"/webhooks/{i}/clientConfig/caBundle","value":cert64})
if ops:
    r = subprocess.run(["kubectl","patch",kind,cfg,"--type=json","-p",json.dumps(ops)],capture_output=True,text=True)
    sys.stderr.write(r.stderr or "")
    sys.exit(r.returncode)
PYEOF
    done
    log_ok "istio webhook caBundle backfilled"
}

# Self-signed internal CA ClusterIssuer every Ingress / kubeflow-gateway depends on.
ensure_internal_ca() {
    if kubectl get secret internal-ca-secret -n cert-manager &>/dev/null; then
        log_info "internal-ca already exists, skip"
        return 0
    fi
    for d in cert-manager cert-manager-webhook cert-manager-cainjector; do
        if ! kubectl wait --for=condition=Available deployment/"$d" -n cert-manager --timeout=300s >/dev/null 2>&1; then
            log_err "cert-manager deployment $d not Available"
            return 1
        fi
    done
    retry 12 10 kubectl get endpoints cert-manager-webhook -n cert-manager -o jsonpath='{.subsets}' >/dev/null 2>&1 \
        || { log_err "cert-manager-webhook still has no endpoints"; return 1; }
    log_info "creating self-signed root CA + internal CA ClusterIssuer (internal-ca)..."
    cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: selfsigned-ca
spec:
  selfSigned: {}
---
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: internal-ca
spec:
  ca:
    secretName: internal-ca-secret
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: internal-ca-cert
  namespace: cert-manager
spec:
  isCA: true
  commonName: nereidforge-internal-ca
  secretName: internal-ca-secret
  privateKey:
    algorithm: RSA
    size: 2048
  duration: 87600h
  renewBefore: 720h
  issuerRef:
    name: selfsigned-ca
    kind: ClusterIssuer
    group: cert-manager.io
EOF
    log_ok "internal-ca ClusterIssuer ready"
}

# Provision the main business tenant as a KF Profile.
ensure_tenant_profile() {
    cat <<EOF | kubectl apply -f - >/dev/null
apiVersion: kubeflow.org/v1
kind: Profile
metadata:
  name: $TENANT
spec:
  owner:
    kind: User
    name: ${DEX_STATIC_EMAIL}
EOF
    if ! retry 12 10 kubectl get ns "$TENANT" >/dev/null 2>&1; then
        log_err "tenant namespace $TENANT never appeared (profile-controller failed?)"
        return 1
    fi
    kubectl label ns "$TENANT" pod-security.kubernetes.io/enforce=privileged --overwrite >/dev/null
    kubectl apply -f - >/dev/null <<EOF
apiVersion: security.istio.io/v1
kind: AuthorizationPolicy
metadata:
  name: edge-proxy-incluster-allow
  namespace: $TENANT
spec:
  action: ALLOW
  rules:
    - from:
        - source:
            ipBlocks: ["$POD_CIDR"]
EOF
    log_ok "tenant Profile $TENANT ready"
}

# Model Registry (Kubeflow Hub backend + db + UI) into the main tenant.
ensure_model_registry() {
    mkdir -p "$WFP_DIR"
    cat > "$WFP_KUSTOMIZATION" <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- ../applications/hub/upstream/overlays/postgres
- ../applications/hub/upstream/options/istio
- ../applications/hub/upstream/options/ui/overlays/istio
namespace: $TENANT
patches:
- target:
    group: networking.istio.io
    version: v1alpha3
    kind: VirtualService
    name: model-registry
  patch: |-
    - op: replace
      path: /spec/gateways/0
      value: kubeflow/kubeflow-gateway
    - op: replace
      path: /spec/http/0/route/0/destination/host
      value: model-registry-service.$TENANT.svc.cluster.local
- target:
    group: networking.istio.io
    version: v1alpha3
    kind: VirtualService
    name: model-registry-ui
  patch: |-
    - op: replace
      path: /spec/gateways/0
      value: kubeflow/kubeflow-gateway
    - op: replace
      path: /spec/http/0/route/0/destination/host
      value: model-registry-ui-service.$TENANT.svc.cluster.local
- target:
    group: networking.istio.io
    version: v1alpha3
    kind: DestinationRule
    name: model-registry-service
  patch: |-
    - op: replace
      path: /spec/host
      value: model-registry-service.$TENANT.svc.cluster.local
- target:
    group: networking.istio.io
    version: v1alpha3
    kind: DestinationRule
    name: model-registry-ui
  patch: |-
    - op: replace
      path: /spec/host
      value: model-registry-ui-service.$TENANT.svc.cluster.local
EOF
    local rendered
    rendered=$(mktemp)
    if ! kustomize build "$WFP_DIR" > "$rendered"; then
        log_err "kustomize build failed: model-registry"
        rm -f "$rendered"
        return 1
    fi
    # Only the tenant-specific Dex identity is substituted; image references
    # stay at their pristine upstream coordinates (root treaty section 18).
    sed -i -E \
      -e "s#^  DEX_USER_PASSWORD: .*#  DEX_USER_PASSWORD: ${DEX_STATIC_PASSWORD_HASH}#" \
      -e "s#email: user@example.com#email: ${DEX_STATIC_EMAIL}#g" \
      "$rendered"
    if [ ! -s "$rendered" ]; then
        log_info "  (no resources to apply, skipping) model-registry"
        rm -f "$rendered"
        return 0
    fi
    if ! kubectl apply -f "$rendered" 2>"$rendered.err"; then
        if grep -qE "262144|Too long" "$rendered.err"; then
            retry 12 10 kubectl apply --server-side=true --field-manager=wfp --force-conflicts -f "$rendered" \
                || { rm -f "$rendered" "$rendered.err"; return 1; }
        elif ! retry 12 10 kubectl apply -f "$rendered"; then
            log_err "kubectl apply failed after retries: model-registry"
            rm -f "$rendered" "$rendered.err"
            return 1
        fi
    fi
    rm -f "$rendered" "$rendered.err" "$WFP_KUSTOMIZATION"
    log_ok "Model Registry deployed into $TENANT (via kubeflow-gateway)"
}

ensure_kubeflow_auth_rbac() {
    local f="/scripts/kubeflow-auth-rbac.yaml"
    [[ -f "$f" ]] || { log_warn "missing $f, skip kubeflow-auth-rbac ensure"; return 0; }
    kubectl apply -f "$f" \
        || { log_err "kubectl apply kubeflow-auth-rbac failed"; return 1; }
    log_ok "kubeflow auth/RBAC wiring ensured"
}

log_info "========== Kubeflow platform-config wiring =========="
ensure_internal_ca
ensure_istio_webhook_cabundle
ensure_tenant_profile
ensure_model_registry
ensure_kubeflow_auth_rbac
log_ok "platform-config wiring done"
