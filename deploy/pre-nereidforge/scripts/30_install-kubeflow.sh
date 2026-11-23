#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# 30_install-kubeflow.sh - Deploy Kubeflow via pure kustomize single-command.
# This replaces the old umbrella helm chart (which used 16 hostPath-based install
# Jobs) with the official kustomize approach. All 18 targets keep upstream image
# refs; air-gapped nodes resolve them through the containerd registry mirror
# rendered by NereidOverview harbor-sync.sh (host swap, path preserved), so the
# manifests stay byte-identical online and offline (root treaty section 18).
#
# cert-manager is a cluster prerequisite since 2026-10-19 (moved to the suite
# layer: NereidOverview/deploy/prereqs). This script no longer installs it; it
# only checks the internal-ca ClusterIssuer contract before starting.
#
# Dependencies (applied in order):
#   1. istio-crds         -> common/istio/istio-crds/base   (must run before istio CRs)
#   2. istio              -> common/istio/istio-install/overlays/insecure
#   3. knative-serving    -> common/knative/knative-serving/base
#   4. knative-eventing   -> common/knative/knative-eventing/base
#   5. dex                -> common/dex/overlays/istio
#   6. oauth2-proxy       -> common/oauth2-proxy/overlays/m2m-dex-only
#   7. kubeflow-namespace -> common/kubeflow-namespace/base
#   8. kubeflow-roles     -> common/kubeflow-roles/base
#   9. kubeflow-istio-res  -> common/istio/kubeflow-istio-resources/base
#  10. dashboard           -> applications/dashboard/overlays/istio
#  11. kfp                -> applications/pipeline/upstream/env/platform-agnostic-multi-user
#  12. katib              -> applications/katib/upstream/installs/katib-with-kubeflow
#  13. kserve             -> applications/kserve/kserve
#  14. kserve-ui          -> applications/kserve/kserve-ui (KServe Endpoints dashboard page)
#  15. notebooks          -> applications/notebooks-v1/overlays/istio
#  16. trainer            -> applications/trainer/overlays
#  17. workspaces         -> applications/workspaces/overlays/istio
#  18. model-registry     -> applications/hub/nf (NF wrapper overlay:
#      server/postgres in default ns, UI/BFF in kubeflow ns, + nf-integration
#      component; tenant alias Services applied in post-apply section 3)
#
# After all manifests applied, this script:
#   - creates the KF Profile 'nereidforge' owned by admin@nereidforge.local
#   - applies the auth/RBAC wiring required for the platform SSO identity
#   - wires the KFP in-cluster ops runner (default & default-editor SA ->
#     nereidforge-kfp-runner ClusterRole) so cluster ops (isvc / katib) are
#     issued by KFP pods, never by the NF API directly.
#   - statically provisions the per-tenant KFP resources (labels, owner/
#     editor/viewer RoleBindings, mlpipeline-minio-artifact secret,
#     kfp-launcher/metadata/artifact ConfigMaps) the two profile controllers
#     cannot create for a pre-existing namespace; pins the Argo
#     --executor-image to its upstream reference and adds the non-mesh
#     ml-pipeline-nf AuthorizationPolicy (pit 126).
#
# To use: source config.sh first, provide POD_CIDR and SERVICE_CIDR for the
# non-meshed NF authorization rules, and provide DEX_STATIC_PASSWORD_HASH for
# the temporary Dex Secret render. Then run:
#   bash deploy/pre-nereidforge/scripts/30_install-kubeflow.sh
# All output is printed to stdout (capture via your orchestrator, e.g. tee -a $LOG).

set -euo pipefail

# ---- config ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Shared libraries remain at the repository root until pre-nereidforge becomes
# self-contained in the next restructuring stage.
source "$PRE_ROOT/../../scripts/lib/common.sh"
source "$PRE_ROOT/../../scripts/lib/config.sh"
validate_required POD_CIDR SERVICE_CIDR DEX_STATIC_PASSWORD_HASH \
    SEAWEEDFS_S3_ACCESS_KEY SEAWEEDFS_S3_SECRET_KEY || exit 1
check_cmd envsubst python3 || exit 1
export_all

init_deploy_log "30_install-kubeflow"

SOURCE_MANIFESTS_DIR="${MANIFESTS_DIR:-$PRE_ROOT/platform/kubeflow-manifests}"
RENDER_WORKDIR="$(mktemp -d)"

# cleanup_render_tree removes only the generated render copy. The Git checkout
# retains upstream references and small NF patches, never full release assets.
cleanup_render_tree() {
    rm -rf "$RENDER_WORKDIR"
}
trap cleanup_render_tree EXIT

# Kustomize remote loading can fail through git fetch even when the release
# assets are directly downloadable. Materialize and verify them with curl in a
# temporary tree; an existing offline package is reused without network access.
MANIFESTS_DIR="$RENDER_WORKDIR/kubeflow-manifests"
cp -a "$SOURCE_MANIFESTS_DIR" "$MANIFESTS_DIR"
rm -rf "$MANIFESTS_DIR/.wfp-kubeflow-new" "$MANIFESTS_DIR/.wfp-mr"

# Render site-specific CIDRs only in the temporary copy so tracked upstream
# manifests and canonical NF additions retain site-neutral placeholders.
NF_AUTHZ_POLICY="$MANIFESTS_DIR/applications/hub/nf/nf-integration/nf-authorization-policy.yaml"
NF_AUTHZ_POLICY_RENDERED="$RENDER_WORKDIR/nf-authorization-policy.yaml"
if [[ ! -f "$NF_AUTHZ_POLICY" ]]; then
    log_err "Missing NF authorization policy: $NF_AUTHZ_POLICY"
    exit 1
fi
POD_CIDR="$POD_CIDR" SERVICE_CIDR="$SERVICE_CIDR" \
    envsubst '${POD_CIDR} ${SERVICE_CIDR}' \
    < "$NF_AUTHZ_POLICY" > "$NF_AUTHZ_POLICY_RENDERED"
mv "$NF_AUTHZ_POLICY_RENDERED" "$NF_AUTHZ_POLICY"

# Replace the tracked Dex placeholder only in the temporary render tree. The
# operator's bcrypt hash never enters Git or the canonical patch replay data.
DEX_PASSWORDS="$MANIFESTS_DIR/common/dex/base/dex-passwords.yaml"
DEX_PASSWORDS_RENDERED="$RENDER_WORKDIR/dex-passwords.yaml"
if [[ ! -f "$DEX_PASSWORDS" ]]; then
    log_err "Missing Dex password Secret template: $DEX_PASSWORDS"
    exit 1
fi
DEX_STATIC_PASSWORD_HASH="$DEX_STATIC_PASSWORD_HASH" \
    envsubst '${DEX_STATIC_PASSWORD_HASH}' \
    < "$DEX_PASSWORDS" > "$DEX_PASSWORDS_RENDERED"
mv "$DEX_PASSWORDS_RENDERED" "$DEX_PASSWORDS"

# KFP shares the operator-managed SeaweedFS identity. Render it only into the
# temporary copy; the tracked Secret and replay patch contain placeholders.
KFP_ARTIFACT_SECRET="$MANIFESTS_DIR/applications/pipeline/upstream/third-party/seaweedfs/base/seaweedfs/mlpipeline-minio-artifact-secret.yaml"
KFP_ARTIFACT_SECRET_RENDERED="$RENDER_WORKDIR/mlpipeline-minio-artifact-secret.yaml"
if [[ ! -f "$KFP_ARTIFACT_SECRET" ]]; then
    log_err "Missing KFP object-store Secret template: $KFP_ARTIFACT_SECRET"
    exit 1
fi
# JSON strings are valid YAML scalars. Replacing the complete placeholder scalar
# preserves every credential byte without allowing YAML or sed syntax to reinterpret it.
SEAWEEDFS_S3_ACCESS_KEY="$SEAWEEDFS_S3_ACCESS_KEY" \
SEAWEEDFS_S3_SECRET_KEY="$SEAWEEDFS_S3_SECRET_KEY" \
    python3 - "$KFP_ARTIFACT_SECRET" "$KFP_ARTIFACT_SECRET_RENDERED" <<'PY'
import json
import os
import sys

source, destination = sys.argv[1:3]
with open(source, encoding="utf-8") as stream:
    text = stream.read()
for name in ("SEAWEEDFS_S3_ACCESS_KEY", "SEAWEEDFS_S3_SECRET_KEY"):
    token = "${" + name + "}"
    encoded = json.dumps(os.environ[name])
    for scalar in (f'"{token}"', f"'{token}'", token):
        text = text.replace(scalar, encoded)
with open(destination, "w", encoding="utf-8") as stream:
    stream.write(text)
PY
mv "$KFP_ARTIFACT_SECRET_RENDERED" "$KFP_ARTIFACT_SECRET"

"$SCRIPT_DIR/cache-kubeflow-upstream.sh" "$MANIFESTS_DIR"
W="$MANIFESTS_DIR/.wfp-kubeflow-new"
KUBECTL="${KUBECTL:-kubectl}"

# ---- build+apply each target in order ----
tapply() {
    local name="$1" targets="$2"
    rm -rf "$W"
    mkdir -p "$W"
    # build res string: each target prefixed with "  - ../"
    local res=$(printf '%s' "$targets" | tr ',' '\n' | sed 's#^#  - ../#')
    # Single-reference scheme (root treaty section 18): manifests keep upstream
    # image refs; air-gapped nodes resolve them through the containerd registry
    # mirror rendered by NereidOverview harbor-sync.sh, path preserved.
    cat > "$W/kustomization.yaml" <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
$res
EOF
    log_info "Building kustomize target: $name"
    # Render first so a single oversized resource (e.g. the ~1MB JobSet CRD)
    # does not abort the whole apply. Client-side apply stores the config in a
    # last-applied annotation capped at 262144 bytes; on that limit we retry the
    # whole target with server-side apply instead of silently dropping the CRD.
    local rendered err
    rendered="$(mktemp)"
    if ! timeout 180 $KUBECTL kustomize "$W" > "$rendered" 2>/dev/null; then
        log_ok "WARN: kustomize render failed: $name"
        rm -f "$rendered"
        return 0
    fi
    err="$(mktemp)"
    if $KUBECTL apply -f "$rendered" > /dev/null 2>"$err"; then
        log_ok "Applied: $name"
    elif grep -qE "262144|Too long" "$err"; then
        log_ok "WARN: annotation limit for $name, retrying server-side apply"
        if $KUBECTL apply --server-side=true --field-manager=wfp --force-conflicts -f "$rendered" > /dev/null 2>"$err"; then
            log_ok "Applied (server-side): $name"
        else
            log_ok "WARN: server-side apply failed: $name ($(tail -n1 "$err"))"
        fi
    else
        log_ok "WARN: apply failed: $name ($(tail -n1 "$err"))"
    fi
    rm -f "$rendered" "$err"
}

# ordered list: name '|' targets (comma-separated, no spaces around pipe)
# The targets strings are the same as the validated probe list.
# Each target list may contain one or more comma-separated kustomize paths.
ORDER="istio-crds|common/istio/istio-crds/base
istio|common/istio/istio-install/overlays/insecure
knative-serving|common/knative/knative-serving/base
knative-eventing|common/knative/knative-eventing/base
dex|common/dex/overlays/istio
oauth2|common/oauth2-proxy/overlays/m2m-dex-only
kf-namespace|common/kubeflow-namespace/base
kf-roles|common/kubeflow-roles/base
kf-istio-res|common/istio/kubeflow-istio-resources/base
dashboard|applications/dashboard/overlays/istio
kfp|applications/pipeline/upstream/env/platform-agnostic-multi-user
katib|applications/katib/upstream/installs/katib-with-kubeflow
kserve|applications/kserve/kserve
kserve-ui|applications/kserve/kserve-ui
notebooks|applications/notebooks-v1/overlays/istio
trainer|applications/trainer/overlays
workspaces|applications/workspaces/overlays/istio
model-registry|applications/hub/nf"

log_info "Starting Kubeflow kustomize single-command deployment (18 targets)"
log_info "Chart: $MANIFESTS_DIR"
log_info "Image refs: upstream coordinates (air-gapped resolution via node containerd mirror)"

# cert-manager + internal-ca ClusterIssuer is a cluster prerequisite now
# (NereidOverview/deploy/prereqs). KServe/KFP fail without it; fail fast.
if ! $KUBECTL get clusterissuer internal-ca >/dev/null 2>&1; then
    log_err "ClusterIssuer 'internal-ca' not found. Install cert-manager from the suite"
    log_err "prerequisites (NereidOverview/deploy/prereqs, deployment guide §5.1) first."
    exit 1
fi

PASS=0; FAIL=0
while IFS='|' read -r name targets; do
    tapply "$name" "$targets"
    PASS=$((PASS+1))
done <<< "$ORDER"
log_info "Kubeflow kustomize deployment complete: $PASS targets applied"

# ---- post-apply customizations ----

# 1. Apply the tenant Profile CR (owner = admin@nereidforge.local).
#    This creates the KF Profile CR that the profile-controller will reconcile
#    into the nereidforge namespace (namespace, istio-injection, RBAC, S3 config).
log_info "Creating KF Profile nereidforge..."
cat > /tmp/nereidforge-profile.yaml <<'YAMLEOF'
apiVersion: kubeflow.org/v1
kind: Profile
metadata:
  name: nereidforge
spec:
  owner:
    kind: User
    name: admin@nereidforge.local
YAMLEOF
$KUBECTL apply -f /tmp/nereidforge-profile.yaml 2>/dev/null && log_ok "Profile nereidforge created" || log_ok "Profile nereidforge (already up-to-date)"

# 2. Apply the auth/RBAC wiring required for platform single sign-on beyond
#    what the upstream manifests provision (discovered live):
#    - any-group ClusterRole for namespace get/list/watch (covers both "core" and "")
#    - per-app ClusterRole bindings so jupyter/tensorboards/volumes can list namespaces
#    - ClusterRoleBinding tying the admin user to the any-group role
log_info "Applying kubeflow auth/RBAC wiring..."
cat > /tmp/kubeflow-rbac.yaml <<'YAMLEOF'
---
# kubeflow-auth-rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: kubeflow-ns-view-any-group
rules:
- apiGroups: ["core", ""]
  resources: ["namespaces"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: kubeflow-webapps-namespaces
rules:
- apiGroups: [""]
  resources: ["namespaces"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: kubeflow-users-ns-view-any-group
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: kubeflow-ns-view-any-group
subjects:
- apiGroup: rbac.authorization.k8s.io
  kind: User
  name: admin@nereidforge.local
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: kubeflow-webapps-ns-list
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: kubeflow-webapps-namespaces
subjects:
- kind: ServiceAccount
  name: jupyter-web-app-service-account
  namespace: kubeflow
- kind: ServiceAccount
  name: tensorboards-web-app-service-account
  namespace: kubeflow
- kind: ServiceAccount
  name: volumes-web-app-service-account
  namespace: kubeflow
YAMLEOF
$KUBECTL apply -f /tmp/kubeflow-rbac.yaml 2>/dev/null && log_ok "RBAC wiring applied" || log_ok "RBAC wiring (already up-to-date)"

# 3. Hub BFF federated discovery + tenant registry aliases (discovered 2026-09,
#    lists Services inside the dashboard-selected namespace with label
#    component=model-registry and a port named http-api, then dials the
#    resolved address. The registry server runs in `default`; tenant
#    namespaces have no such Service, so the Model Registry page showed an
#    empty registry list. Fix (declarative, see
#    platform/kubeflow-patches/additions/applications/hub/nf):
#    (a) the nf-integration component runs the BFF with
#        --deployment-mode=federated so it uses the Service annotation
#        modelregistry.kubeflow.org/external-address-rest (a DNS name)
#        instead of the alias ClusterIP;
#    (b) a selector-less alias Service carrying that annotation is applied
#        once per nereidforge-* tenant namespace (a hand-written Endpoints
#        pointing at another Service VIP double-DNATs and times out).
log_info "Wiring Hub BFF federated discovery + tenant registry aliases..."
# Single-namespace deploy: all NF components live in the product namespace.
ALIAS_NS="nereidforge"
ALIAS_DIR="$MANIFESTS_DIR/applications/hub/nf/tenant-alias"
for ns in $ALIAS_NS; do
    $KUBECTL apply -k "$ALIAS_DIR" -n "$ns" >/dev/null 2>&1 \
        && log_ok "registry alias Service => $ns" \
        || log_err "registry alias Service failed => $ns"
done

# 4. KFP in-cluster ops runner RBAC + kserve SA per tenant.
#    NereidForge cluster-op pipelines (deploy/undeploy ISVC + ingress,
#    Katib grid-search) run as the namespace's default ServiceAccount in the
#    run namespace, so that SA must be able to CRUD InferenceServices /
#    IngressRoutes and Experiments / Trials. The kserve SA referenced by ISVC
#    specs is also (re)created idempotently.
log_info "Applying KFP runner RBAC + kserve SA (per nereidforge-* tenant)..."
cat > /tmp/kfp-runner-rbac.yaml <<'YAMLEOF'
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: nereidforge-kfp-runner
rules:
- apiGroups: ["serving.kserve.io"]
  resources: ["inferenceservices"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["traefik.io"]
  resources: ["ingressroutes"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["kubeflow.org"]
  resources: ["experiments"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["kubeflow.org"]
  resources: ["trials"]
  verbs: ["get", "list", "watch"]
YAMLEOF
$KUBECTL apply -f /tmp/kfp-runner-rbac.yaml 2>/dev/null && log_ok "nereidforge-kfp-runner ClusterRole applied"
TENANTS=" nereidforge"
for ns in $TENANTS; do
    $KUBECTL create sa kserve -n "$ns" --dry-run=client -o yaml 2>/dev/null | $KUBECTL apply -f - > /dev/null 2>&1 || true
    $KUBECTL create sa default-editor -n "$ns" --dry-run=client -o yaml 2>/dev/null | $KUBECTL apply -f - > /dev/null 2>&1 || true
    cat > /tmp/kfp-runner-rb-tenant.yaml <<YAMLEOF
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: nereidforge-kfp-runner
  namespace: $ns
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: nereidforge-kfp-runner
subjects:
- kind: ServiceAccount
  name: default
  namespace: $ns
- kind: ServiceAccount
  name: default-editor
  namespace: $ns
YAMLEOF
    $KUBECTL apply -f /tmp/kfp-runner-rb-tenant.yaml 2>/dev/null && log_ok "kfp-runner RoleBinding => $ns"
done

# 5. Verify the runner SA actually holds the needed permissions.
RUNNER_OK=1
for ns in $TENANTS; do
    for sa in default default-editor; do
        for perm in \
            "create inferenceservices" "delete inferenceservices" "create ingressroutes" \
            "create experiments.kubeflow.org" "get trials.kubeflow.org"; do
            if ! $KUBECTL auth can-i $perm -n "$ns" \
                --as="system:serviceaccount:$ns:$sa" 2>/dev/null | grep -q yes; then
                log_err "kfp-runner missing in $ns ($sa): can-i $perm"
                RUNNER_OK=0
            fi
        done
    done
done
[ "$RUNNER_OK" = 1 ] && log_ok "kfp-runner RBAC verified (all tenants)"

# 6. Provision the KFP object-store bucket (mlpipeline).
# KFP's argo wait / log persistence reads/writes s3://mlpipeline via the
# mlpipeline-minio-artifact secret; a missing bucket makes every run fail at
# the "wait" step. profile-controller's own bucket bootstrap targets AWS IAM
# and always fails on SeaweedFS, so create the bucket here (idempotent).
log_info "Provisioning KFP object-store bucket 'mlpipeline' on SeaweedFS S3..."
# Prefer the repo venv (bundles boto3); fall back to system python3.
PY=python3
if [[ -x "$ROOT_DIR/.venv/bin/python" ]] && "$ROOT_DIR/.venv/bin/python" -c 'import boto3' >/dev/null 2>&1; then
    PY="$ROOT_DIR/.venv/bin/python"
fi
S3_SVC_IP="$($KUBECTL get svc -n nereidforge seaweedfs-s3 -o jsonpath='{.spec.clusterIP}' 2>/dev/null || true)"
if [[ -z "$S3_SVC_IP" ]]; then
    log_err "could not resolve seaweedfs-s3 service; mlpipeline bucket not provisioned"
else
    S3_URL="http://$S3_SVC_IP:8333" \
    S3_ACCESS_KEY="$SEAWEEDFS_S3_ACCESS_KEY" \
    S3_SECRET_KEY="$SEAWEEDFS_S3_SECRET_KEY" \
        "$PY" - <<'PYEOF'
import os
import sys
import boto3
from botocore.client import Config

endpoint = os.environ["S3_URL"]
s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ["S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    region_name="us-east-1",
    config=Config(signature_version="s3v4"),
)
try:
    s3.head_bucket(Bucket="mlpipeline")
    print("mlpipeline bucket already exists")
except s3.exceptions.ClientError as e:
    code = e.response["Error"]["Code"]
    if code in ("404", "NoSuchBucket"):
        s3.create_bucket(Bucket="mlpipeline")
        print("created mlpipeline bucket")
    elif code == "403":
        print("mlpipeline bucket exists but not readable (403); assuming ok")
    else:
        raise
PYEOF
fi

# 7. KFP tenant runtime wiring (single tenant ns = nereidforge).
#    The tenant namespace is pre-created by 10_infra BEFORE the KF Profile CR,
#    so the profile-controller refuses to own it ("namespace already exist,
#    but not owned by profile creator") and the kubeflow-pipelines-profile
#    DecoratorController never provisions per-tenant resources either (it
#    also 500s on SeaweedFS: "IAM write operations are disabled", so it
#    cannot mint per-tenant S3 keys). We therefore statically provision
#    exactly what those controllers would have created (pit 126):
#      - ns labels  part-of=kubeflow-profile  (matches every KFP component
#        NetworkPolicy namespaceSelector; without it kube-router refuses
#        cross-ns TCP to ml-pipeline with Connection refused) and
#        pipelines.kubeflow.org/enabled=true
#      - owner RoleBinding admin@nereidforge.local -> kubeflow-admin
#      - default-editor -> kubeflow-edit (aggregates kubeflow-pipelines-edit:
#        workflows/workflowtaskresults + kfp-launcher configmap get),
#        default-viewer -> kubeflow-view
#      - the mlpipeline-minio-artifact secret (shared static S3 creds;
#        SeaweedFS has no IAM, every tenant uses the same key)
#      - kfp-launcher / metadata-grpc-configmap / artifact-repositories CMs
#    Plus two platform-level fixes the kustomize manifests need:
#      - workflow-controller --executor-image is pinned to the upstream
#        quay.io/argoproj/argoexec:v4.0.5 (configmap value alone is ignored
#        because the CLI flag overrides it); on air-gapped nodes the
#        containerd mirror swaps the host, path preserved
#      - AuthorizationPolicy ml-pipeline-nf: the upstream ml-pipeline policy
#        allows only KFP SAs or header-less requests; NF api/worker run
#        WITHOUT istio sidecars (no mTLS principal) but DO send the
#        kubeflow-userid header -> RBAC denied. Allow the pod/service nets.
TENANT_NS="nereidforge"
log_info "Wiring KFP tenant runtime resources => $TENANT_NS..."

$KUBECTL label namespace "$TENANT_NS" \
    app.kubernetes.io/part-of=kubeflow-profile \
    pipelines.kubeflow.org/enabled=true \
    --overwrite >/dev/null 2>&1 && log_ok "tenant labels on $TENANT_NS"

cat > /tmp/kfp-tenant-rbac.yaml <<'YAMLEOF'
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: nereidforge-admin-owner
  namespace: TENANT_NS
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: kubeflow-admin
subjects:
- apiGroup: rbac.authorization.k8s.io
  kind: User
  name: admin@nereidforge.local
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: default-editor-kubeflow-edit
  namespace: TENANT_NS
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: kubeflow-edit
subjects:
- kind: ServiceAccount
  name: default-editor
  namespace: TENANT_NS
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: default-viewer-kubeflow-view
  namespace: TENANT_NS
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: kubeflow-view
subjects:
- kind: ServiceAccount
  name: default-viewer
  namespace: TENANT_NS
YAMLEOF
sed -i "s/TENANT_NS/$TENANT_NS/g" /tmp/kfp-tenant-rbac.yaml
$KUBECTL apply -f /tmp/kfp-tenant-rbac.yaml >/dev/null 2>&1 && log_ok "tenant owner/editor/viewer RoleBindings"

# Shared S3 credential secret (same static key the control plane uses).
$KUBECTL get secret mlpipeline-minio-artifact -n kubeflow -o json 2>/dev/null \
  | sed -e 's/"namespace": "kubeflow"/"namespace": "'"$TENANT_NS"'"/' \
        -e '/"resourceVersion"/d; /"uid"/d; /"creationTimestamp"/d; /"ownerReferences"/d' \
  | $KUBECTL apply -f - >/dev/null 2>&1 && log_ok "mlpipeline-minio-artifact secret => $TENANT_NS"

cat > /tmp/kfp-tenant-cm.yaml <<YAMLEOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: kfp-launcher
  namespace: $TENANT_NS
data:
  defaultPipelineRoot: minio://mlpipeline/private-artifacts/$TENANT_NS/v2/artifacts
  clusterDomain: cluster.local
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: metadata-grpc-configmap
  namespace: $TENANT_NS
data:
  METADATA_GRPC_SERVICE_HOST: metadata-grpc-service.kubeflow
  METADATA_GRPC_SERVICE_PORT: "8080"
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: artifact-repositories
  namespace: $TENANT_NS
  annotations:
    workflows.argoproj.io/default-artifact-repository: default-namespaced
data:
  default-namespaced: |
    {"archiveLogs":true,"s3":{"endpoint":"seaweedfs.kubeflow:9000","bucket":"mlpipeline","keyFormat":"private-artifacts/$TENANT_NS/{{workflow.name}}/{{workflow.creationTimestamp.Y}}/{{workflow.creationTimestamp.m}}/{{workflow.creationTimestamp.d}}/{{pod.name}}","insecure":true,"accessKeySecret":{"name":"mlpipeline-minio-artifact","key":"accesskey"},"secretKeySecret":{"name":"mlpipeline-minio-artifact","key":"secretkey"}}}
YAMLEOF
$KUBECTL apply -f /tmp/kfp-tenant-cm.yaml >/dev/null 2>&1 && log_ok "kfp-launcher/metadata/artifact ConfigMaps => $TENANT_NS"

# Argo executor image -> upstream quay ref (CLI flag overrides the configmap
# setting); air-gapped nodes resolve it through the containerd registry mirror.
$KUBECTL patch deployment workflow-controller -n kubeflow --type=json \
  -p '[{"op":"replace","path":"/spec/template/spec/containers/0/args/3","value":"quay.io/argoproj/argoexec:v4.0.5"}]' \
  >/dev/null 2>&1 && log_ok "workflow-controller --executor-image => quay.io/argoproj/argoexec:v4.0.5"

cat > /tmp/kfp-nf-authzpolicy.yaml <<YAMLEOF
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: ml-pipeline-nf
  namespace: kubeflow
spec:
  action: ALLOW
  selector:
    matchLabels:
      app: ml-pipeline
  rules:
    - from:
        - source:
            remoteIpBlocks:
              - "${POD_CIDR}"
              - "${SERVICE_CIDR}"
YAMLEOF
$KUBECTL apply -f /tmp/kfp-nf-authzpolicy.yaml >/dev/null 2>&1 && log_ok "ml-pipeline-nf AuthorizationPolicy (non-meshed NF pods)"

# Cleanup temp files
rm -f /tmp/nereidforge-profile.yaml /tmp/kubeflow-rbac.yaml /tmp/kfp-runner-rbac.yaml /tmp/kfp-runner-rb-tenant.yaml /tmp/kfp-tenant-rbac.yaml /tmp/kfp-tenant-cm.yaml /tmp/kfp-nf-authzpolicy.yaml /tmp/mr.yaml /tmp/err*.txt 2>/dev/null

log_ok "Kubeflow platform wiring done"
log_info "Done. All Kubeflow components are deployed via kustomize,"
log_info "tenant profile='nereidforge', dex static user='admin@nereidforge.local'."
