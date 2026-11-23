#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# labelstudio.sh deploys the Label Studio Helm chart.

set -euo pipefail

PRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB_DIR="$PRE_ROOT/scripts/lib"
source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

CHARTS_DIR="$PRE_ROOT/charts"

log_info "========== Deploy Label Studio (Helm) =========="

: "${LABEL_STUDIO_STORAGE:=20Gi}"

read_password() {
    local prompt="$1"
    local var_name="$2"
    local value
    if [[ -n "${!var_name:-}" ]]; then
        value="${!var_name}"
        log_info "$prompt: ******** (from the environment)"
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

export LABEL_STUDIO_TOKEN="test-label-studio-token"

if [[ -z "$LABEL_STUDIO_TOKEN" ]]; then
    LABEL_STUDIO_TOKEN=$(gen_password 40)
    log_info "Generated a random Label Studio token"
fi

# Use the existing chart directory.
LS_CHART_DIR="$CHARTS_DIR/labelstudio"

# Resolve the pristine upstream image coordinate (lib/config.sh). Air-gapped
# nodes resolve it through the containerd registry mirror; imagePullSecret
# stays empty so the chart renders no in-pod pull secret (root treaty section 18).
docker_ref_parts "$IMG_LABEL_STUDIO"
LS_REGISTRY="${_REF_REGISTRY:-docker.io}"
LS_REPOSITORY="$_REF_REPOSITORY"
LS_TAG="$_REF_TAG"

# Render the chart values with the upstream image.
cat > "$LS_CHART_DIR/values.yaml" <<EOF
image:
  registry: $LS_REGISTRY
  repository: $LS_REPOSITORY
  tag: $LS_TAG
imagePullSecret: ""

storageClass: $STORAGE_CLASS

postgresql:
  host: $PG_CLUSTER_NAME-rw.nereidforge.svc.cluster.local
  port: 5432
  database: labelstudio
  user: $PG_DB_USER
  existingSecret: pg-${PG_CLUSTER_NAME}-credentials
  existingSecretKey: password

valkey:
  host: valkey-master.nereidforge.svc.cluster.local
  port: 6379

s3:
    existingSecret: seaweedfs-s3-credentials
    accessKeyKey: accessKey
    secretKeyKey: secretKey
    bucket: labelstudio
    endpoint: http://seaweedfs-s3.nereidforge.svc.cluster.local:8333
    useSSL: false

ingress:
  enabled: true
  className: traefik
  annotations:
    cert-manager.io/cluster-issuer: internal-ca
  host: labelstudio.$BASE_DOMAIN
  tls: true

persistence:
  size: $LABEL_STUDIO_STORAGE

resources:
  limits:
    cpu: "1000m"
    memory: "2Gi"
  requests:
    cpu: "500m"
    memory: "1Gi"

token: "$LABEL_STUDIO_TOKEN"

# Extra env for the LS container: trust the HTTPS origin used behind the TLS
# terminating ingress, otherwise Django's CSRF origin check rejects login POSTs
# with "Origin checking failed ... does not match any trusted origins".
env:
  - name: CSRF_TRUSTED_ORIGINS
    value: "https://labelstudio.$BASE_DOMAIN"
EOF

log_info "Deploy Label Studio..."
cd "$LS_CHART_DIR"
ls charts/*.tgz >/dev/null 2>&1 || helm dependency build 2>/dev/null || true
helm upgrade --install labelstudio . -n nereidforge --create-namespace --wait --timeout 10m

wait_deployment_ready "labelstudio" "nereidforge"

# Idempotent repair for a Label Studio first-run quirk (see pit 116): the stock
# image's bootstrap creates the admin Organization + OrganizationMember but can
# leave htx_user.active_organization_id NULL. Projects created over the API then
# get organization_id NULL, and task import 500s with
# "'NoneType' object has no attribute 'created_by'" while ML-backend webhook
# creation fails "NOT NULL constraint failed: webhook.organization_id". This
# sets the missing back-reference and adopts any orphaned projects. ORM-based,
# so it works on sqlite today and postgres after the DB wiring is fixed. Retry:
# on a fresh PVC the admin/org only exist once first-run migrations finish.
LS_REPAIR_PY='
from users.models import User
from organizations.models import Organization, OrganizationMember
from projects.models import Project
fixed_u = fixed_p = 0
for u in User.objects.all():
    if getattr(u, "active_organization_id", None) is None:
        m = (OrganizationMember.objects.filter(user=u, deleted_at__isnull=True)
             .select_related("organization").first())
        if m:
            u.active_organization = m.organization
            u.save(update_fields=["active_organization"])
            fixed_u += 1
for p in Project.objects.filter(organization__isnull=True):
    org = (Organization.objects.filter(created_by_id=p.created_by_id).first()
           if p.created_by_id else None) or Organization.objects.order_by("id").first()
    if org:
        p.organization = org
        p.save(update_fields=["organization_id"])
        fixed_p += 1
print("LS_ORG_REPAIR users=%d projects=%d" % (fixed_u, fixed_p))
'
for i in $(seq 1 30); do
    OUT=$(kubectl -n nereidforge exec deploy/labelstudio -i -- \
        python label_studio/manage.py shell <<<"$LS_REPAIR_PY" 2>&1) && echo "$OUT" | grep -q LS_ORG_REPAIR && {
        echo "$OUT" | grep LS_ORG_REPAIR; break; }
    sleep 5
done

log_ok "Label Studio deployment complete"
log_info "Access URL: https://labelstudio.$BASE_DOMAIN"
log_info "API Token: $LABEL_STUDIO_TOKEN"
