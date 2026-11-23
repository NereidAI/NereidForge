#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# seaweedfs.sh - Deploy SeaweedFS (Helm chart).

set -euo pipefail

PRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB_DIR="$PRE_ROOT/scripts/lib"
source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

CHARTS_DIR="$PRE_ROOT/charts"
S3_CREDENTIALS_SECRET="seaweedfs-s3-credentials"

log_info "========== Deploying SeaweedFS (Helm) =========="

: "${SEAWEEDFS_MASTER_STORAGE:=10Gi}"
: "${SEAWEEDFS_VOLUME_STORAGE:=200Gi}"
check_cmd python3 || exit 1
ensure_ns "nereidforge"

# The shared Secret is the credential source of truth. Reuse it on every rerun,
# even if site input has changed, so a normal provisioning/upgrade pass cannot
# rotate the identity out from under existing consumers. Migrate the identity
# from either legacy config form before the chart stops rendering that config.
S3_CREDENTIALS_CREATE_ONLY=true
if $KUBECTL get secret "$S3_CREDENTIALS_SECRET" -n nereidforge >/dev/null 2>&1; then
    S3_CREDENTIALS_CREATE_ONLY=false
    load_seaweedfs_s3_credentials secret "$S3_CREDENTIALS_SECRET" secret || exit 1
    log_info "Reusing the existing SeaweedFS credentials Secret (credentials not logged)"
elif $KUBECTL get secret seaweedfs-s3-config -n nereidforge >/dev/null 2>&1; then
    load_seaweedfs_s3_credentials secret seaweedfs-s3-config config || exit 1
    log_info "Preserving the existing SeaweedFS identity in the shared credentials Secret"
elif $KUBECTL get configmap seaweedfs-s3-config -n nereidforge >/dev/null 2>&1; then
    load_seaweedfs_s3_credentials configmap seaweedfs-s3-config config || exit 1
    log_info "Preserving the existing SeaweedFS identity while moving it into a Kubernetes Secret"
else
    # First-time initialization only. The access key has no interactive fallback
    # because deployment normally runs unattended; the secret key may be prompted.
    validate_required SEAWEEDFS_S3_ACCESS_KEY || exit 1
    read_password() {
        local prompt="$1"
        local var_name="$2"
        local value
        if [[ -n "${!var_name:-}" ]]; then
            value="${!var_name}"
            log_info "$prompt: ******** (from environment variable)"
        else
            read -rsp "$prompt: " value
            echo
        fi
        export "$var_name"="$value"
    }

    read_password "S3 Secret Key" SEAWEEDFS_S3_SECRET_KEY
    validate_required SEAWEEDFS_S3_ACCESS_KEY SEAWEEDFS_S3_SECRET_KEY || exit 1
fi

# JSON preserves every environment-safe credential byte. Stream the Secret
# directly to kubectl: no Helm values, command-line literal, handoff, or disk file
# contains the identity. resource-policy=keep also protects legacy Helm-owned
# credentials when this chart switches to an external-Secret reference.
if [[ "$S3_CREDENTIALS_CREATE_ONLY" == true ]]; then
    # When the canonical Secret was absent at the initial check, this installer
    # may only create it, including during legacy migration. Capture diagnostics
    # for AlreadyExists classification without printing a rejected manifest or
    # any candidate credentials. A concurrent winner is never overwritten with
    # this process's candidate.
    if create_error="$(
        {
            python3 "$LIB_DIR/s3_credentials.py" manifest \
                --namespace nereidforge --name "$S3_CREDENTIALS_SECRET" \
                | $KUBECTL create -f - >/dev/null
        } 2>&1
    )"; then
        log_info "Created the SeaweedFS credentials Secret (credentials not logged)"
    elif [[ "$create_error" == *"AlreadyExists"* ]]; then
        log_info "Reusing the concurrently created SeaweedFS credentials Secret"
    else
        log_err "Could not create the SeaweedFS credentials Secret"
        exit 1
    fi
else
    # An existing canonical identity retains its stored credentials while its
    # generated config and protection metadata are reconciled.
    if ! python3 "$LIB_DIR/s3_credentials.py" manifest \
        --namespace nereidforge --name "$S3_CREDENTIALS_SECRET" \
        | $KUBECTL apply -f - >/dev/null; then
        log_err "Could not create or update the SeaweedFS credentials Secret"
        exit 1
    fi
fi
# Reload the canonical object after creation, reconciliation, or an AlreadyExists
# race so bucket provisioning always uses the identity that is actually stored.
load_seaweedfs_s3_credentials secret "$S3_CREDENTIALS_SECRET" secret || exit 1

cd "$CHARTS_DIR/seaweedfs"
ls charts/*.tgz >/dev/null 2>&1 || helm dependency build

helm upgrade --install seaweedfs . -n nereidforge --create-namespace \
    --set image="$IMG_SEAWEEDFS" \
    --set storageClass="$STORAGE_CLASS" \
    --set master.persistence.size="$SEAWEEDFS_MASTER_STORAGE" \
    --set volume.persistence.size="$SEAWEEDFS_VOLUME_STORAGE" \
    --set s3.enabled=true \
    --set s3.existingSecret="$S3_CREDENTIALS_SECRET" \
    --set s3.configKey="s3-config.json" \
    --wait --timeout 10m

wait_deployment_ready "seaweedfs-master" "nereidforge"
wait_statefulset_ready "seaweedfs-volume" "nereidforge"
wait_deployment_ready "seaweedfs-filer" "nereidforge"
wait_deployment_ready "seaweedfs-s3" "nereidforge"

# Verify each endpoint.
log_info "Verifying SeaweedFS endpoints..."
sleep 10
kubectl run -n nereidforge sw-test-$$ --rm -i --restart=Never \
    --image=$IMG_BUSYBOX -- \
    wget -q -O - http://seaweedfs-master.nereidforge:9333/ | grep -q "SeaweedFS" && log_ok "Master UI is healthy"

kubectl run -n nereidforge sw-test2-$$ --rm -i --restart=Never \
    --image=$IMG_BUSYBOX -- \
    wget -q -O - http://seaweedfs-filer.nereidforge:8888/ | grep -q "SeaweedFS Filer" && log_ok "Filer UI is healthy"

kubectl run -n nereidforge sw-test3-$$ --rm -i --restart=Never \
    --image=$IMG_BUSYBOX -- \
    wget -q -O - http://seaweedfs-s3.nereidforge:8333/ | grep -q "ListAllMyBucketsResult" && log_ok "S3 API is healthy"

# Create every application bucket before any consumer starts. The helper exits
# non-zero on endpoint, authentication, authorization, creation, or verification
# failures; a successful HeadBucket alone is not considered sufficient access.
log_info "Provisioning and verifying S3 buckets (mlpipeline, nereidforge, labelstudio, cvat)..."
PY=python3
if [[ -x "$PRE_ROOT/.venv/bin/python" ]] && "$PRE_ROOT/.venv/bin/python" -c 'import boto3' >/dev/null 2>&1; then
    PY="$PRE_ROOT/.venv/bin/python"
fi
if ! "$PY" -c 'import boto3' >/dev/null 2>&1; then
    log_err "Python boto3 is required to provision and verify the S3 buckets"
    exit 1
fi
if ! S3_SVC_IP="$($KUBECTL get svc -n nereidforge seaweedfs-s3 -o jsonpath='{.spec.clusterIP}')"; then
    log_err "Could not resolve the SeaweedFS S3 Service; buckets were not provisioned"
    exit 1
fi
if [[ -z "$S3_SVC_IP" ]]; then
    log_err "Could not resolve the SeaweedFS S3 Service; buckets were not provisioned"
    exit 1
fi
S3_URL="http://$S3_SVC_IP:8333" \
S3_ACCESS_KEY="$SEAWEEDFS_S3_ACCESS_KEY" \
S3_SECRET_KEY="$SEAWEEDFS_S3_SECRET_KEY" \
    "$PY" "$LIB_DIR/provision_s3_buckets.py"
log_ok "All required S3 buckets are provisioned and authenticated access is verified"

log_ok "SeaweedFS deployment completed"
log_info "Connection information:"
echo "  Master: seaweedfs-master.nereidforge.svc.cluster.local:9333"
echo "  Volume: seaweedfs-volume.nereidforge.svc.cluster.local:8080"
echo "  Filer UI: http://seaweedfs-filer.nereidforge.svc.cluster.local:8888"
echo "  Filer gRPC: seaweedfs-filer.nereidforge.svc.cluster.local:18888"
echo "  S3 API: http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"
echo "  S3 credentials: Kubernetes Secret $S3_CREDENTIALS_SECRET (not logged)"
