#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# prepare-secrets.sh generates secrets.yaml and optional registry pull secrets.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/config.sh"

log_info "Prepare Secrets..."

# Fail before writing an incomplete credential file. Annotation-service tokens
# are intentionally excluded because their bootstrap stage creates them later.
validate_required PG_ADMIN_PASSWORD JWT_SECRET \
    SEAWEEDFS_S3_ACCESS_KEY SEAWEEDFS_S3_SECRET_KEY \
    DEX_STATIC_PASSWORD_HASH || exit 1

ensure_private_dir "$DEPLOY_VALUES_DIR"
# Generate secrets.yaml outside Git from the tracked template. Existing files
# retain their values; reruns only tighten the operator-only permissions.
if [[ ! -f "$SECRETS_FILE" ]]; then
    if [[ -f "$SECRETS_TEMPLATE" ]]; then
        log_info "Generate $SECRETS_FILE from the template"
        export PG_ADMIN_PASSWORD JWT_SECRET SEAWEEDFS_S3_ACCESS_KEY SEAWEEDFS_S3_SECRET_KEY
        export LABEL_STUDIO_TOKEN CVAT_TOKEN DEX_STATIC_PASSWORD_HASH
        (umask 077; envsubst < "$SECRETS_TEMPLATE" > "$SECRETS_FILE")
        chmod 600 "$SECRETS_FILE"
        log_warn "Review and replace sensitive values in $SECRETS_FILE"
    else
        log_err "Template file is missing: $SECRETS_TEMPLATE"
        exit 1
    fi
else
    chmod 600 "$SECRETS_FILE"
    log_info "$SECRETS_FILE already exists; skipping generation"
fi

# Optional registry pull secret. Default installs use public coordinates with
# anonymous pull, so no secret is created. Set all three variables in the
# operator environment to provision the secret in the existing namespaces:
#   REGISTRY_HOST / REGISTRY_USER / REGISTRY_PASS
NAMESPACES=("nereidforge" "kubeflow" "cert-manager" "istio-system" "knative-serving" "knative-eventing" "kserve" "auth" "oauth2-proxy")

if [[ -n "${REGISTRY_HOST:-}" && -n "${REGISTRY_USER:-}" && -n "${REGISTRY_PASS:-}" ]]; then
    for ns in "${NAMESPACES[@]}"; do
        if $KUBECTL get ns "$ns" &>/dev/null; then
            create_registry_secret "$ns" "harbor-registry-secret" \
                "$REGISTRY_HOST" "$REGISTRY_USER" "$REGISTRY_PASS"
        fi
    done
else
    log_info "REGISTRY_HOST/USER/PASS not set; skipping registry pull secrets (public anonymous pull)"
fi

log_ok "Secrets preparation complete"
