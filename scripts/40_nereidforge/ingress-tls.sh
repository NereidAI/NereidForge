#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# ingress-tls.sh - Configure Ingress TLS for a tenant

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB_DIR="$REPO_ROOT/scripts/lib"

source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

TENANT="${1:-nereidforge}"

log_info "========== Configure Ingress TLS ($TENANT) =========="

# Wait for all Certificates in the namespace to become Ready
log_info "Waiting for cert-manager to issue certificates..."
CERTS=$(kubectl get certificate -n "$TENANT" -o name 2>/dev/null || true)
if [[ -z "$CERTS" ]]; then
    log_info "No certificates to wait for (may be auto-created by Ingress)"
else
    for c in $CERTS; do
        log_info "Waiting for $c Ready..."
        retry 30 10 kubectl wait --for=condition=Ready "$c" -n "$TENANT" --timeout=300s || true
    done
fi

kubectl get certificate -n "$TENANT" 2>/dev/null || true

log_ok "Ingress TLS configuration complete"
