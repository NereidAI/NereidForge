#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Finalize the non-secret NF Helm handoff after pre installation succeeds.
# Only the two pre-owned LS/CVAT token keys are written to the runtime Secret.
# Server-side apply preserves independently managed NF JWT/SSO keys. No secret
# is written to values.generated.yaml, Helm arguments, or deployment logs.

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PRE_ROOT/scripts/lib/common.sh"
source "$PRE_ROOT/scripts/lib/config.sh"

validate_required BASE_DOMAIN DEX_STATIC_EMAIL PG_CLUSTER_NAME PG_DB_NAME PG_DB_USER || exit 1
check_cmd python3 base64 || exit 1

TOKEN_FILE="$PRE_HANDOFF_DIR/service-tokens.env"
if [[ ! -f "$TOKEN_FILE" ]]; then
    log_err "Missing retained service token handoff: $TOKEN_FILE"
    exit 1
fi
chmod 600 "$TOKEN_FILE"
# shellcheck source=/dev/null
source "$TOKEN_FILE"
validate_required NEREIDFORGE_LABELSTUDIO_TOKEN NEREIDFORGE_CVAT_TOKEN || exit 1
export NEREIDFORGE_LABELSTUDIO_TOKEN NEREIDFORGE_CVAT_TOKEN

RUNTIME_SECRET="nereidforge-service-credentials"
# The manifest exists only on the kubectl apply stdin stream. pre owns exactly
# these two fields; NF-owned fields added by another field manager are retained.
if ! python3 - <<'PY' | "$KUBECTL" apply --server-side --field-manager=pre-nereidforge --force-conflicts -f - >/dev/null
import json
import os

print(json.dumps({
    "apiVersion": "v1",
    "kind": "Secret",
    "metadata": {
        "name": "nereidforge-service-credentials",
        "namespace": "nereidforge",
        "labels": {"app.kubernetes.io/part-of": "pre-nereidforge"},
    },
    "type": "Opaque",
    "stringData": {
        "NEREIDFORGE_LABELSTUDIO_TOKEN": os.environ["NEREIDFORGE_LABELSTUDIO_TOKEN"],
        "NEREIDFORGE_CVAT_TOKEN": os.environ["NEREIDFORGE_CVAT_TOKEN"],
    },
}))
PY
then
    log_err "Could not retain $RUNTIME_SECRET; NF handoff was not generated"
    exit 1
fi
log_ok "Retained pre-owned LS/CVAT token keys in nereidforge/$RUNTIME_SECRET"

# Copy only the public CA certificate. The source private key is never read or
# replicated. This destination is an existing-Secret chart input, not a value.
if ! "$KUBECTL" get secret nereid-internal-ca-tls -n cert-manager \
    -o jsonpath='{.data.ca\.crt}' >/dev/null 2>&1; then
    log_err "Missing cert-manager/nereid-internal-ca-tls ca.crt; NF handoff was not generated"
    exit 1
fi
if ! "$KUBECTL" get secret nereid-internal-ca-tls -n cert-manager \
    -o jsonpath='{.data.ca\.crt}' | base64 -d | \
    "$KUBECTL" create secret generic nereid-internal-ca -n nereidforge \
        --from-file=ca.crt=/dev/stdin --dry-run=client -o yaml | \
    "$KUBECTL" apply --server-side --field-manager=pre-nereidforge --force-conflicts \
        -f - >/dev/null; then
    log_err "Could not replicate the public CA certificate; NF handoff was not generated"
    exit 1
fi
log_ok "Replicated public CA certificate to nereidforge/nereid-internal-ca"

VALUES_FILE="$PRE_HANDOFF_DIR/values.generated.yaml"
python3 "$PRE_ROOT/scripts/lib/generate_nf_handoff.py" generate \
    --output "$VALUES_FILE" \
    --pre-version "$PRE_RELEASE_TAG" \
    --source-commit "$SOURCE_COMMIT" \
    --base-domain "$BASE_DOMAIN" \
    --storage-class "$STORAGE_CLASS" \
    --pg-cluster "$PG_CLUSTER_NAME" \
    --pg-database "$PG_DB_NAME" \
    --pg-user "$PG_DB_USER" \
    --dex-email "$DEX_STATIC_EMAIL" \
    --frontend-host "${FRONTEND_HOST:-}" \
    --sso-mode "${SSO_MODE:-ns}" \
    --sso-ns-base-url "${SSO_NS_BASE_URL:-}" \
    --sso-oidc-issuer "${SSO_OIDC_ISSUER:-}" \
    --sso-oidc-client-id "${SSO_OIDC_CLIENT_ID:-nereidforge}"

log_ok "NF Helm handoff ready: $VALUES_FILE (non-secret, mode 644)"
log_info "Create the NF-owned runtime/admin Secret keys and an operator overlay, then run scripts/40_check-nf-handoff.sh"
