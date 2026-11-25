#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# bootstrap-tokens.sh - Bootstrap LS/CVAT admins and fixed DRF tokens.
# Independent random UI passwords and both hand-offs are mode-0600 local files.
# A saved hand-off preserves passwords on reruns; explicit values rotate them.

set -euo pipefail
umask 077

PRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB_DIR="$PRE_ROOT/scripts/lib"
source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

TOKEN_FILE="$NF_HANDOFF_DIR/service-tokens.env"
CRED_FILE="$NF_HANDOFF_DIR/service-admin-credentials.env"
ensure_private_dir "$NF_HANDOFF_DIR"

# Capture explicit rotations before loading the saved deployment-managed values.
LS_PASSWORD_OVERRIDE="${LS_ADMIN_PASSWORD:-}"
CVAT_PASSWORD_OVERRIDE="${CVAT_ADMIN_PASSWORD:-}"
LS_ADMIN_PASSWORD=""
CVAT_ADMIN_PASSWORD=""
if [[ -f "$CRED_FILE" ]]; then
    chmod 600 "$CRED_FILE"
    # shellcheck source=/dev/null
    source "$CRED_FILE"
fi

# A missing hand-off triggers one recovery rotation, including on legacy installs.
# This makes the administrator credential retrievable without a shared default.
LS_ROTATE=false
if [[ -n "$LS_PASSWORD_OVERRIDE" ]]; then
    LS_ADMIN_PASSWORD="$LS_PASSWORD_OVERRIDE"
    LS_ROTATE=true
elif [[ -z "$LS_ADMIN_PASSWORD" ]]; then
    LS_ADMIN_PASSWORD="$(openssl rand -hex 24)"
    LS_ROTATE=true
fi

CVAT_ROTATE=false
if [[ -n "$CVAT_PASSWORD_OVERRIDE" ]]; then
    CVAT_ADMIN_PASSWORD="$CVAT_PASSWORD_OVERRIDE"
    CVAT_ROTATE=true
elif [[ -z "$CVAT_ADMIN_PASSWORD" ]]; then
    CVAT_ADMIN_PASSWORD="$(openssl rand -hex 24)"
    CVAT_ROTATE=true
fi

log_info "========== LS/CVAT admin and token bootstrap =========="

wait_deployment_ready "labelstudio" "nereidforge"
wait_deployment_ready "cvat-backend-server" "nereidforge"

LS_POD=$(kubectl get pods -n nereidforge -l app=labelstudio -o jsonpath='{.items[0].metadata.name}')
CVAT_POD=$(kubectl get pods -n nereidforge -l app=cvat-app,component=server -o jsonpath='{.items[0].metadata.name}')

# Passwords travel over kubectl stdin rather than command arguments or logs.
LS_PASSWORD_B64="$(printf '%s' "$LS_ADMIN_PASSWORD" | base64 -w0)"
LS_PY=$(cat <<'PYEOF'
import base64
import django
import os
import sys
sys.path.insert(0, '/label-studio/label_studio')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings.label_studio')
django.setup()
from users.models import User
from organizations.models import Organization
from rest_framework.authtoken.models import Token
rotate = sys.stdin.readline().strip() == 'true'
password = base64.b64decode(sys.stdin.readline().strip()).decode('utf-8')
u, created = User.objects.get_or_create(username='admin', defaults={
    'is_superuser': True, 'is_staff': True, 'email': 'admin@nereidforge.local'})
if created or rotate:
    u.set_password(password)
u.save()
if not Organization.objects.filter(users=u).exists():
    Organization.create_organization(created_by=u, title='NereidForge')
for o in Organization.objects.all():
    o.jwt.legacy_api_tokens_enabled = True
    o.jwt.save()
print('LS_TOKEN:' + Token.objects.get_or_create(user=u)[0].key)
PYEOF
)
if ! LS_OUT=$(printf '%s\n%s\n' "$LS_ROTATE" "$LS_PASSWORD_B64" | \
    kubectl exec -i -n nereidforge "$LS_POD" -c labelstudio -- \
    sh -c 'cd /label-studio/label_studio && exec /label-studio/.venv/bin/python -c "$1"' \
    sh "$LS_PY" 2>/dev/null); then
    log_err "LS admin/token bootstrap failed; no credential value was logged"
    exit 1
fi
LS_TOKEN=$(printf '%s\n' "$LS_OUT" | sed -n 's/^LS_TOKEN://p' | tail -n 1)
if [[ -z "$LS_TOKEN" ]]; then
    log_err "LS token was not returned; no credential value was logged"
    exit 1
fi
log_ok "LS admin and token bootstrap completed"

CVAT_PASSWORD_B64="$(printf '%s' "$CVAT_ADMIN_PASSWORD" | base64 -w0)"
CVAT_PY=$(cat <<'PYEOF'
import base64
import django
import os
import sys
sys.path.insert(0, '/home/django/backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cvat.settings.production')
django.setup()
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
rotate = sys.stdin.readline().strip() == 'true'
password = base64.b64decode(sys.stdin.readline().strip()).decode('utf-8')
u, created = User.objects.get_or_create(username='admin', defaults={
    'is_superuser': True, 'is_staff': True, 'email': 'admin@nereidforge.local'})
if created or rotate:
    u.set_password(password)
u.save()
tok, _ = Token.objects.get_or_create(user=u)
print('CVAT_TOKEN:' + tok.key)
PYEOF
)
if ! CVAT_OUT=$(printf '%s\n%s\n' "$CVAT_ROTATE" "$CVAT_PASSWORD_B64" | \
    kubectl exec -i -n nereidforge "$CVAT_POD" -c cvat-backend -- \
    python3 /home/django/manage.py shell -c "$CVAT_PY" 2>/dev/null); then
    log_err "CVAT admin/token bootstrap failed; no credential value was logged"
    exit 1
fi
CVAT_TOKEN=$(printf '%s\n' "$CVAT_OUT" | sed -n 's/^CVAT_TOKEN://p' | tail -n 1)
if [[ -z "$CVAT_TOKEN" ]]; then
    log_err "CVAT token was not returned; no credential value was logged"
    exit 1
fi
log_ok "CVAT admin and token bootstrap completed"

# Write hand-offs only after both applications accept the bootstrap operations.
cat > "$TOKEN_FILE" <<EOF
# Auto-generated operator hand-off; save securely and never commit.
NEREIDFORGE_LABELSTUDIO_TOKEN=$LS_TOKEN
NEREIDFORGE_CVAT_TOKEN=$CVAT_TOKEN
EOF
chmod 600 "$TOKEN_FILE"

cat > "$CRED_FILE" <<EOF
# Auto-generated UI administrator hand-off; save securely and never commit.
LS_ADMIN_USERNAME=admin
LS_ADMIN_EMAIL=admin@nereidforge.local
LS_ADMIN_PASSWORD=$(printf '%q' "$LS_ADMIN_PASSWORD")
CVAT_ADMIN_USERNAME=admin
CVAT_ADMIN_PASSWORD=$(printf '%q' "$CVAT_ADMIN_PASSWORD")
EOF
chmod 600 "$CRED_FILE"

log_ok "Tokens written to $TOKEN_FILE (mode 600)"
log_ok "UI admin credentials written to $CRED_FILE (mode 600)"
