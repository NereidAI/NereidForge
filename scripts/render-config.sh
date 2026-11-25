#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# render-config.sh - Render unified config template with Jinja2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/configs/unified"
TEMPLATE="${CONFIG_DIR}/config.yaml.j2"
OUTPUT="${CONFIG_DIR}/config.rendered.yaml"

# Defaults
ENV="${ENV:-prod}"
DOMAIN="${DOMAIN:-}"
TENANT="${TENANT:-nereidforge}"
DB_PASSWORD="${DB_PASSWORD:-}"
JWT_SECRET="${JWT_SECRET:-}"
LABELSTUDIO_TOKEN="${LABELSTUDIO_TOKEN:-}"
CVAT_TOKEN="${CVAT_TOKEN:-}"
DB_NAME="${DB_NAME:-}"
SSO_NS_BASE_URL="${SSO_NS_BASE_URL:-}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-}"
S3_SECRET_KEY="${S3_SECRET_KEY:-}"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --env ENV           Environment: dev|prod (default: prod)
  --domain DOMAIN     Base domain (required for prod; default: localhost for dev)
  --tenant TENANT     Tenant name (default: nereidforge)
  --db-password PASS  Database password
  --db-name NAME      Database name (default: nereidforge for prod, nereidforge_dev for dev)
  --jwt-secret SECRET JWT secret
  --ls-token TOKEN    Label Studio token
  --cvat-token TOKEN  CVAT token
  --sso-ns-base-url URL  NereidSphere base URL (required for prod)
  --s3-access-key KEY  Object-store access key (optional for local dev only)
  --s3-secret-key KEY  Object-store secret key (optional for local dev only)
  -h, --help          Show this help

Environment variables can also be used: ENV, DOMAIN, TENANT, DB_PASSWORD, JWT_SECRET, LABELSTUDIO_TOKEN, CVAT_TOKEN, SSO_NS_BASE_URL, S3_ACCESS_KEY, S3_SECRET_KEY
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --env) ENV="$2"; shift 2 ;;
        --domain) DOMAIN="$2"; shift 2 ;;
        --tenant) TENANT="$2"; shift 2 ;;
        --db-password) DB_PASSWORD="$2"; shift 2 ;;
        --jwt-secret) JWT_SECRET="$2"; shift 2 ;;
        --ls-token) LABELSTUDIO_TOKEN="$2"; shift 2 ;;
        --cvat-token) CVAT_TOKEN="$2"; shift 2 ;;
        --db-name) DB_NAME="$2"; shift 2 ;;
        --sso-ns-base-url) SSO_NS_BASE_URL="$2"; shift 2 ;;
        --s3-access-key) S3_ACCESS_KEY="$2"; shift 2 ;;
        --s3-secret-key) S3_SECRET_KEY="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ "${ENV}" != "dev" && "${ENV}" != "prod" ]]; then
    echo "Error: --env must be dev or prod" >&2
    exit 1
fi
if [[ "${ENV}" == "dev" && -z "${DOMAIN}" ]]; then
    DOMAIN="localhost"
fi
if [[ "${ENV}" == "prod" ]]; then
    if [[ -z "${DOMAIN}" ]]; then echo "Error: --domain required for prod" >&2; exit 1; fi
    if [[ -z "${DB_PASSWORD}" ]]; then echo "Error: --db-password required for prod" >&2; exit 1; fi
    if [[ -z "${JWT_SECRET}" ]]; then echo "Error: --jwt-secret required for prod" >&2; exit 1; fi
    if [[ -z "${SSO_NS_BASE_URL}" ]]; then echo "Error: --sso-ns-base-url required for prod" >&2; exit 1; fi
    # Production S3 credentials stay in the pre-managed Kubernetes Secret and
    # reach pods through secretKeyRef, never this non-secret ConfigMap render.
fi

if ! command -v jinja2 &>/dev/null; then
    pip3 install --break-system-packages --quiet jinja2-cli
fi

export PATH="$HOME/.local/bin:$PATH"

echo "Rendering config for env=${ENV}, domain=${DOMAIN}, tenant=${TENANT}..."

jinja2 "${TEMPLATE}" \
    -D env="${ENV}" \
    -D domain="${DOMAIN}" \
    -D tenant="${TENANT}" \
    -D db_password="${DB_PASSWORD}" \
    -D jwt_secret="${JWT_SECRET}" \
    -D ls_token="${LABELSTUDIO_TOKEN}" \
    -D cvat_token="${CVAT_TOKEN}" \
    -D db_name="${DB_NAME}" \
    -D sso_ns_base_url="${SSO_NS_BASE_URL}" \
    -D s3_access_key="${S3_ACCESS_KEY}" \
    -D s3_secret_key="${S3_SECRET_KEY}" \
    > "${OUTPUT}"

echo "Rendered config written to ${OUTPUT}"
