#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Local source/package/chart verification only: no cluster or registry access,
# dependency downloads, image builds, or cache preparation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$PRE_ROOT/scripts/lib/common.sh"
check_cmd python3

log_info "Running pre-NereidForge stdlib regression tests"
python3 -m unittest discover -s "$PRE_ROOT/tests" -p 'test_*.py' -v

log_info "Checking shell syntax"
while IFS= read -r -d '' script; do
    bash -n "$script"
done < <(find "$PRE_ROOT/scripts" "$PRE_ROOT/platform/kubeflow-patches" \
    -type f -name '*.sh' -print0)

if ! command -v helm >/dev/null 2>&1; then
    log_warn "helm is unavailable; skipped lint/template for the five charts"
    exit 0
fi

CHARTS=(cvat labelstudio postgresql seaweedfs valkey)
log_info "Linting the five pre charts"
for chart in "${CHARTS[@]}"; do
    helm lint "$PRE_ROOT/charts/$chart"
done

RENDER_DIR="$(mktemp -d)"
trap 'rm -rf "$RENDER_DIR"' EXIT

for chart in labelstudio seaweedfs valkey; do
    log_info "Rendering $chart"
    helm template "$chart" "$PRE_ROOT/charts/$chart" > "$RENDER_DIR/$chart.yaml"
done

log_info "Rendering CVAT with its six vendored dependency archives"
cp -a "$PRE_ROOT/charts/cvat" "$RENDER_DIR/cvat"
mkdir -p "$RENDER_DIR/cvat/charts"
cp "$RENDER_DIR/cvat"/*.tgz "$RENDER_DIR/cvat/charts/"
helm template cvat "$RENDER_DIR/cvat" -f "$PRE_ROOT/values/cvat.yaml" \
    --set ingress.hostname=cvat.example.test > "$RENDER_DIR/cvat.yaml"

# The CNPG operator chart is not vendored in tracked source. Render the local
# wrapper with that external operator disabled and a metadata-only dependency
# placeholder; no repository is contacted and no dependency is installed.
log_info "Rendering PostgreSQL wrapper with the external CNPG operator disabled"
cp -a "$PRE_ROOT/charts/postgresql" "$RENDER_DIR/postgresql"
mkdir -p "$RENDER_DIR/postgresql/charts/cloudnative-pg"
cat > "$RENDER_DIR/postgresql/charts/cloudnative-pg/Chart.yaml" <<'EOF'
apiVersion: v2
name: cloudnative-pg
version: 0.29.0
EOF
helm template postgresql "$RENDER_DIR/postgresql" --set cnpg.enabled=false \
    > "$RENDER_DIR/postgresql.yaml"

log_ok "Local pre-NereidForge verification passed (no cluster or registry checks)"
