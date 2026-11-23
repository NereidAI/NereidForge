#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# kubeflow-check.sh - Verify Kubeflow (L1: component readiness).

set -euo pipefail

PRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PRE_ROOT/../../scripts/lib/common.sh"
source "$PRE_ROOT/../../scripts/lib/config.sh"

log_info "========== Verifying Kubeflow =========="

log_info "Checking Central Dashboard..."
wait_deployment_ready "dashboard" "kubeflow"

log_info "Checking KFP Pipeline..."
wait_deployment_ready "ml-pipeline" "kubeflow"
wait_deployment_ready "ml-pipeline-ui" "kubeflow"

log_info "Checking Jupyter Notebooks..."
wait_deployment_ready "jupyter-web-app-deployment" "kubeflow"

log_info "Checking Katib..."
wait_deployment_ready "katib-controller" "kubeflow"
wait_deployment_ready "katib-ui" "kubeflow"

log_info "Checking KServe..."
wait_deployment_ready "kserve-controller-manager" "kserve"

log_ok "Kubeflow verification passed"
