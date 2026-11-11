#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# images.sh defines the complete public offline-image inventory and tar helpers.
#
# Self-built references come from nf_ref() so category exports, the root
# manifest, charts, and runtime code cannot drift onto different repositories.
# Third-party images retain their product-pinned upstream coordinates. This
# library only pulls, exports, and loads public references; publication into a
# customer registry belongs exclusively to NereidOverview harbor-sync.sh.

set -euo pipefail

source "$LIB_DIR/common.sh"
source "$LIB_DIR/config.sh"

# IMAGE_CATEGORIES fixes archive and manifest ordering across Bash versions.
IMAGE_CATEGORIES=(base kubeflow inference annotation nf training k3s-recovered tools)

# CATEGORY_IMAGES is the single inventory used by offline pull/export/load and
# by images-manifest.txt generation. Full python:3.14 remains only for
# rebuilding previously delivered artifacts; current compatible sources use
# python:3.14-slim, with the user-required TensorFlow/Paddle Python 3.11 pins.
declare -A CATEGORY_IMAGES=(
    ["base"]="
$IMG_POSTGRESQL
$IMG_POSTGRESQL_INSTANCE_MGR
$IMG_VALKEY
$IMG_SEAWEEDFS
$IMG_OPA
docker.io/library/python:3.11-slim
docker.io/library/python:3.12-slim
docker.io/library/python:3.14-slim
docker.io/library/python:3.14
$IMG_LLAMAFACTORY
$IMG_ROCM_PYTORCH
$IMG_JUPYTER_BASE
docker.io/library/ubuntu:24.04
docker.io/library/node:24-alpine
docker.io/library/nginx:alpine
nvcr.io/nvidia/tensorrt:24.05-py3
nvcr.io/nvidia/tensorrt:24.06-py3
"

    # Platform references are inventory records only; this change does not
    # modify or deploy Kubeflow. Its product-pinned Alpine 3.23 remains an
    # external-upstream exception to the NF-owned Alpine 3.24.4 utility base.
    ["kubeflow"]="
registry.istio.io/release/pilot:1.30.3
registry.istio.io/release/proxyv2:1.30.3
quay.io/jetstack/cert-manager-controller:v1.21.1
quay.io/jetstack/cert-manager-webhook:v1.21.1
quay.io/jetstack/cert-manager-cainjector:v1.21.1
quay.io/oauth2-proxy/oauth2-proxy:v7.15.3
ghcr.io/kubeflow/kfp-api-server:2.17.0
ghcr.io/kubeflow/kfp-cache-deployer:2.17.0
ghcr.io/kubeflow/kfp-cache-server:2.17.0
ghcr.io/kubeflow/kfp-frontend:2.17.0
ghcr.io/kubeflow/kfp-metadata-envoy:2.17.0
ghcr.io/kubeflow/kfp-metadata-writer:2.17.0
ghcr.io/kubeflow/kfp-persistence-agent:2.17.0
ghcr.io/kubeflow/kfp-scheduled-workflow-controller:2.17.0
ghcr.io/kubeflow/kfp-viewer-crd-controller:2.17.0
ghcr.io/kubeflow/kfp-visualization-server:2.17.0
quay.io/argoproj/workflow-controller:v4.0.5
ghcr.io/metacontroller/metacontroller:v4.16.2
registry.k8s.io/jobset/jobset:v0.12.0
ghcr.io/chrislusf/seaweedfs:4.34
docker.io/library/busybox:1.28
kserve/kserve-controller:v0.20.0
kserve/kserve-localmodel-controller:v0.20.0
kserve/llmisvc-controller:v0.20.0
ghcr.io/kubeflow/trainer/trainer-controller-manager:v2.3.0
ghcr.io/kubeflow/notebooks/workspaces-backend:v2.0.0-beta.0
ghcr.io/kubeflow/notebooks/workspaces-controller:v2.0.0-beta.0
ghcr.io/kubeflow/notebooks/workspaces-frontend:v2.0.0-beta.0
ghcr.io/kubeflow/hub/server:v0.3.14
ghcr.io/kubeflow/hub/ui:v0.3.14
gcr.io/knative-releases/knative.dev/eventing/cmd/controller@sha256:72d8700f084d275f889b3663ee5096c455e0aae919586aaeee384644173de4c1
gcr.io/knative-releases/knative.dev/eventing/cmd/jobsink@sha256:b95c48ff0bda886d502577ad8b638fe0e37b092d5e5bf0434935fb8c3322844d
gcr.io/knative-releases/knative.dev/eventing/cmd/mtping@sha256:65d1d4877d45879e5c08440acf81781bf1b232e8eb5c250549f8bc7b3575a041
gcr.io/knative-releases/knative.dev/eventing/cmd/requestreply@sha256:af81e3dc1ae915ba2405cb5983f3825edb798038391649b5d9acda917b8ceb6e
gcr.io/knative-releases/knative.dev/eventing/cmd/webhook@sha256:50da07e9a1241a9cb4ffdf6d73ac27e690a32f79dd30282e862b254167812a5d
gcr.io/knative-releases/knative.dev/pkg/apiextensions/storageversion/cmd/migrate@sha256:3251489b1e27cf54ed0c83e8e2d9a2f4eae28bd400f08174a1c41b10388eff3f
gcr.io/knative-releases/knative.dev/pkg/apiextensions/storageversion/cmd/migrate@sha256:f7ed5718308c5b272c8ae37f8d097aab483f02870dd98f94464764b982e8d8f7
docker.io/alpine:3.23
docker.io/alpine/k8s:1.36.2
quay.io/argoproj/argoexec:v4.0.5
ghcr.io/kubeflow/kfp-driver:2.17.0
ghcr.io/kubeflow/kfp-launcher:2.17.0
ghcr.io/containerd/busybox:latest
kserve/storage-initializer:v0.20.0
ghcr.io/kserve/models-web-app:1.0.1
quay.io/jetstack/cert-manager-acmesolver:v1.21.1
ghcr.io/kubeflow/katib/suggestion-darts:v0.19.0
ghcr.io/kubeflow/katib/suggestion-enas:v0.19.0
ghcr.io/kubeflow/katib/suggestion-goptuna:v0.19.0
ghcr.io/kubeflow/katib/suggestion-hyperband:v0.19.0
ghcr.io/kubeflow/katib/suggestion-hyperopt:v0.19.0
ghcr.io/kubeflow/katib/suggestion-optuna:v0.19.0
ghcr.io/kubeflow/katib/suggestion-pbt:v0.19.0
ghcr.io/kubeflow/katib/suggestion-skopt:v0.19.0
ghcr.io/kubeflow/katib/earlystopping-medianstop:v0.19.0
ghcr.io/kubeflow/katib/tfevent-metrics-collector:v0.19.0
ghcr.io/kubeflow/katib/file-metrics-collector:v0.19.0
ghcr.io/kubeflow/katib/enas-cnn-cifar10-cpu:v0.19.0
ghcr.io/kubeflow/kubeflow/notebook-servers/codeserver-python:v1.11.0
ghcr.io/kubeflow/kubeflow/notebook-servers/jupyter-pytorch-full:v1.11.0
ghcr.io/kubeflow/kubeflow/notebook-servers/rstudio-tidyverse:v1.11.0
ghcr.io/kubeflow/kubeflow/notebook-servers/jupyter-tensorflow-full:v1.11.0
ghcr.io/kubeflow/kubeflow/notebook-servers/jupyter-scipy:v1.11.0
ghcr.io/kubeflow/katib/pytorch-mnist-cpu:v0.19.0
"

    ["inference"]="
$(nf_ref inference/model-server-openvino-xpu "$RELEASE_TAG")
$IMG_OPENVINO
openvino/model_server:2025.1
$IMG_TRITONSERVER
$IMG_PYTORCH
$IMG_TGI
"

    ["annotation"]="
$IMG_CVAT_SERVER
$IMG_CVAT_UI
$IMG_LABEL_STUDIO
$IMG_LABEL_STUDIO_ML_BACKEND
"

    ["nf"]="
$(nf_ref nereidforge-backend "$RELEASE_TAG")
$(nf_ref nereidforge-frontend "$RELEASE_TAG")
$(nf_ref ml-backend/ml-backend "$RELEASE_TAG")
$(nf_ref ml-backend/ml-backend-custom "$RELEASE_TAG")
$(nf_ref prediction/jupyter-base py3.12)
$(nf_ref runtime "$RELEASE_TAG")
"

    ["training"]="
$(nf_ref trainer/torch-cpu "$RELEASE_TAG")
$(nf_ref trainer/torch-cu "$RELEASE_TAG")
$(nf_ref trainer/torch-rocm "$RELEASE_TAG")
$(nf_ref trainer/torch-xpu "$RELEASE_TAG")
$(nf_ref trainer/paddle-cpu "$RELEASE_TAG")
$(nf_ref trainer/paddle-cu "$RELEASE_TAG")
$(nf_ref trainer/llm-cu "$RELEASE_TAG")
$(nf_ref trainer/llm-rocm "$RELEASE_TAG")
$(nf_ref trainer/llm-xpu "$RELEASE_TAG")
$(nf_ref trainer/xpu-base "$RELEASE_TAG")
$(nf_ref trainer/convert "$RELEASE_TAG")
$(nf_ref trainer/convert-tf "$RELEASE_TAG")
$(nf_ref trainer/convert-pp "$RELEASE_TAG")
$(nf_ref train/optimizer "$RELEASE_TAG")
$(nf_ref train/evaluator "$RELEASE_TAG")
$(nf_ref train/evaluator-xpu "$RELEASE_TAG")
"

    ["k3s-recovered"]="
rancher/mirrored-pause:3.10.2
rancher/mirrored-coredns-coredns:1.14.6
rancher/mirrored-library-traefik:3.7.8
rancher/klipper-lb:v0.4.17
rancher/klipper-helm:v0.13.3-build20260727
rancher/mirrored-metrics-server:v0.9.0
rancher/local-path-provisioner:v0.0.36
"

    ["tools"]="
$IMG_BUSYBOX
$IMG_ALPINE
$IMG_CURL
$IMG_KUBECTL
"
)

# get_category_images prints one category's references, including no other
# category's entries; callers filter blank lines and comments.
get_category_images() {
    local cat="$1"
    # Values already begin and end with one newline. Preserve them verbatim so
    # the final category cannot add an extra blank line at manifest EOF.
    printf '%s' "${CATEGORY_IMAGES[$cat]:-}"
}

# list_categories prints stable category names and their reference counts.
list_categories() {
    local cat count
    for cat in "${IMAGE_CATEGORIES[@]}"; do
        count=$(get_category_images "$cat" | grep -cvE '^[[:space:]]*(#|$)' || true)
        echo "  $cat ($count images)"
    done
}

# pull_category pulls every public reference in one category into Docker's
# local cache and returns the first failure through retry().
pull_category() {
    local cat="$1"
    local images line
    images=$(get_category_images "$cat")
    [[ -n "${images//[[:space:]]/}" ]] || { log_err "Unknown category: $cat"; return 1; }

    log_info "Pulling category: $cat"
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        log_info "  pull: $line"
        retry 3 5 docker pull "$line"
    done <<< "$images"
}

# pull_all_categories pulls every category in deterministic order.
pull_all_categories() {
    local cat
    for cat in "${IMAGE_CATEGORIES[@]}"; do
        pull_category "$cat"
    done
}

# export_category pulls and saves one category as a layer-deduplicated tar.gz;
# references inside the archive remain byte-identical to runtime manifests.
export_category() {
    local cat="$1"
    local output_dir="${2:-$OFFLINE_DIR/images}"
    local images line output
    local img_list=()
    images=$(get_category_images "$cat")
    [[ -n "${images//[[:space:]]/}" ]] || { log_err "Unknown category: $cat"; return 1; }

    mkdir -p "$output_dir"
    output="$output_dir/images-${cat}.tar.gz"
    log_info "Exporting category: $cat -> $output"

    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        log_info "  pull: $line"
        retry 3 5 docker pull "$line" >/dev/null
        img_list+=("$line")
    done <<< "$images"

    docker save "${img_list[@]}" | gzip > "$output"
    log_ok "Export complete: $output ($(du -h "$output" | cut -f1))"
}

# export_all_categories exports the complete inventory in stable order.
export_all_categories() {
    local output_dir="${1:-$OFFLINE_DIR/images}"
    local cat
    mkdir -p "$output_dir"
    for cat in "${IMAGE_CATEGORIES[@]}"; do
        export_category "$cat" "$output_dir"
    done
    log_ok "All images exported: $output_dir"
}

# load_category loads one previously exported category archive.
load_category() {
    local cat="$1"
    local input_dir="${2:-$OFFLINE_DIR/images}"
    local input="$input_dir/images-${cat}.tar.gz"

    [[ -f "$input" ]] || { log_err "File not found: $input"; return 1; }
    log_info "Loading category: $cat"
    docker load < "$input"
    log_ok "Load complete: $cat"
}

# load_all_categories loads every category archive from an offline bundle.
load_all_categories() {
    local input_dir="${1:-$OFFLINE_DIR/images}"
    local cat
    for cat in "${IMAGE_CATEGORIES[@]}"; do
        load_category "$cat" "$input_dir"
    done
    log_ok "All images loaded"
}

# generate_manifest writes a deterministic host-free manifest from the same
# category inventory used for export; regeneration is byte-for-byte testable.
generate_manifest() {
    local output="${1:-$OFFLINE_DIR/images-manifest.txt}"
    local cat
    cat > "$output" <<EOF
# NereidForge image manifest — public source references only (root treaty section 18).
# Generated by: bash infra/scripts/05_export-images.sh --manifest <output>
# Blank lines and lines starting with '#' are ignored by delivery tooling.
# Canonical runtime references remain unchanged; NereidOverview harbor-sync.sh
# optionally mirrors these sources into a customer-selected internal registry.
# Categories: ${IMAGE_CATEGORIES[*]}
EOF
    for cat in "${IMAGE_CATEGORIES[@]}"; do
        {
            echo ""
            echo "## $cat"
            get_category_images "$cat"
        } >> "$output"
    done
    log_ok "Manifest generated: $output"
}
