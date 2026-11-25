# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# NereidForge local development, build, and verification entry points.
#
# Every container target leaves an image in the local Docker daemon. This
# Makefile never publishes artifacts or changes a cluster. GitHub Actions owns
# GHCR production builds; customer-selected private delivery is owned by the
# separate NereidOverview harbor-sync tooling.

# Optional local build acceleration (treaty §19.3): copy local.mk.example to
# local.mk (git-ignored) and set regional proxy/mirror endpoints there. A
# missing file is a no-op, so a direct-internet developer gets the pristine
# upstream build that CI runs.
-include local.mk

.PHONY: help dev-build dev-all dev-clean docker-build \
        build-api build-frontend build-ml-backend \
        build-trainer build-trainer-paddle build-trainer-timm build-trainer-llm \
        build-optimizer build-evaluator build-evaluator-xpu build-ovms-xpu \
        plugin-build plugin-validate images-build helm-package installer-package \
        build-runtime build-xpu-base build-trainer-vendor \
        build-ml-backend-custom build-jupyter-base \
        test test-scripts nf-verify lint fmt license-check license-report scaffold-plugin

# help prints the local-only development, build, and verification interface.
help:
	@echo "NereidForge local build and verification"
	@echo ""
	@echo "Development:"
	@echo "  make dev-build           Prepare the repository venv and frontend dependencies"
	@echo "  make dev-all             Start host-side backend and frontend development servers"
	@echo "  make dev-clean           Stop the development servers without deleting data"
	@echo ""
	@echo "Local binary and image builds:"
	@echo "  make build-api           Prepare the Python backend environment"
	@echo "  make build-frontend      Build the frontend production bundle"
	@echo "  make docker-build        Build backend and frontend images"
	@echo "  make images-build        Build the core image set (TRAINER_VENDORS is optional)"
	@echo "  make build-trainer       Build trainer-torch-cpu"
	@echo "  make build-trainer-paddle Build trainer-paddle-cpu"
	@echo "  make build-trainer-llm   Build the approved legacy trainer-llm definition locally"
	@echo "  make build-runtime       Build nf-runtime"
	@echo "  make build-optimizer     Build train-optimizer"
	@echo "  make build-evaluator     Build train-evaluator"
	@echo "  make build-evaluator-xpu Build train-evaluator-xpu"
	@echo "  make build-xpu-base      Build trainer-xpu-base (requires TORCH_XPU_INDEX)"
	@echo "  make build-trainer-vendor kind=torch vendor=xpu  Build one vendor trainer"
	@echo "  make build-ml-backend    Build ml-backend"
	@echo "  make build-ml-backend-custom  Build ml-backend-custom"
	@echo "  make build-jupyter-base  Build prediction-jupyter-base"
	@echo "  make build-ovms-xpu      Build the Intel GPU OVMS image"
	@echo ""
	@echo "Plugins, packaging, and verification:"
	@echo "  make plugin-build name=tensorrt  Build the local opt-tensorrt plugin image"
	@echo "  make plugin-validate name=tensorrt out=/tmp/out"
	@echo "  make helm-package       Archive exactly three NF charts (no upload)"
	@echo "  make installer-package  Deprecated local-only legacy full-stack archive"
	@echo "  make -C deploy/pre-nereidforge helm-package  Archive five pre charts"
	@echo "  make test / lint / fmt / license-check"
	@echo "  make test-scripts    Run stdlib script integration tests without dev dependencies"
	@echo "  make nf-verify       Verify application/suite delivery and NF charts offline"
	@echo ""
	@echo "Every image build carries three tags (VERSION, VERSION-<utc-ts>, VERSION-<git-sha>);"
	@echo "no target publishes an artifact or deploys to a cluster."

# Shared local build variables. Node tools resolve from PATH on every host.
# No tracked VERSION file: an explicit RELEASE_TAG, generated installer
# release.env, or exact Git tag supplies the version (treaty §19.5). GitHub's
# automatic source archives have neither Git metadata nor release.env and fail
# closed instead of silently building the rolling development channel.
ifeq ($(wildcard .git),)
ifeq ($(wildcard release.env),)
$(error Unsupported unversioned source archive; use nereidforge-installer-<version>.tar.gz or a Git checkout)
endif
endif
VERSION ?= $(shell if [ -n "$${RELEASE_TAG:-}" ]; then printf '%s\n' "$$RELEASE_TAG"; elif [ -f release.env ]; then . ./release.env; printf '%s\n' "$$RELEASE_TAG"; else ref=$$(git describe --tags --exact-match --exclude 'pre-nereidforge/v*' 2>/dev/null) && printf '%s\n' "$${ref#v}" || printf '%s\n' 0.0.0-dev; fi)
# Three-tag scheme (treaty §19.2): every local image build tags the moving
# VERSION coordinate plus UTC-timestamp and git-short-SHA traceability tags.
# Local builds stay native/single-arch; multi-arch indexes are a publication-
# time property of the tag-triggered GitHub Actions workflow.
BUILD_TS := $(shell date -u +%Y%m%d-%H%M)
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILD_DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
# OCI labels use the same tag-derived version and source revision as the local
# tags, plus an RFC3339 creation timestamp independent of the compact tag stamp.
IMAGE_METADATA_ARGS := --build-arg VERSION=$(VERSION) --build-arg GIT_SHA=$(GIT_SHA) --build-arg BUILD_DATE=$(BUILD_DATE)
# three_tags expands to the three -t flags for one full reference
# (repo:version-component). The notebook workbench passes its py3.12 ref, so
# the same helper produces py3.12, py3.12-<ts> and py3.12-<sha>.
three_tags = -t $(1) -t $(1)-$(BUILD_TS) -t $(1)-$(GIT_SHA)
NPM ?= npm
NPX ?= npx
NODE ?= node

# Generic network endpoints use pristine upstream defaults when empty.
# Specialized XPU/ROCm/Paddle-GPU indexes are mandatory and checked by their
# build owners; generic PyPI is never a valid substitute for those wheels.
# ?= keeps an environment-provided value, including an explicit empty one.
PIP_INDEX_URL    ?=
PIP_TRUSTED_HOST ?=
APT_MIRROR       ?=
TORCH_CPU_INDEX  ?=
TORCH_XPU_INDEX  ?=
TORCH_ROCM_INDEX ?=
HF_ENDPOINT      ?=
PADDLE_GPU_INDEX ?=
MIRROR_ARGS := \
	--build-arg PIP_INDEX_URL=$(PIP_INDEX_URL) \
	--build-arg PIP_TRUSTED_HOST=$(PIP_TRUSTED_HOST) \
	--build-arg APT_MIRROR=$(APT_MIRROR) \
	--build-arg TORCH_CPU_INDEX=$(TORCH_CPU_INDEX) \
	--build-arg TORCH_XPU_INDEX=$(TORCH_XPU_INDEX) \
	--build-arg TORCH_ROCM_INDEX=$(TORCH_ROCM_INDEX) \
	--build-arg PYPI_INDEX=$(PIP_INDEX_URL) \
	--build-arg HF_ENDPOINT=$(HF_ENDPOINT) \
	--build-arg PADDLE_GPU_INDEX=$(PADDLE_GPU_INDEX)
# Explicit VERSION keeps every image reference aligned with its OCI version
# when a caller overrides the default; the notebook retains its py3.12 tag.
BACKEND_IMG := $(shell bash scripts/image-ref.sh nereidforge-backend $(VERSION))
FRONTEND_IMG := $(shell bash scripts/image-ref.sh nereidforge-frontend $(VERSION))
MLBACKEND_IMG := $(shell bash scripts/image-ref.sh ml-backend/ml-backend $(VERSION))
TRAINER_IMG := $(shell bash scripts/image-ref.sh trainer/torch-cpu $(VERSION))
TRAINER_PADDLE_IMG := $(shell bash scripts/image-ref.sh trainer/paddle-cpu $(VERSION))
TRAINER_TIMM_IMG := $(TRAINER_IMG)
TRAINER_LLM_IMG := $(shell bash scripts/image-ref.sh trainer-llm $(VERSION))
OPTIMIZER_IMG := $(shell bash scripts/image-ref.sh train/optimizer $(VERSION))
EVALUATOR_IMG := $(shell bash scripts/image-ref.sh train/evaluator $(VERSION))
EVALUATOR_XPU_IMG := $(shell bash scripts/image-ref.sh train/evaluator-xpu $(VERSION))
RUNTIME_IMG := $(shell bash scripts/image-ref.sh runtime $(VERSION))
XPU_BASE_IMG := $(shell bash scripts/image-ref.sh trainer/xpu-base $(VERSION))
MLBACKEND_CUSTOM_IMG := $(shell bash scripts/image-ref.sh ml-backend/ml-backend-custom $(VERSION))
JUPYTER_BASE_IMG := $(shell bash scripts/image-ref.sh prediction/jupyter-base py3.12)
JUPYTER_BASE_UPSTREAM := $(shell bash scripts/image-ref.sh --env IMG_JUPYTER_BASE)
# OVMS + Intel NEO userspace for Arc GPU serving (class-A self-built image;
# base is the mirrored/pinned upstream OVMS). CI uses the Dockerfile default;
# local/offline builds pass OVMS_BASE_IMG=<mirror>/inference/model-server-openvino:0.0.1
OVMS_XPU_IMG := $(shell bash scripts/image-ref.sh inference/model-server-openvino-xpu $(VERSION))
OVMS_BASE_IMG ?=
# Optional public-base overrides for local mirrors. Empty-independent defaults
# match the Dockerfiles; specialized vendor wheel guards remain mandatory.
ML_PYTHON_BASE ?= python:3.11-slim
UBUNTU_BASE_IMAGE ?= ubuntu:24.04
# Browser-side dataset upload guards are Vite build-time settings. The backend
# limits are overridden independently with NEREIDFORGE_STORAGE_UPLOAD_MAX_*.
VITE_DATASET_UPLOAD_MAX_FILES ?= 200
VITE_DATASET_UPLOAD_MAX_BYTES ?= 1073741824
FE_UPLOAD_ARGS := --build-arg VITE_DATASET_UPLOAD_MAX_FILES=$(VITE_DATASET_UPLOAD_MAX_FILES) --build-arg VITE_DATASET_UPLOAD_MAX_BYTES=$(VITE_DATASET_UPLOAD_MAX_BYTES)

PYTHON_BACKEND_DIR := python-backend
FRONTEND_DIR := frontend
ML_BACKEND_DIR := builders/ml-backend
TRAINER_DIR := builders/trainer-torch
PLUGINS_DIR := optimizer-plugins

# === Development environment ===
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
PY_BACKEND := $(VENV_PY)
DEV_BACKEND_PORT := 8080
DEV_FRONTEND_PORT := 5173

# dev-build prepares the shared repository venv and frontend dependencies.
dev-build:
	@test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	$(VENV_PIP) install -e ./python-backend
	$(VENV_PIP) install kfp==2.17.0 kale nbformat
	$(VENV_PIP) install ruff pytest
	@cd $(FRONTEND_DIR) && $(NPM) ci --ignore-scripts
	@echo "dev-build complete: .venv and frontend dependencies are ready"

# dev-all starts host-side hot reload against the existing development data services.
dev-all: dev-build
	@setsid nohup ./scripts/run-dev.sh > /tmp/nereidforge-api.log 2>&1 < /dev/null &
	@cd $(FRONTEND_DIR) && setsid nohup $(NPM) run dev > /tmp/nereidforge-frontend.log 2>&1 < /dev/null &
	@sleep 3
	@echo "dev-all started in the background:"
	@echo "  Backend (0.0.0.0:8080, local: http://127.0.0.1:$(DEV_BACKEND_PORT))  (log: /tmp/nereidforge-api.log)"
	@echo "  Frontend (0.0.0.0:5173, local: http://127.0.0.1:$(DEV_FRONTEND_PORT)) (log: /tmp/nereidforge-frontend.log)"
	@echo "  Existing development services and the nereidforge_dev data domain remain unchanged"

# dev-clean stops only the host development listeners and preserves all data.
dev-clean:
	@-fuser -k $(DEV_BACKEND_PORT)/tcp 2>/dev/null || true
	@-fuser -k $(DEV_FRONTEND_PORT)/tcp 2>/dev/null || true
	@echo "dev-clean: stopped the development listeners without deleting data"

# === Local image builds ===
# docker-build builds the backend and frontend without publishing either image.
# Backend Dockerfile uses repo-root-relative COPY paths (python-backend,
# configs), so the build context must be the repo root, not the backend dir.
docker-build:
	docker build $(IMAGE_METADATA_ARGS) $(MIRROR_ARGS) -f python-backend/Dockerfile $(call three_tags,$(BACKEND_IMG)) .
	docker build $(IMAGE_METADATA_ARGS) --build-arg VITE_APP_VERSION=$(VERSION) --build-arg VITE_IMAGE_TAG=$(VERSION) $(FE_UPLOAD_ARGS) $(call three_tags,$(FRONTEND_IMG)) $(FRONTEND_DIR)
	@echo "docker-build complete: backend and frontend images are local only (three tags, nothing pushed)"

# === Python backend environment ===
build-api:
	@test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	$(VENV_PIP) install -e ./python-backend
	$(VENV_PIP) install kfp==2.17.0 kale nbformat

# === Frontend production bundle ===
build-frontend:
	cd $(FRONTEND_DIR) && $(NPM) ci && $(NPM) run build

# === ML Backend local image ===
# The resolver collapses the historical category/name pair to the canonical
# flat ml-backend repository used by the Helm chart and built-in catalog.
build-ml-backend:
	docker build $(IMAGE_METADATA_ARGS) $(MIRROR_ARGS) --build-arg PYTHON_BASE_IMAGE=$(ML_PYTHON_BASE) $(call three_tags,$(MLBACKEND_IMG)) $(ML_BACKEND_DIR)

# === NereidForge local image builds; publication is intentionally absent ===
# trainer/torch-cpu — torchvision + timm + smp, every vision task in one image
# (per-vendor GPU variants: make build-trainer-vendor kind=torch vendor=cu|rocm|xpu)
build-trainer:
	docker build $(IMAGE_METADATA_ARGS) $(MIRROR_ARGS) $(call three_tags,$(TRAINER_IMG)) -f builders/trainer-torch/Dockerfile builders

# trainer/paddle-cpu — PaddleX / PP-YOLOE (second supported framework);
# context builders so the component and the Trainer SDK ship together.
build-trainer-paddle:
	docker build $(IMAGE_METADATA_ARGS) $(MIRROR_ARGS) $(call three_tags,$(TRAINER_PADDLE_IMG)) -f builders/trainer-paddle/Dockerfile builders

# Legacy alias: classification moved into the torch image (ADR-014 D1).
build-trainer-timm: build-trainer

# Approved legacy trainer-llm definition: local verification only, never a
# production publication source. Vendor-specific LLM images use the target below.
build-trainer-llm:
	docker build $(IMAGE_METADATA_ARGS) $(MIRROR_ARGS) $(call three_tags,$(TRAINER_LLM_IMG)) -f kfp-pipelines/docker/Dockerfile.trainer-llm kfp-pipelines/docker

# optimizer (FROM self-built runtime base -> pass its profile-resolved ref)
build-optimizer:
	docker build $(IMAGE_METADATA_ARGS) $(MIRROR_ARGS) $(call three_tags,$(OPTIMIZER_IMG)) --build-arg RUNTIME_IMAGE=$(RUNTIME_IMG) -f builders/optimizer/Dockerfile builders

# evaluator
build-evaluator:
	docker build $(IMAGE_METADATA_ARGS) $(MIRROR_ARGS) $(call three_tags,$(EVALUATOR_IMG)) --build-arg RUNTIME_IMAGE=$(RUNTIME_IMG) -f builders/evaluator/Dockerfile builders

# Intel GPU packaging/verification sidecar (Ubuntu XPU base + OpenVINO). It
# depends on trainer/xpu-base, the public flat GHCR image trainer-xpu-base.
build-evaluator-xpu:
	docker build $(IMAGE_METADATA_ARGS) $(MIRROR_ARGS) $(call three_tags,$(EVALUATOR_XPU_IMG)) --build-arg XPU_BASE_IMAGE=$(XPU_BASE_IMG) -f builders/evaluator-xpu/Dockerfile builders

# OVMS with Intel NEO userspace (Arc/Battlemage GPU serving). Base image is an
# overridable build ARG (public upstream default); offline builds pass a
# site-mirrored OVMS base via OVMS_BASE_IMG=<registry>/inference/model-server-openvino:0.0.1
# (the node containerd mirror normally makes this unnecessary).
build-ovms-xpu:
	docker build $(IMAGE_METADATA_ARGS) $(MIRROR_ARGS) $(call three_tags,$(OVMS_XPU_IMG)) $(if $(OVMS_BASE_IMG),--build-arg BASE_IMAGE=$(OVMS_BASE_IMG)) -f builders/inference-model-server-openvino-xpu/Dockerfile builders/inference-model-server-openvino-xpu

# KFP step runtime base (generated Dockerfile.runtime is tracked; builds the
# python:3.14-slim + kfp closure image that optimizer/evaluator/timm FROM).
# Local build only: GHCR publication is owned independently by the
# tag-triggered publish-images workflow, which never calls make.
build-runtime:
	docker build $(IMAGE_METADATA_ARGS) $(MIRROR_ARGS) $(call three_tags,$(RUNTIME_IMG)) -f kfp-pipelines/docker/Dockerfile.runtime kfp-pipelines/docker

# Intel XPU vendor base (Ubuntu + XPU userspace + torch+xpu); prerequisite of
# the torch-xpu / llm-xpu images. Its specialized wheel index is mandatory.
build-xpu-base:
	@test -n "$(TORCH_XPU_INDEX)" || (echo "ERROR: set TORCH_XPU_INDEX for +xpu wheels"; exit 1)
	docker build $(IMAGE_METADATA_ARGS) $(MIRROR_ARGS) --build-arg BASE_IMAGE=$(UBUNTU_BASE_IMAGE) $(call three_tags,$(XPU_BASE_IMG)) builders/trainer-xpu

# One per-vendor trainer image: make build-trainer-vendor kind=torch vendor=cu
build-trainer-vendor:
	@test -n "$(kind)" -a -n "$(vendor)" || (echo "Usage: make build-trainer-vendor kind=torch|paddle|llm vendor=cu|rocm|xpu"; exit 1)
	RELEASE_TAG=$(VERSION) GIT_SHA=$(GIT_SHA) BUILD_DATE=$(BUILD_DATE) bash scripts/build-vendor-trainer.sh $(kind) $(vendor)

# ml-backend-custom (custom vLLM/OCR families; FROM python:3.11-slim)
build-ml-backend-custom:
	docker build $(IMAGE_METADATA_ARGS) $(MIRROR_ARGS) --build-arg PYTHON_BASE_IMAGE=$(ML_PYTHON_BASE) $(call three_tags,$(MLBACKEND_CUSTOM_IMG)) builders/ml-backend-custom

# Jupyter workbench (keeps its own py3.12 version component; FROM the
# quay.io/jupyter upstream base referenced by JUPYTER_BASE_UPSTREAM). The
# three-tag helper yields py3.12, py3.12-<ts> and py3.12-<sha>.
build-jupyter-base:
	docker build $(IMAGE_METADATA_ARGS) $(MIRROR_ARGS) --build-arg JUPYTER_BASE_IMAGE=$(JUPYTER_BASE_UPSTREAM) $(call three_tags,$(JUPYTER_BASE_IMG)) builders/notebook-base

# === Plugin build/validation (local images only; no push target) ===
# PLUGIN_IMG resolves the optional developer artifact through the same public
# flat-name/VERSION scheme as every other local suite image. Plugin tags are not
# independent release versions and no registry override is accepted.
PLUGIN_IMG = $(shell bash scripts/image-ref.sh opt-$(name) $(VERSION))
plugin-build:
	@test -n "$(name)" || (echo "Usage: make plugin-build name=<plugin>"; exit 1)
	docker build $(IMAGE_METADATA_ARGS) $(call three_tags,$(PLUGIN_IMG)) $(PLUGINS_DIR)/$(name)

plugin-validate:
	@test -n "$(name)" || (echo "Usage: make plugin-validate name=<plugin> out=<dir>"; exit 1)
	@test -n "$(out)" || (echo "Usage: make plugin-validate name=<plugin> out=<dir>"; exit 1)
	$(VENV_PY) $(PLUGINS_DIR)/$(name)/tests/test_manifest.py
	$(VENV_PY) $(PLUGINS_DIR)/$(name)/tests/test_smoke.py --output $(out)

# images-build runs the aggregate local build. Backend, frontend, CPU trainers,
# converters, and the selected vendor matrix remain in the local Docker daemon,
# each carrying the three §19.2 tags. The script never pushes.
images-build:
	VERSION=$(VERSION) GIT_SHA=$(GIT_SHA) BUILD_DATE=$(BUILD_DATE) bash scripts/02_build-images.sh

# === Local chart and legacy installer packaging (no push target) ===
# helm-package archives exactly the three NF product charts under charts-packaged/.
# Supporting charts are packaged independently by deploy/pre-nereidforge.
# installer-package remains only for legacy local full-stack source compatibility;
# NF GitHub Releases no longer build or attach this repository-shaped archive.
helm-package:
	bash scripts/03_package-charts.sh

installer-package:
	@echo "WARNING: deprecated local-only compatibility archive; not an NF release asset" >&2
	bash scripts/04_package-installer.sh

# === Code quality ===
# Standalone stdlib integration tests need neither the repository venv nor npm.
test-scripts:
	@echo "=== Script integration tests ==="
	python3 -m unittest discover -s scripts/tests -p 'test_*.py' -v

# Application-owned first delivery seam. These stdlib tests include backend and
# frontend coordinate/compatibility checks plus offline Helm lint/render/package
# checks when Helm is installed. They never run pre or builder unit tests,
# install dependencies, build images, contact registries, or publish artifacts.
nf-verify:
	@echo "=== Application and suite delivery verification ==="
	python3 -m unittest scripts.tests.test_nf_delivery scripts.tests.test_nf_publication scripts.tests.test_nf_legacy_compatibility scripts.tests.test_online_installer_verification -v
	bash -n scripts/lib/nf_publication.sh

test: test-scripts
	@echo "=== Python tests ==="
	@# Component roots isolate identical top-level module names during collection.
	cd $(PLUGINS_DIR)/tensorrt && ../../$(VENV_PY) -m pytest tests -v
	cd $(ML_BACKEND_DIR) && ../../$(VENV_PY) -m pytest tests -v
	cd $(TRAINER_DIR) && ../../$(VENV_PY) -m pytest tests -v
	cd builders && ../$(VENV_PY) -m pytest tests -v
	cd $(PYTHON_BACKEND_DIR) && ../$(VENV_PY) -m pytest tests -v
	@echo "=== Frontend tests ==="
	cd $(FRONTEND_DIR) && npm test

lint:
	@echo "=== Python lint ==="
	$(VENV_PY) -m ruff check $(PLUGINS_DIR) $(ML_BACKEND_DIR) $(TRAINER_DIR) $(PYTHON_BACKEND_DIR)
	@echo "=== Frontend lint ==="
	cd $(FRONTEND_DIR) && npm run lint

fmt:
	@echo "=== Python fmt ==="
	$(VENV_PY) -m ruff format $(PLUGINS_DIR) $(ML_BACKEND_DIR) $(TRAINER_DIR) $(PYTHON_BACKEND_DIR)
	@echo "=== Frontend fmt ==="
	cd $(FRONTEND_DIR) && npm run format

# === License compliance (LICENSE-POLICY.md) ===
license-check:
	bash scripts/check-license-compliance.sh
	bash scripts/check-legal-copies.sh

license-report:
	bash scripts/check-license-compliance.sh --env $(VENV_PY)

# === Local plugin scaffold generation ===
scaffold-plugin:
	@test -n "$(name)" || (echo "Usage: make scaffold-plugin name=<name> kind=optimizer|evaluator"; exit 1)
	@test -n "$(kind)" || (echo "Usage: make scaffold-plugin name=<name> kind=optimizer|evaluator"; exit 1)
	@mkdir -p $(PLUGINS_DIR)/$(name)/tests
	@cat > $(PLUGINS_DIR)/$(name)/manifest.yaml <<-'EOF'
	name: nereid-$(name)
	kind: $(kind)
	version: 0.1.0
	vendor: nvidia
	supported_formats:
	  input: ["onnx"]
	  output: ["plan"]
	hardware_target: nvidia-tensorrt
	supported_precision: ["fp32", "fp16", "int8"]
	architectures: ["x86_64", "arm64"]
	resources:
	  requests: { cpu: "2", memory: "8Gi", gpu: 1 }
	  limits: { cpu: "8", memory: "32Gi", gpu: 1 }
	environment:
	  required_vars: ["CUDA_VISIBLE_DEVICES"]
	  optional_vars: ["TRT_CACHE_DIR"]
	EOF
	@cat > $(PLUGINS_DIR)/$(name)/Dockerfile <<-'DOCKER_EOF'
	FROM nvcr.io/nvidia/tensorrt:24.06-py3
	WORKDIR /opt
	COPY manifest.yaml /etc/plugin/manifest.yaml
	COPY plugin.py /opt/plugin.py
	RUN chmod +x /opt/plugin.py
	ENTRYPOINT ["python", "/opt/plugin.py"]
	DOCKER_EOF
	@cat > $(PLUGINS_DIR)/$(name)/plugin.py <<-'PY_EOF'
	#!/usr/bin/env python3
	import argparse, json, os, sys
	from pathlib import Path

	def main():
	    parser = argparse.ArgumentParser()
	    parser.add_argument("--input-uri", required=True)
	    parser.add_argument("--output-dir", required=True)
	    parser.add_argument("--config-json", required=True)
	    parser.add_argument("--artifact-id", required=True)
	    args = parser.parse_args()

	    cfg = json.loads(args.config_json)
	    output_dir = Path(args.output_dir)
	    output_dir.mkdir(parents=True, exist_ok=True)

	    print(f"Plugin {args.artifact_id} stub executed", file=sys.stderr)
	    sys.exit(0)

	if __name__ == "__main__":
	    main()
	PY_EOF
	@cat > $(PLUGINS_DIR)/$(name)/tests/test_manifest.py <<-'TEST_EOF'
	import yaml, sys
	from pathlib import Path

	def test_manifest():
	    manifest = yaml.safe_load(Path(__file__).parent.parent.joinpath("manifest.yaml").read_text())
	    assert manifest["name"].startswith("nereid-")
	    assert manifest["kind"] in ("optimizer", "evaluator")
	    assert "vendor" in manifest
	    assert "hardware_target" in manifest
	    assert "supported_precision" in manifest
	    assert "architectures" in manifest

	if __name__ == "__main__":
	    test_manifest()
	    print("Manifest test passed")
	TEST_EOF
	@cat > $(PLUGINS_DIR)/$(name)/tests/test_smoke.py <<-'SMOKE_EOF'
	#!/usr/bin/env python3
	import argparse, json, subprocess, sys, tempfile
	from pathlib import Path

	def main():
	    parser = argparse.ArgumentParser()
	    parser.add_argument("--output", required=True)
	    args = parser.parse_args()

	    # Build plugin image
	    plugin_dir = Path(__file__).parent.parent
	    image = f"nereid-plugin-{plugin_dir.name}:test"
	    subprocess.run(["docker", "build", "-t", image, str(plugin_dir)], check=True)

	    # Run stub
	    with tempfile.TemporaryDirectory() as tmp:
	        result = subprocess.run([
	            "docker", "run", "--rm",
	            "-v", f"{tmp}:/output",
	            image,
	            "--input-uri", "dummy",
	            "--output-dir", "/output",
	            "--config-json", "{}",
	            "--artifact-id", "test"
	        ], capture_output=True, text=True)
	        if result.returncode != 0:
	            print(f"Smoke test failed: {result.stderr}")
	            sys.exit(1)

	    Path(args.output).write_text("smoke test passed\n")
	    print("Smoke test passed")

	if __name__ == "__main__":
	    main()
	SMOKE_EOF
	@chmod +x $(PLUGINS_DIR)/$(name)/plugin.py
	@echo "Plugin scaffold created at $(PLUGINS_DIR)/$(name)"