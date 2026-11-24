# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Builder-owned verification only: no image publication, registry login, or
# cluster access. Use the repository's existing .venv when present; callers may
# override PYTHON with another prepared Python interpreter.
PYTHON ?= python3
ifneq ($(wildcard ../.venv/bin/python),)
PYTHON := $(abspath ../.venv/bin/python)
endif

TEST_PYTHONPATH := $(abspath sdk):$(abspath ml-backend/vendor/label-studio-ml)$(if $(PYTHONPATH),:$(PYTHONPATH))
PYTEST = PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(TEST_PYTHONPATH)" $(PYTHON) -m pytest

.PHONY: verify verify-buildx

# Offline unit and source-contract tests. Component working directories preserve
# the root suite's isolation of modules that share names such as common/models.
verify:
	@echo "=== Trainer SDK tests ==="
	cd sdk && $(PYTEST) tests -v
	@echo "=== Torch trainer tests ==="
	cd trainer-torch && $(PYTEST) tests -v
	@echo "=== Paddle trainer tests ==="
	cd trainer-paddle && $(PYTEST) tests -v
	@echo "=== ML backend tests ==="
	cd ml-backend && $(PYTEST) tests -v
	@echo "=== Shared builder tests ==="
	$(PYTEST) tests/test_s3_credentials.py -v
	@echo "=== Optimizer plugin tests ==="
	cd ../optimizer-plugins/tensorrt && $(PYTEST) tests -v
	@echo "=== Builder image inventory tests ==="
	$(PYTEST) tests/test_image_inventory.py -v
	@echo "=== Dockerfile and build-environment contract tests (offline) ==="
	$(PYTEST) tests/test_dockerfile_contracts.py -v

# Mandatory on the PR runner, separate from routine offline verification.
# The generated Dockerfile starts FROM scratch and replays only the XPU COPY
# instructions, so this checks a clean checkout without pulling a base image,
# installing dependencies, contacting a registry, or running a full image build.
verify-buildx:
	NF_REQUIRE_BUILDX=1 $(PYTEST) tests/test_dockerfile_contracts.py::BuilderDockerfileContractTest::test_xpu_dockerfile_copy_sources_build_from_scratch_in_clean_checkout -v
