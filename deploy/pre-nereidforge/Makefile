# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

PRE_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PRE_OUTPUT_DIR ?= $(PRE_ROOT)/dist
PRE_CHARTS_OUTPUT_DIR ?= $(PRE_OUTPUT_DIR)/charts-packaged
PRE_CHARTS_ARCHIVE ?= $(PRE_OUTPUT_DIR)/pre-nereidforge-charts.tar.gz

.PHONY: help verify package helm-package

help:
	@echo "Pre-NereidForge local interface (no cluster access or artifact publication)"
	@echo "  make verify    Run stdlib/shell tests, package reparses, and chart checks"
	@echo "  make package   Build and reparse the versioned ONLINE source installer"
	@echo "  make helm-package  Package exactly five supporting charts (no upload)"
	@echo ""
	@echo "Optional package overrides:"
	@echo "  PRE_RELEASE_TAG=0.0.1 PRE_OUTPUT_DIR=/tmp/pre-release make package"

verify:
	@bash "$(PRE_ROOT)/scripts/verify.sh"

package:
	@PRE_RELEASE_TAG="$(PRE_RELEASE_TAG)" PRE_OUTPUT_DIR="$(PRE_OUTPUT_DIR)" \
		bash "$(PRE_ROOT)/scripts/package.sh"

helm-package:
	@PRE_CHARTS_OUTPUT_DIR="$(PRE_CHARTS_OUTPUT_DIR)" \
		PRE_CHARTS_ARCHIVE="$(PRE_CHARTS_ARCHIVE)" \
		PRE_CHARTS_VERSION="$(PRE_CHARTS_VERSION)" \
		bash "$(PRE_ROOT)/scripts/package-charts.sh"
