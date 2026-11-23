# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

PRE_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PRE_OUTPUT_DIR ?= $(PRE_ROOT)/dist

.PHONY: help verify package

help:
	@echo "Pre-NereidForge local interface (no cluster or registry operations)"
	@echo "  make verify    Run stdlib/shell tests, package reparses, and chart checks"
	@echo "  make package   Build and reparse the versioned ONLINE source installer"
	@echo ""
	@echo "Optional package overrides:"
	@echo "  PRE_RELEASE_TAG=0.0.1 PRE_OUTPUT_DIR=/tmp/pre-release make package"

verify:
	@bash "$(PRE_ROOT)/scripts/verify.sh"

package:
	@PRE_RELEASE_TAG="$(PRE_RELEASE_TAG)" PRE_OUTPUT_DIR="$(PRE_OUTPUT_DIR)" \
		bash "$(PRE_ROOT)/scripts/package.sh"
