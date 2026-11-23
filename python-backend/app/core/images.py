# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for NereidForge self-built image references.

Every tracked manifest names public coordinates only (root treaty §18):
self-built images always resolve to ``ghcr.io/nereidai/<flat-name>`` and
class-B serving runtimes keep their true upstream coordinates. Air-gapped
clusters resolve the same refs through the node-level containerd registry
mirror configured by NereidOverview's harbor-sync tooling; there is no
in-process registry/layout switch.

EVERY self-built image ships with the tag-sourced release version (root
treaty §19.5): backend/frontend, trainers, evaluator, optimizer, the KFP
runtime, converters and ml-backend all move together when the git tag moves.
The bare SemVer is baked into the backend/worker image as NF_IMAGE_TAG by the
Docker build (--build-arg VERSION=<bare-semver>); a non-tag checkout or local
development falls back to the rolling 0.0.0-dev dev channel. The same tag
feeds the frontend build (VITE_IMAGE_TAG) so UI dropdowns can never select a
different version than the backend.

NOT covered (keep their own tags):
  * Upstream images: python:3.14-slim,
    pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel (CUDA trainer base, ADR-023),
    rocm/pytorch:rocm6.3_..., intel/intel-gpu-plugin:0.36.0, ...
  * prediction/jupyter-base keeps its Python-version tag (py3.12) by product
    decision (rebuild-in-place dual-use workbench image).
"""
from __future__ import annotations

import os

GHCR = "ghcr.io/nereidai"
# Rolling dev channel for non-tag checkouts; release images override this via
# the NF_IMAGE_TAG environment variable baked at build time.
FALLBACK_TAG = "0.0.0-dev"

# GHCR only allows <owner>/<package>; the historical category/name pairs are
# flattened with the category kept as a prefix. Aliases collapse duplicated
# prefixes (ml-backend/ml-backend-custom -> ml-backend-custom).
FLAT_ALIASES = {
    "ml-backend/ml-backend": "ml-backend",
    "ml-backend/ml-backend-custom": "ml-backend-custom",
    "runtime": "nf-runtime",
}

# Class B (upstream, not built by NF) serving runtimes. Pods always reference
# these original registries; an air-gapped node mirror swaps the host while
# preserving the path.
SERVING_UPSTREAMS = {
    "openvino": "openvino/model_server:latest",
    "triton": "nvcr.io/nvidia/tritonserver:26.01-py3",
    "torch": "pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime",
}


def _release_tag() -> str:
    """Resolve the release tag: the baked NF_IMAGE_TAG env, else the dev channel.

    The repository carries no VERSION file (root treaty §19.5); the git tag is
    the single version source and the Docker build injects its bare SemVer as
    NF_IMAGE_TAG. Anything outside a release image builds 0.0.0-dev.
    """
    return os.getenv("NF_IMAGE_TAG", "").strip() or FALLBACK_TAG


RELEASE_TAG = _release_tag()


def _flat_name(repo: str) -> str:
    return FLAT_ALIASES.get(repo, repo.replace("/", "-"))


def nf_image(repo: str, tag: str | None = None) -> str:
    """nf_image returns the flat GHCR reference of a self-built image.

    ``ghcr.io/nereidai/<category>-<name>:<tag>`` (aliases collapse duplicated
    prefixes); the tag defaults to the repo release tag.
    """
    tag = tag or RELEASE_TAG
    return f"{GHCR}/{_flat_name(repo)}:{tag}"


def serving_image(kind: str) -> str:
    """serving_image returns an upstream serving runtime image (class B: not self-built)."""
    return SERVING_UPSTREAMS[kind]
