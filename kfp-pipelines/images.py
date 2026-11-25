# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Release tag for NF self-built KFP step images — tag-sourced version model.

KFP @dsl.component decorators evaluate ``base_image`` at import/compile time;
compile_all.py / upload_pipelines.py run inside the repo tree, where no
VERSION file exists (root treaty §19.5): the git tag is the single version
source, the backend image bakes its bare SemVer as NF_IMAGE_TAG, and every
non-tag context resolves the rolling 0.0.0-dev dev channel. Every self-built
step image moves with that tag; cutting a new git tag changes every compiled
pipeline.

All refs are public coordinates (root treaty §18): self-built steps always
resolve to flat ``ghcr.io/nereidai/<name>`` refs and base images keep their
true upstream coordinates. Air-gapped clusters resolve them through the
node-level containerd mirror; there is no registry/layout switch in code.

NOT covered (keep their own tags): upstream bases
(python:3.14-slim, pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel,
rocm/pytorch:rocm6.3, ...) and
prediction/jupyter-base, whose tag is the Python version (py3.12) by product
decision.

CAUTION: lightweight component function BODIES are serialized by kfp and run
in the cluster without this module — image literals needed inside a body must
stay inline (they are invisible to imports). Decorator arguments and
module-level constants used as decorator args are safe.
"""
from __future__ import annotations

import os

GHCR = "ghcr.io/nereidai"
# Rolling dev channel for non-tag checkouts; release images override this via
# the NF_IMAGE_TAG environment variable baked at build time.
FALLBACK_TAG = "0.0.0-dev"

FLAT_ALIASES = {
    "ml-backend/ml-backend": "ml-backend",
    "ml-backend/ml-backend-custom": "ml-backend-custom",
    "runtime": "nf-runtime",
}


def _release_tag() -> str:
    """Resolve the release tag: the baked NF_IMAGE_TAG env, else the dev channel."""
    return os.getenv("NF_IMAGE_TAG", "").strip() or FALLBACK_TAG


RELEASE_TAG = _release_tag()


def nf_image(repo: str) -> str:
    """nf_image returns the flat GHCR reference of a self-built image at the release tag."""
    flat = FLAT_ALIASES.get(repo, repo.replace("/", "-"))
    return f"{GHCR}/{flat}:{RELEASE_TAG}"


def base_image(name: str = "python") -> str:
    """base_image returns a public upstream base image reference.

    KFP step components build on the Docker Hub official Python image. Paddle
    and TensorFlow images use their explicit Python 3.11 build args instead of
    this default resolver.
    """
    if name != "python":
        raise KeyError(f"unknown base image: {name}")
    return "python:3.14-slim"
