# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Trainer catalog, task vocabulary and legacy image aliasing (ADR-014)."""

import pytest

from app.core.images import nf_image
from app.services.images import BUILTIN_SEED, LEGACY_TRAINER_REPOS, canonical_trainer_ref
from app.services.jobs import JobService


def _repos():
    return {entry["repository"].rsplit(":", 1)[0] for entry in BUILTIN_SEED}


def test_builtin_trainers_cover_torch_and_paddle_per_vendor():
    repos = _repos()
    for vendor in ("cpu", "cu", "rocm", "xpu"):
        assert nf_image(f"trainer/torch-{vendor}").rsplit(":", 1)[0] in repos
    for vendor in ("cpu", "cu"):
        assert nf_image(f"trainer/paddle-{vendor}").rsplit(":", 1)[0] in repos
    # Paddle ships no official ROCm wheels (torch-only on AMD) and no XPU
    # wheel exists on PyPI at all (zero xpu builds of paddlepaddle), so both
    # are deliberately absent from the default matrix (ADR-014 D-01).
    assert nf_image("trainer/paddle-rocm").rsplit(":", 1)[0] not in repos
    assert nf_image("trainer/paddle-xpu").rsplit(":", 1)[0] not in repos


def test_catalog_has_no_brand_or_legacy_trainer_repos():
    repos = _repos()
    for legacy in LEGACY_TRAINER_REPOS:
        assert nf_image(legacy).rsplit(":", 1)[0] not in repos
    descriptions = " ".join(str(e.get("description", "")) for e in BUILTIN_SEED).lower()
    assert "yolo" not in descriptions
    assert "ultralytics" not in descriptions


@pytest.mark.parametrize(
    "legacy",
    list(LEGACY_TRAINER_REPOS),
)
def test_legacy_trainer_refs_resolve_to_torch_images(legacy):
    nested = f"hub.example.com/nereidai/{legacy}:v9.9.9"
    flat = f"ghcr.io/nereidai/{legacy.replace('/', '-')}:v9.9.9"
    expected = nf_image(LEGACY_TRAINER_REPOS[legacy]).rsplit(":", 1)[0]
    assert canonical_trainer_ref(nested).rsplit(":", 1)[0] == expected
    assert canonical_trainer_ref(flat).rsplit(":", 1)[0] == expected
    # The legacy tag is preserved (a pinned job keeps its pinned version).
    assert canonical_trainer_ref(nested).endswith(":v9.9.9")


def test_custom_and_current_refs_pass_through():
    custom = "registry.example.com/team/custom-trainer:dev"
    assert canonical_trainer_ref(custom) == custom
    current = nf_image("trainer/torch-cu")
    assert canonical_trainer_ref(current) == current


@pytest.mark.parametrize(
    ("markers", "expected"),
    [
        ({"task_type": "detection"}, "detection"),
        ({"task_type": "segmentation"}, "instance_seg"),
        ({"task_type": "mask"}, "instance_seg"),
        ({"task_type": "keypoint"}, "pose"),
        ({"task_type": "classification"}, "classification"),
        ({"task_type": "timm"}, "classification"),
        ({"task_type": "multilabel"}, "multilabel"),
        ({"task": "semantic_segmentation"}, "semantic_seg"),
        ({"task_type": "obb"}, "detection"),
        ({}, "detection"),
    ],
)
def test_job_task_normalisation(markers, expected):
    assert JobService._job_task(markers, {}) == expected


def test_default_archs_cover_every_trainer_task():
    # Keep in sync with TASKS in builders/trainer-torch/models.py.
    trainer_tasks = {
        "classification", "multilabel", "detection",
        "instance_seg", "semantic_seg", "pose",
    }
    assert set(JobService.DEFAULT_ARCHS) == trainer_tasks
