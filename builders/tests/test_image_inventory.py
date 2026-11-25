# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Keep the builder-owned release inventory aligned with existing build gates."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = ROOT / "builders" / "image-inventory.json"
WORKFLOWS = ROOT / ".github" / "workflows"

EXPECTED_IMAGES = frozenset(
    {
        "inference-model-server-openvino-xpu",
        "ml-backend",
        "ml-backend-custom",
        "nf-runtime",
        "prediction-jupyter-base",
        "train-evaluator",
        "train-evaluator-xpu",
        "train-optimizer",
        "trainer-convert",
        "trainer-convert-pp",
        "trainer-convert-tf",
        "trainer-llm-cu",
        "trainer-llm-rocm",
        "trainer-llm-xpu",
        "trainer-paddle-cpu",
        "trainer-paddle-cu",
        "trainer-torch-cpu",
        "trainer-torch-cu",
        "trainer-torch-rocm",
        "trainer-torch-xpu",
        "trainer-xpu-base",
    }
)
KFP_IMAGES = frozenset(
    {"nf-runtime", "trainer-convert", "trainer-convert-pp", "trainer-convert-tf"}
)
AMD_NVIDIA_IMAGES = frozenset(
    {
        "trainer-llm-cu",
        "trainer-llm-rocm",
        "trainer-paddle-cu",
        "trainer-torch-cu",
        "trainer-torch-rocm",
    }
)
EXPECTED_DEPENDENCIES = {
    "train-evaluator": {"nf-runtime"},
    "train-optimizer": {"nf-runtime"},
    "train-evaluator-xpu": {"trainer-xpu-base"},
    "trainer-torch-xpu": {"trainer-xpu-base"},
    "trainer-llm-xpu": {"trainer-xpu-base"},
}
APPLICATION_IMAGES = frozenset({"nereidforge-backend", "nereidforge-frontend"})
DEPENDENCY_PATTERN = re.compile(
    r"ghcr\.io/nereidai/(nf-runtime|trainer-xpu-base):"
)


@pytest.fixture(scope="module")
def inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def images(inventory: dict) -> dict[str, dict]:
    return {image["name"]: image for image in inventory["images"]}


@lru_cache(maxsize=4)
def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _development_entries() -> dict[str, dict]:
    document = _workflow("publish-dev.yml")
    jobs = document["jobs"]
    entries = [
        *jobs["publish-base"]["strategy"]["matrix"]["include"],
        *jobs["publish-dependent"]["strategy"]["matrix"]["include"],
    ]
    assert len({entry["name"] for entry in entries}) == len(entries)
    return {entry["name"]: entry for entry in entries}


@lru_cache(maxsize=1)
def _production_steps() -> dict[str, dict]:
    document = _workflow("publish-images.yml")
    steps = {}
    for job_name in ("core", "vendor-xpu", "vendor-cu", "vendor-rocm"):
        for step in document["jobs"][job_name]["steps"]:
            if step.get("uses") != "docker/build-push-action@v6":
                continue
            match = re.search(
                r"ghcr\.io/nereidai/([a-z0-9-]+):", step["with"]["tags"]
            )
            assert match, step["name"]
            assert match.group(1) not in steps, match.group(1)
            steps[match.group(1)] = step["with"]
    return steps


@lru_cache(maxsize=1)
def _cache_warm_steps() -> dict[tuple[str, str], dict]:
    document = _workflow("ci.yml")
    steps = {}
    for job_name in ("core", "vendor-xpu", "vendor-cu", "vendor-rocm"):
        for step in document["jobs"][job_name]["steps"]:
            if step.get("uses") != "docker/build-push-action@v6":
                continue
            definition = step["with"]
            key = (definition["file"], definition["context"])
            assert key not in steps, key
            steps[key] = definition
    return steps


def _platforms(definition: dict) -> list[str]:
    return definition["platforms"].split(",")


def _assert_build_definition(image: dict, definition: dict) -> None:
    assert definition["file"] == image["dockerfile"], image["name"]
    assert definition["context"] == image["context"], image["name"]
    assert _platforms(definition) == image["platforms"], image["name"]


def test_inventory_has_exact_release_and_profile_counts(inventory: dict) -> None:
    assert inventory["schema"] == "nf.builder-image-inventory/1"
    names = [image["name"] for image in inventory["images"]]
    assert len(names) == 21
    assert len(set(names)) == 21
    assert set(names) == EXPECTED_IMAGES
    assert set(inventory["profiles"]) == {"full", "cpu-xpu"}
    assert len(inventory["profiles"]["full"]) == 21
    assert set(inventory["profiles"]["full"]) == EXPECTED_IMAGES
    assert len(inventory["profiles"]["cpu-xpu"]) == 16
    assert set(inventory["profiles"]["cpu-xpu"]) == EXPECTED_IMAGES - AMD_NVIDIA_IMAGES
    assert EXPECTED_IMAGES - set(inventory["profiles"]["cpu-xpu"]) == AMD_NVIDIA_IMAGES
    assert "opt-tensorrt" not in names
    assert all("opt-tensorrt" not in profile for profile in inventory["profiles"].values())


def test_inventory_declares_17_builder_and_four_kfp_dockerfiles(images: dict) -> None:
    builder_images = {
        name
        for name, image in images.items()
        if image["dockerfile"].startswith("builders/")
    }
    kfp_images = set(images) - builder_images
    assert len(builder_images) == 17
    assert kfp_images == KFP_IMAGES

    for name, image in images.items():
        dockerfile = Path(image["dockerfile"])
        context = Path(image["context"])
        assert context in dockerfile.parents, name
        assert (ROOT / dockerfile).is_file(), name
        assert (ROOT / context).is_dir(), name
        assert image["platforms"], name
        assert len(image["platforms"]) == len(set(image["platforms"])), name
        assert set(image["platforms"]) <= {"linux/amd64", "linux/arm64"}, name


def test_dependencies_are_explicit_and_reference_inventory_images(images: dict) -> None:
    actual = {}
    for name, image in images.items():
        dependencies = image["dependencies"]
        assert len(dependencies) == len(set(dependencies)), name
        assert name not in dependencies, name
        assert set(dependencies) <= set(images), name
        if dependencies:
            actual[name] = set(dependencies)
    assert actual == EXPECTED_DEPENDENCIES


def test_notebook_is_the_only_fixed_version_exception(images: dict) -> None:
    exceptions = {
        name: image.get("version")
        for name, image in images.items()
        if "version" in image
    }
    assert exceptions == {"prediction-jupyter-base": "py3.12"}

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "prediction/jupyter-base py3.12" in makefile
    notebook_tags = _production_steps()["prediction-jupyter-base"]["tags"]
    assert "prediction-jupyter-base:py3.12\n" in notebook_tags
    assert "prediction-jupyter-base:py3.12-${{ steps.stamp.outputs.sha }}" in notebook_tags
    assert "prediction-jupyter-base:py3.12-${{ steps.stamp.outputs.ts }}" in notebook_tags


def test_development_build_matrices_match_inventory(images: dict) -> None:
    entries = _development_entries()
    assert set(entries) == EXPECTED_IMAGES | APPLICATION_IMAGES
    for name, image in images.items():
        definition = entries[name]
        _assert_build_definition(image, definition)
        dependencies = set(DEPENDENCY_PATTERN.findall(definition.get("build-args") or ""))
        assert dependencies == set(image["dependencies"]), name


def test_production_build_steps_match_inventory_and_dependencies(images: dict) -> None:
    steps = _production_steps()
    assert set(steps) == EXPECTED_IMAGES
    for name, image in images.items():
        definition = steps[name]
        _assert_build_definition(image, definition)
        dependencies = set(DEPENDENCY_PATTERN.findall(definition.get("build-args") or ""))
        assert dependencies == set(image["dependencies"]), name
        assert definition["push"] is True, name
        assert definition["provenance"] is False, name
        assert definition["sbom"] is False, name


def test_cache_warm_builds_match_inventory_and_local_base_handoff(images: dict) -> None:
    steps = _cache_warm_steps()
    expected_keys = {(image["dockerfile"], image["context"]) for image in images.values()}
    assert set(steps) == expected_keys
    for name, image in images.items():
        definition = steps[(image["dockerfile"], image["context"])]
        _assert_build_definition(image, definition)
        dependencies = set(DEPENDENCY_PATTERN.findall(definition.get("build-args") or ""))
        assert dependencies == set(image["dependencies"]), name
        named_contexts = definition.get("build-contexts") or ""
        for dependency in image["dependencies"]:
            assert f"{dependency}:0.0.0-dev=oci-layout:///" in named_contexts, name
        assert definition["push"] is False, name


def test_builder_verification_workflow_is_pr_only_and_non_publishing() -> None:
    path = WORKFLOWS / "builders-verify.yml"
    text = path.read_text(encoding="utf-8")
    assert re.search(r"(?m)^on:\n  pull_request:\n", text)
    assert '"builders/**"' in text
    assert '"kfp-pipelines/docker/**"' in text
    assert '"optimizer-plugins/**"' in text
    assert '".github/workflows/builders-verify.yml"' in text
    assert re.search(r"(?m)^permissions:\n  contents: read\n", text)
    assert "make -C builders verify\n" in text
    assert "make -C builders verify-buildx\n" in text
    assert "onnx onnxruntime" in text
    assert "docker/setup-buildx-action@" in text
    makefile = (ROOT / "builders" / "Makefile").read_text(encoding="utf-8")
    assert "cd ../optimizer-plugins/tensorrt && $(PYTEST) tests -v" in makefile
    assert "workflow_dispatch:" not in text
    assert "packages: write" not in text
    assert "secrets." not in text
    assert "docker/login-action" not in text
    assert "push: true" not in text
