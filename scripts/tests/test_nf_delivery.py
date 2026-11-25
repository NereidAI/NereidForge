#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Application-owned image composition and delivery verification contracts."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
APPLICATION_INVENTORY = ROOT / "configs" / "application-image-inventory.json"
BUILDER_INVENTORY = ROOT / "builders" / "image-inventory.json"
PUBLISH_DEV = ROOT / ".github" / "workflows" / "publish-dev.yml"
PUBLISH_IMAGES = ROOT / ".github" / "workflows" / "publish-images.yml"
PACKAGE_CHARTS = ROOT / "scripts" / "03_package-charts.sh"
EXPORT_IMAGES = ROOT / "scripts" / "05_export-images.sh"

APPLICATION_IMAGES = {"nereidforge-backend", "nereidforge-frontend"}
AMD_NVIDIA_IMAGES = {
    "trainer-llm-cu",
    "trainer-llm-rocm",
    "trainer-paddle-cu",
    "trainer-torch-cu",
    "trainer-torch-rocm",
}
BUILDER_IMAGES = {
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
FULL_IMAGES = BUILDER_IMAGES | APPLICATION_IMAGES
CPU_XPU_IMAGES = FULL_IMAGES - AMD_NVIDIA_IMAGES
# Hardware dev profiles publish builder images only: the two application
# images are exclusive to the backend-frontend profile (16 and 21 images).
DEV_HARDWARE_IMAGES = BUILDER_IMAGES - AMD_NVIDIA_IMAGES
DEV_FULL_IMAGES = BUILDER_IMAGES
NF_CHARTS = {
    "nereidforge": "charts/nereidforge",
    "nereidforge-ml-backend": "charts/ml-backend",
    "nereidforge-tenant-check": "charts/tenant-check",
}
APPLICATION_DEFINITIONS = {
    "nereidforge-backend": {
        "context": ".",
        "file": "python-backend/Dockerfile",
        "platforms": ["linux/amd64", "linux/arm64"],
    },
    "nereidforge-frontend": {
        "context": "frontend",
        "file": "frontend/Dockerfile",
        "platforms": ["linux/amd64", "linux/arm64"],
    },
}


def inventory_entries(path):
    """Read the list-based image inventory schema used by both Modules."""
    document = json.loads(path.read_text(encoding="utf-8"))
    raw_entries = document.get("images") if isinstance(document, dict) else None
    if not isinstance(raw_entries, list):
        raise AssertionError(f"inventory images must be a list: {path}")
    entries = {}
    for entry in raw_entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise AssertionError(f"every inventory image needs a string name: {path}")
        name = entry["name"]
        if name in entries:
            raise AssertionError(f"duplicate image {name!r}: {path}")
        entries[name] = entry
    return entries


def job_block(text, name):
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [a-z][a-z0-9-]*:\n|\Z)",
        text,
    )
    if not match:
        raise AssertionError(f"workflow job not found: {name}")
    return match.group(1)


def matrix_entry(text, job_name, name):
    match = re.search(
        rf"(?ms)^          - name: {re.escape(name)}\s*$\n(.*?)(?=^          - name:|\Z)",
        job_block(text, job_name),
    )
    if not match:
        raise AssertionError(f"matrix entry not found: {job_name}/{name}")
    return match.group(0)


def matrix_names(text, job_name):
    return re.findall(
        r"(?m)^          - name: ([a-z0-9-]+)\s*$",
        job_block(text, job_name),
    )


def step_block(text, marker):
    start = text.index(marker)
    end = text.find("\n      - name:", start + len(marker))
    return text[start:] if end == -1 else text[start:end]


def run_block(text, marker):
    """Return a literal run block after a unique step marker, without PyYAML."""
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines) if marker in line]
    if len(matches) != 1:
        raise AssertionError(f"expected one step marker {marker!r}, found {matches}")
    start = matches[0]
    run_start = next(
        (index for index in range(start + 1, len(lines)) if lines[index].strip() == "run: |"),
        None,
    )
    if run_start is None:
        raise AssertionError(f"step {marker!r} has no literal run block")
    content = []
    for line in lines[run_start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) < 10:
            break
        content.append(line[10:] if len(line) >= 10 else "")
    return "\n".join(content) + "\n"


def workflow_image_lists(text):
    """Extract the four literal profile lists embedded in publish-dev."""
    preflight = job_block(text, "preflight")
    alias = job_block(text, "alias-images")
    matches = (
        re.search(r"(?ms)^          images=\(\n(.*?)^          \)", preflight),
        re.search(r"(?ms)^            images\+=\(\n(.*?)^            \)", preflight),
        re.search(r"(?ms)^          images = \[\n(.*?)^          \]", alias),
        re.search(r"(?ms)^              images \+= \[\n(.*?)^              \]", alias),
    )
    if not all(matches):
        raise AssertionError("publish-dev profile lists are incomplete")
    shell_base, shell_extra, python_base, python_extra = matches
    shell_names = lambda match: set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", match.group(1)))
    python_names = lambda match: set(re.findall(r'"([a-z0-9-]+)"', match.group(1)))
    return (
        shell_names(shell_base),
        shell_names(shell_extra),
        python_names(python_base),
        python_names(python_extra),
    )


class ApplicationInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = inventory_entries(APPLICATION_INVENTORY)
        cls.dev = PUBLISH_DEV.read_text(encoding="utf-8")
        cls.formal = PUBLISH_IMAGES.read_text(encoding="utf-8")

    def test_inventory_declares_exactly_the_two_current_application_images(self):
        self.assertEqual(self.entries.keys(), APPLICATION_IMAGES)
        for name, expected in APPLICATION_DEFINITIONS.items():
            with self.subTest(image=name):
                entry = self.entries[name]
                self.assertEqual(entry.get("context"), expected["context"])
                self.assertEqual(entry["dockerfile"], expected["file"])
                self.assertEqual(entry["platforms"], expected["platforms"])
                self.assertTrue((ROOT / expected["context"]).is_dir())
                self.assertTrue((ROOT / expected["file"]).is_file())
                self.assertNotIn("opt-tensorrt", entry)

    def test_development_application_matrices_use_inventory_coordinates(self):
        # Application images exist only in the narrow publish-applications
        # matrix; the hardware publish-base matrix must not build them.
        for name, expected in APPLICATION_DEFINITIONS.items():
            entry = matrix_entry(self.dev, "publish-applications", name)
            self.assertIn(f"context: {expected['context']}\n", entry)
            self.assertIn(f"file: {expected['file']}\n", entry)
            self.assertIn("platforms: " + ",".join(expected["platforms"]) + "\n", entry)
            self.assertIn(
                "ghcr.io/nereidai/${{ matrix.name }}:${{ needs.preflight.outputs.immutable }}",
                job_block(self.dev, "publish-applications"),
            )
        base_names = matrix_names(self.dev, "publish-base")
        for name in APPLICATION_IMAGES:
            self.assertNotIn(name, base_names)

    def test_formal_application_matrix_retains_sources_and_multi_architecture_indexes(self):
        applications = job_block(self.formal, "applications")
        for component, expected in (("backend", "nereidforge-backend"), ("frontend", "nereidforge-frontend")):
            with self.subTest(component=component):
                match = re.search(
                    rf"(?ms)^          - component: {component}\n(.*?)(?=^          - component:|^    steps:)",
                    applications,
                )
                self.assertIsNotNone(match)
                definition = APPLICATION_DEFINITIONS[expected]
                self.assertIn(f"context: {definition['context']}\n", match.group(0))
                self.assertIn(f"file: {definition['file']}\n", match.group(0))
        self.assertIn("platforms: linux/amd64,linux/arm64", applications)
        self.assertIn("ghcr.io/nereidai/nereidforge-${{ matrix.component }}", applications)

    def test_backend_and_frontend_image_coordinates_remain_stable_in_the_single_chart(self):
        # Both application containers keep their public image names; only the
        # Helm chart/release structure was merged into charts/nereidforge.
        values = (ROOT / NF_CHARTS["nereidforge"] / "values.yaml").read_text(encoding="utf-8")
        for name in APPLICATION_IMAGES:
            dockerfile_text = (ROOT / APPLICATION_DEFINITIONS[name]["file"]).read_text(encoding="utf-8")
            with self.subTest(image=name):
                self.assertIn(f"repository: ghcr.io/nereidai/{name}", values)
                self.assertIn(f"org.opencontainers.image.title='{name}'", dockerfile_text)

    def test_backend_image_resolver_preserves_flat_coordinates_and_compatibility_aliases(self):
        path = ROOT / "python-backend" / "app" / "core" / "images.py"
        spec = importlib.util.spec_from_file_location("nf_delivery_images", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = {
            "nereidforge-backend": "nereidforge-backend",
            "nereidforge-frontend": "nereidforge-frontend",
            "trainer/torch-cpu": "trainer-torch-cpu",
            "trainer/paddle-cu": "trainer-paddle-cu",
            "train/optimizer": "train-optimizer",
            "ml-backend/ml-backend": "ml-backend",
            "ml-backend/ml-backend-custom": "ml-backend-custom",
            "runtime": "nf-runtime",
        }
        for nested, flat in expected.items():
            with self.subTest(nested=nested):
                self.assertEqual(module.nf_image(nested, "1.2.3"), f"ghcr.io/nereidai/{flat}:1.2.3")
        self.assertEqual(
            module.nf_image("prediction/jupyter-base", "py3.12"),
            "ghcr.io/nereidai/prediction-jupyter-base:py3.12",
        )

    def test_frontend_uses_the_same_coordinates_and_hardware_compatibility_map(self):
        source = (ROOT / "frontend" / "src" / "utils" / "images.ts").read_text(encoding="utf-8")
        self.assertIn("const REGISTRY = 'ghcr.io/nereidai'", source)
        for alias in (
            "'ml-backend/ml-backend': 'ml-backend'",
            "'ml-backend/ml-backend-custom': 'ml-backend-custom'",
            "runtime: 'nf-runtime'",
        ):
            self.assertIn(alias, source)
        for image in sorted(BUILDER_IMAGES & {
            "trainer-torch-cpu", "trainer-torch-cu", "trainer-torch-rocm", "trainer-torch-xpu",
            "trainer-paddle-cpu", "trainer-paddle-cu", "trainer-llm-cu", "trainer-llm-rocm",
            "trainer-llm-xpu",
        }):
            self.assertIn(image.replace("-", "/", 1) if image.startswith("trainer-") else image, source)
        self.assertIn("family === 'paddle' && (accelerator === 'amd' || accelerator === 'intel')", source)
        self.assertIn("if (vendorToken && base === `llm${vendorToken}`)", source)
        self.assertIn("wanted = `${family}${vendorToken || '-cpu'}`", source)


class ImageCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BUILDER_INVENTORY.is_file():
            raise AssertionError(
                "builders/image-inventory.json is required for application suite composition"
            )
        cls.builder_document = json.loads(BUILDER_INVENTORY.read_text(encoding="utf-8"))
        cls.builder_entries = inventory_entries(BUILDER_INVENTORY)
        cls.application_entries = inventory_entries(APPLICATION_INVENTORY)
        cls.dev = PUBLISH_DEV.read_text(encoding="utf-8")

    def test_inventories_compose_full_23_and_cpu_xpu_18_without_duplicates(self):
        self.assertEqual(self.builder_entries.keys(), BUILDER_IMAGES)
        self.assertEqual(self.application_entries.keys(), APPLICATION_IMAGES)
        self.assertTrue(self.builder_entries.keys().isdisjoint(self.application_entries.keys()))
        profiles = self.builder_document["profiles"]
        self.assertEqual(len(profiles["full"]), len(set(profiles["full"])))
        self.assertEqual(len(profiles["cpu-xpu"]), len(set(profiles["cpu-xpu"])))
        self.assertEqual(set(profiles["full"]), BUILDER_IMAGES)
        self.assertEqual(set(profiles["cpu-xpu"]), BUILDER_IMAGES - AMD_NVIDIA_IMAGES)
        self.assertEqual(len(profiles["cpu-xpu"]), 16)
        full = set(profiles["full"]) | self.application_entries.keys()
        cpu_xpu = set(profiles["cpu-xpu"]) | self.application_entries.keys()
        self.assertEqual(len(BUILDER_IMAGES), 21)
        self.assertEqual(len(full), 23)
        self.assertEqual(full, FULL_IMAGES)
        self.assertEqual(len(cpu_xpu), 18)
        self.assertEqual(cpu_xpu, CPU_XPU_IMAGES)
        self.assertEqual(FULL_IMAGES - cpu_xpu, AMD_NVIDIA_IMAGES)
        self.assertEqual(len(AMD_NVIDIA_IMAGES), 5)
        self.assertNotIn("opt-tensorrt", full)
        self.assertNotIn("opt-tensorrt", cpu_xpu)
        version_exceptions = {
            name: entry["version"]
            for name, entry in self.builder_entries.items()
            if "version" in entry
        }
        self.assertEqual(version_exceptions, {"prediction-jupyter-base": "py3.12"})

    def test_development_profile_lists_match_the_composed_inventories(self):
        shell_base, shell_extra, python_base, python_extra = workflow_image_lists(self.dev)
        for base in (shell_base, python_base):
            self.assertEqual(base, DEV_HARDWARE_IMAGES)
            self.assertEqual(len(base), 16)
            self.assertTrue(base.isdisjoint(APPLICATION_IMAGES))
        for extra in (shell_extra, python_extra):
            self.assertEqual(extra, AMD_NVIDIA_IMAGES)
            self.assertEqual(len(extra), 5)
        self.assertEqual(shell_base | shell_extra, DEV_FULL_IMAGES)
        self.assertEqual(python_base | python_extra, DEV_FULL_IMAGES)
        preflight = job_block(self.dev, "preflight")
        self.assertIn("full) expected_count=21", preflight)
        self.assertIn("cpu-xpu) expected_count=16", preflight)
        self.assertIn("backend-frontend) expected_count=2", preflight)
        self.assertIn('test "${#images[@]}" -eq "$EXPECTED_COUNT"', preflight)
        self.assertNotIn("opt-tensorrt", self.dev)

    def test_formal_release_still_requires_all_23_composed_images(self):
        release = job_block(PUBLISH_IMAGES.read_text(encoding="utf-8"), "release-assets")
        match = re.search(r"(?ms)^          expected = \{\n(.*?)^          \}", release)
        self.assertIsNotNone(match)
        formal_images = set(re.findall(r'"([a-z0-9-]+)"', match.group(1)))
        self.assertEqual(formal_images, FULL_IMAGES)
        self.assertEqual(len(formal_images), 23)
        self.assertIn("assert len(expected) == 23", release)
        self.assertIn("assert references.keys() == expected", release)
        self.assertIn(
            'expected_tag = "py3.12" if repository == "prediction-jupyter-base" else sys.argv[2]',
            release,
        )
        self.assertNotIn("opt-tensorrt", formal_images)
        charts = job_block(PUBLISH_IMAGES.read_text(encoding="utf-8"), "publish-charts")
        self.assertIn("needs: [applications, core, vendor-xpu, vendor-cu, vendor-rocm]", charts)

    def test_exported_release_coordinates_match_inventory_and_jupyter_exception(self):
        with tempfile.TemporaryDirectory(prefix="nf-delivery-manifest-") as directory:
            manifest = Path(directory) / "images.txt"
            env = os.environ.copy()
            env.update(RELEASE_TAG="2.3.4", JWT_SECRET="local-manifest-test-only")
            result = subprocess.run(
                ["bash", "scripts/05_export-images.sh", "--manifest", str(manifest)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            references = {}
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if line.startswith("ghcr.io/nereidai/"):
                    repository, tag = line.rsplit(":", 1)
                    name = repository.removeprefix("ghcr.io/nereidai/")
                    self.assertNotIn(name, references, f"duplicate release coordinate: {name}")
                    references[name] = tag
            self.assertEqual(references.keys(), FULL_IMAGES)
            for name, tag in references.items():
                expected = "py3.12" if name == "prediction-jupyter-base" else "2.3.4"
                self.assertEqual(tag, expected, name)
            self.assertNotIn("opt-tensorrt", references)


class FormalReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = PUBLISH_IMAGES.read_text(encoding="utf-8")

    def test_release_matrix_has_exactly_three_nf_charts(self):
        charts = job_block(self.text, "publish-charts")
        names = matrix_names(self.text, "publish-charts")
        self.assertEqual(set(names), NF_CHARTS.keys())
        self.assertEqual(len(names), 3)
        self.assertEqual(charts.count("path: charts/"), 3)
        self.assertNotIn("deploy/pre-nereidforge", charts)
        self.assertNotIn("version_mode: upstream", charts)

        release = job_block(self.text, "release-assets")
        self.assertIn('test "${#charts[@]}" -eq 3', release)
        self.assertIn("grep -c '^oci://' \"$manifest\")\" -eq 3", release)

    def test_release_has_no_pre_or_source_installer_assets(self):
        release = job_block(self.text, "release-assets")
        for forbidden in (
            "scripts/04_package-installer.sh",
            "nereidforge-installer-",
            "deploy/pre-nereidforge/charts",
            "charts/postgresql:",
            "charts/cvat:",
        ):
            self.assertNotIn(forbidden, release)
        self.assertIn('"${release_dir}/nereidforge-images-${VERSION}.txt"', release)
        self.assertIn('"${release_dir}/nereidforge-charts-${VERSION}.txt"', release)
        self.assertIn("assert len(expected) == 23", release)
        self.assertIn('test "${#assets[@]}" -eq 2', release)


class DevelopmentGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = PUBLISH_DEV.read_text(encoding="utf-8")

    def test_manual_main_dispatch_and_immutable_build_identities_are_unchanged(self):
        trigger = re.search(r"(?ms)^on:\n(.*?)(?=^permissions:\n)", self.text)
        self.assertIsNotNone(trigger)
        self.assertIn("  workflow_dispatch:\n", trigger.group(1))
        self.assertIn("      image_profile:\n", trigger.group(1))
        self.assertIn("        default: full\n", trigger.group(1))
        self.assertIn("          - cpu-xpu\n", trigger.group(1))
        self.assertIn("          - backend-frontend\n", trigger.group(1))
        for forbidden in ("  push:\n", "  pull_request:\n", "  merge_group:\n"):
            self.assertNotIn(forbidden, trigger.group(1))
        self.assertNotIn("path-scope:", self.text)
        preflight = job_block(self.text, "preflight")
        self.assertIn('test "$GITHUB_EVENT_NAME" = workflow_dispatch', preflight)
        self.assertIn('test "$GITHUB_REF_TYPE" = branch', preflight)
        self.assertIn('test "$GITHUB_REF_NAME" = main', preflight)
        for job_name in ("publish-base", "publish-dependent"):
            job = job_block(self.text, job_name)
            self.assertIn(
                "tags: ghcr.io/nereidai/${{ matrix.name }}:${{ needs.preflight.outputs.immutable }}",
                job,
            )
            self.assertNotIn("${{ needs.preflight.outputs.floating }}", job)

        script = run_block(self.text, "id: ver")
        with tempfile.TemporaryDirectory(prefix="nf-dev-manual-") as directory:
            output = Path(directory) / "output"
            env = os.environ.copy()
            env.update(
                GITHUB_EVENT_NAME="workflow_dispatch",
                GITHUB_REF_TYPE="branch",
                GITHUB_REF_NAME="main",
                GITHUB_SHA="0123456789abcdef0123456789abcdef01234567",
                GITHUB_OUTPUT=str(output),
                IMAGE_PROFILE="full",
            )
            result = subprocess.run(
                ["bash", "-c", script], env=env, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            values = dict(line.split("=", 1) for line in output.read_text().splitlines())
            self.assertEqual(values["immutable"], "0.0.0-dev-0123456")
            self.assertEqual(values["floating"], "0.0.0-dev")
            self.assertEqual(values["image_profile"], "full")
            self.assertEqual(values["expected_count"], "21")

            cpu_output = Path(directory) / "cpu-output"
            env.update(IMAGE_PROFILE="cpu-xpu", GITHUB_OUTPUT=str(cpu_output))
            result = subprocess.run(
                ["bash", "-c", script], env=env, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            cpu_values = dict(line.split("=", 1) for line in cpu_output.read_text().splitlines())
            self.assertEqual(cpu_values["image_profile"], "cpu-xpu")
            self.assertEqual(cpu_values["expected_count"], "16")

            application_output = Path(directory) / "application-output"
            env.update(IMAGE_PROFILE="backend-frontend", GITHUB_OUTPUT=str(application_output))
            result = subprocess.run(
                ["bash", "-c", script], env=env, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            application_values = dict(
                line.split("=", 1) for line in application_output.read_text().splitlines()
            )
            self.assertEqual(application_values["image_profile"], "backend-frontend")
            self.assertEqual(application_values["expected_count"], "2")
            env["IMAGE_PROFILE"] = "full"

            invalid = (
                ("push", "branch", "main"),
                ("workflow_dispatch", "branch", "topic"),
                ("workflow_dispatch", "tag", "0.0.0-dev"),
            )
            for index, (event, ref_type, ref_name) in enumerate(invalid):
                with self.subTest(event=event, ref_type=ref_type, ref_name=ref_name):
                    env.update(
                        GITHUB_EVENT_NAME=event,
                        GITHUB_REF_TYPE=ref_type,
                        GITHUB_REF_NAME=ref_name,
                        GITHUB_OUTPUT=str(Path(directory) / f"invalid-{index}"),
                    )
                    result = subprocess.run(
                        ["bash", "-c", script], env=env, text=True, capture_output=True, check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)

    def test_immutable_reuse_and_collision_checks_fail_closed(self):
        preflight = job_block(self.text, "preflight")
        self.assertIn("source scripts/lib/nf_publication.sh", preflight)
        self.assertIn("EXPECTED_REVISION: ${{ github.sha }}", preflight)
        self.assertIn('if pre_oras_resolve "$reference"; then', preflight)
        self.assertIn('_pre_oras manifest fetch "${reference}@${PRE_ORAS_DIGEST}"', preflight)
        self.assertIn("actual_revision != revision or actual_version != immutable", preflight)
        self.assertIn("immutable collision", preflight)
        self.assertIn('if [[ "$status" -eq 10 ]]; then', preflight)
        self.assertIn("refusing to treat a registry error as absence", preflight)
        self.assertLess(self.text.index("  preflight:\n"), self.text.index("  publish-base:\n"))

    def test_anonymous_verification_precedes_every_selected_floating_alias(self):
        alias = job_block(self.text, "alias-images")
        self.assertIn(
            "needs: [preflight, publish-applications, publish-base, publish-dependent]",
            alias,
        )
        self.assertIn("always()", alias)
        self.assertIn("needs.publish-applications.result == 'success'", alias)
        self.assertIn("needs.publish-base.result == 'success'", alias)
        self.assertIn("needs.publish-dependent.result == 'success'", alias)
        immutable_marker = "- name: Anonymously verify every immutable image and record its digest"
        alias_marker = "- name: Alias floating tags only to anonymously verified immutable digests"
        floating_marker = "- name: Anonymously verify selected floating aliases against immutable digests"
        self.assertLess(alias.index(immutable_marker), alias.index(alias_marker))
        self.assertLess(alias.index(alias_marker), alias.index(floating_marker))

        immutable = step_block(alias, immutable_marker)
        self.assertIn("runner.temp }}/anonymous/oras-config.json", immutable)
        self.assertIn('digest = oras(["resolve"], reference)', immutable)
        self.assertIn('manifest = json.loads(oras(["manifest", "fetch"], reference))', immutable)
        self.assertIn("annotations.get(\"org.opencontainers.image.revision\") == revision", immutable)
        self.assertIn("annotations.get(\"org.opencontainers.image.version\") == immutable", immutable)
        self.assertIn("assert len(images) == expected_count", immutable)
        self.assertIn('("linux", "amd64") in platforms and ("linux", "arm64") in platforms', immutable)
        self.assertIn("check=True", immutable)
        self.assertIn("DIGEST_FILE", immutable)
        for forbidden in ("GHCR_TOKEN", "secrets.", "oras login", "registry login"):
            self.assertNotIn(forbidden, immutable)

        aliasing = step_block(alias, alias_marker)
        self.assertIn('"${repository}@${digest}" "$FLOATING"', aliasing)
        floating = step_block(alias, floating_marker)
        self.assertIn("assert len(digests) == int(os.environ[\"EXPECTED_COUNT\"])", floating)
        self.assertIn("assert floating_digest == immutable_digest", floating)
        self.assertIn("runner.temp }}/anonymous/oras-config.json", floating)
        for forbidden in ("GHCR_TOKEN", "secrets.", "oras login", "registry login"):
            self.assertNotIn(forbidden, floating)

    def test_suite_charts_wait_for_suite_images_and_application_profile_excludes_them(self):
        charts = job_block(self.text, "publish-charts")
        self.assertIn("if: inputs.image_profile != 'backend-frontend'", charts)
        names = matrix_names(self.text, "publish-charts")
        self.assertEqual(len(names), 3)
        self.assertEqual(set(names), NF_CHARTS.keys())
        self.assertIn("needs: alias-images", charts)
        self.assertNotIn("deploy/pre-nereidforge", charts)
        package = PACKAGE_CHARTS.read_text(encoding="utf-8")
        entries = re.findall(
            r'^    "(charts/[^:]+):(nereidforge[^:]*):version"$', package, re.MULTILINE,
        )
        self.assertEqual(len(entries), 3)
        self.assertEqual({name for _, name in entries}, NF_CHARTS.keys())
        self.assertEqual({path for path, _ in entries}, set(NF_CHARTS.values()))
        self.assertNotIn("deploy/pre-nereidforge/charts", package)
        self.assertIn('[[ "${#CHARTS[@]}" -eq 3 ]]', package)
        self.assertIn('[[ "${#package_files[@]}" -eq 3 ]]', package)

    def test_nf_delivery_has_no_pre_source_dependency(self):
        self.assertNotIn("deploy/pre-nereidforge/scripts", self.text)
        self.assertEqual(self.text.count("source scripts/lib/nf_publication.sh"), 2)


class CacheClassificationTests(unittest.TestCase):
    def execute_scope(self, paths, outputs):
        text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        script = run_block(text, "id: scope")
        with tempfile.TemporaryDirectory(prefix="nf-path-scope-") as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "nf-tests@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "NF Tests"], check=True)
            (repo / ".baseline").write_text("base\n")
            subprocess.run(["git", "-C", str(repo), "add", ".baseline"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "base"], check=True)
            for relative in paths:
                target = repo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("changed\n")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "change"], check=True)
            output = Path(directory) / "github-output"
            env = os.environ.copy()
            env["GITHUB_OUTPUT"] = str(output)
            result = subprocess.run(
                ["bash", "-c", script],
                cwd=repo,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            values = dict(
                line.split("=", 1)
                for line in output.read_text().splitlines()
                if "=" in line
            )
            self.assertEqual({key: values[key] for key in outputs}, outputs)

    def test_pre_only_paths_do_not_warm_nf_images(self):
        paths = (
            "deploy/pre-nereidforge/values/cvat.yaml",
            "deploy/pre-nereidforge/scripts/package.sh",
            ".github/workflows/pre-publish.yml",
        )
        self.execute_scope(paths, {"applications": "false", "core": "false", "vendor": "false"})

    def test_nf_cache_classifier_selects_expected_scopes_and_fails_safe(self):
        self.execute_scope(
            ("python-backend/app/main.py",),
            {"applications": "true", "core": "false", "vendor": "false"},
        )
        self.execute_scope(
            (".github/workflows/publish-images.yml",),
            {"applications": "true", "core": "true", "vendor": "true"},
        )
        self.execute_scope(
            ("unknown/new-input.txt",),
            {"applications": "true", "core": "true", "vendor": "true"},
        )


@unittest.skipUnless(shutil.which("helm"), "helm is required for offline chart checks")
class OfflineChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.version = "0.0.0-dev-abcdef1"
        cls.temporary = tempfile.TemporaryDirectory(prefix="nf-delivery-charts-")
        cls.output = Path(cls.temporary.name) / "packages"
        env = os.environ.copy()
        env.update(
            CHART_VERSION=cls.version,
            CHARTS_OUTPUT_DIR=str(cls.output),
            CHARTS_ARCHIVE=str(Path(cls.temporary.name) / "charts.tar.gz"),
            SOURCE_DATE_EPOCH="0",
        )
        result = subprocess.run(
            ["bash", "scripts/03_package-charts.sh"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)
        cls.packages = {
            path.name: path
            for path in cls.output.glob("*.tgz")
        }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def values_args(self, chart_name):
        # The consolidated chart requires the frontend ingress host; the
        # required value now lives under the frontend: block.
        return ["--set", "frontend.ingress.host=forge.example.test"] if chart_name == "nereidforge" else []

    def test_packager_produces_exactly_three_nf_charts_with_immutable_metadata(self):
        expected = {f"{name}-{self.version}.tgz" for name in NF_CHARTS}
        self.assertEqual(self.packages.keys(), expected)
        for name in NF_CHARTS:
            with self.subTest(chart=name):
                metadata = subprocess.run(
                    ["helm", "show", "chart", str(self.packages[f"{name}-{self.version}.tgz"])],
                    text=True,
                    capture_output=True,
                    check=True,
                ).stdout
                self.assertRegex(metadata, rf"(?m)^name: {re.escape(name)}$")
                self.assertRegex(metadata, rf"(?m)^version: {re.escape(self.version)}$")
                self.assertRegex(metadata, rf"(?m)^appVersion: [\"']?{re.escape(self.version)}[\"']?$")

    def test_release_tag_fallback_preserves_chart_sources_and_archive(self):
        with tempfile.TemporaryDirectory(prefix="nf-release-charts-") as directory:
            base = Path(directory)
            output = base / "packages"
            archive = base / "charts.tar.gz"
            sources = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (ROOT / "charts").glob("*/Chart.yaml")
            }
            env = os.environ.copy()
            env.pop("CHART_VERSION", None)
            env.update(
                RELEASE_TAG="1.2.3",
                CHARTS_OUTPUT_DIR=str(output),
                CHARTS_ARCHIVE=str(archive),
                CHARTS_ARCHIVE_LAYOUT="packages",
            )
            result = subprocess.run(
                ["bash", "scripts/03_package-charts.sh"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            packages = sorted(output.glob("*.tgz"))
            self.assertEqual(len(packages), 3)
            self.assertEqual(
                {path.name for path in packages},
                {f"{name}-1.2.3.tgz" for name in NF_CHARTS},
            )
            for name in NF_CHARTS:
                with self.subTest(chart=name):
                    metadata = subprocess.run(
                        ["helm", "show", "chart", str(output / f"{name}-1.2.3.tgz")],
                        text=True,
                        capture_output=True,
                        check=True,
                    ).stdout
                    self.assertRegex(metadata, rf"(?m)^name: {re.escape(name)}$")
                    self.assertRegex(metadata, r"(?m)^version: 1\.2\.3$")
                    self.assertRegex(metadata, r"(?m)^appVersion: [\"']?1\.2\.3[\"']?$")
            for path, digest in sources.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            with tarfile.open(archive, "r:gz") as bundle:
                names = bundle.getnames()
                self.assertEqual(len([name for name in names if name.endswith(".tgz")]), 3)
                self.assertFalse(any(name.startswith("deploy/") for name in names))

    def test_lint_and_template_are_offline_and_suite_image_tags_align(self):
        for name, relative in NF_CHARTS.items():
            with self.subTest(chart=name):
                values_args = self.values_args(name)
                subprocess.run(
                    ["helm", "lint", str(ROOT / relative), *values_args],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                rendered = subprocess.run(
                    [
                        "helm", "template", name,
                        str(self.packages[f"{name}-{self.version}.tgz"]),
                        *values_args,
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=True,
                ).stdout
                self.assertTrue(rendered.strip())
                references = set(re.findall(
                    r"ghcr\.io/nereidai/[A-Za-z0-9][A-Za-z0-9._/-]*:[A-Za-z0-9_][A-Za-z0-9._-]*",
                    rendered,
                ))
                mismatches = sorted(ref for ref in references if not ref.endswith(":" + self.version))
                self.assertEqual(mismatches, [])
                self.assertNotIn("opt-tensorrt", rendered)


class VerificationEntryPointTests(unittest.TestCase):
    def test_nf_verify_is_offline_and_does_not_run_pre_or_builder_unit_tests(self):
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        match = re.search(r"(?ms)^nf-verify:\n(.*?)(?=^[a-z][a-z0-9-]*:|\Z)", makefile)
        self.assertIsNotNone(match)
        target = match.group(0)
        self.assertIn("test_nf_delivery", target)
        self.assertIn("test_nf_publication", target)
        self.assertIn("test_nf_legacy_compatibility", target)
        self.assertIn("test_online_installer_verification", target)
        self.assertNotIn("deploy/pre-nereidforge", target)
        self.assertNotIn("builders/tests", target)
        self.assertNotIn("cd builders", target)
        self.assertNotIn("docker build", target)
        self.assertNotIn("npm ci", target)
        self.assertNotIn("pip install", target)

    def test_pull_request_verification_never_publishes(self):
        workflow = (ROOT / ".github" / "workflows" / "nf-verify.yml").read_text(encoding="utf-8")
        self.assertIn("pull_request:", workflow)
        self.assertIn("builders/image-inventory.json", workflow)
        self.assertIn("configs/application-image-inventory.json", workflow)
        self.assertIn("scripts/tests/test_nf_legacy_compatibility.py", workflow)
        self.assertIn("scripts/60_verify/nf-check.sh", workflow)
        self.assertIn("scripts/tests/test_online_installer_verification.py", workflow)
        self.assertIn("'kfp-pipelines/**'", workflow)
        self.assertIn("'!kfp-pipelines/docker/**'", workflow)
        self.assertIn(".github/workflows/ci.yml", workflow)
        self.assertIn("make nf-verify", workflow)
        self.assertNotIn("packages: write", workflow)
        self.assertNotIn("workflow_dispatch:", workflow)
        for forbidden in ("docker/build-push-action", "oras push", "helm push", "gh release"):
            self.assertNotIn(forbidden, workflow)


if __name__ == "__main__":
    unittest.main(verbosity=2)
