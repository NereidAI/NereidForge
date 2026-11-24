#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


PRE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = PRE_ROOT.parents[1]
WORKFLOWS = REPOSITORY / ".github" / "workflows"
PRE_CHARTS = {"postgresql", "valkey", "seaweedfs", "labelstudio", "cvat"}
NF_CHARTS = {
    "nereidforge-ml-backend",
    "nereidforge-backend",
    "nereidforge-frontend",
    "nereidforge-tenant-check",
}
NF_IMAGES = {
    "inference-model-server-openvino-xpu", "ml-backend", "ml-backend-custom",
    "nereidforge-backend", "nereidforge-frontend", "nf-runtime",
    "prediction-jupyter-base", "train-evaluator", "train-evaluator-xpu",
    "train-optimizer", "trainer-convert", "trainer-convert-pp",
    "trainer-convert-tf", "trainer-llm-cu", "trainer-llm-rocm",
    "trainer-llm-xpu", "trainer-paddle-cpu", "trainer-paddle-cu",
    "trainer-torch-cpu", "trainer-torch-cu", "trainer-torch-rocm",
    "trainer-torch-xpu", "trainer-xpu-base",
}


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


def step_block(text, marker):
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines) if marker in line]
    if not matches:
        raise AssertionError(f"expected a step marker {marker!r}")
    blocks = []
    for start in matches:
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if lines[index].startswith("      - name:")
            ),
            len(lines),
        )
        blocks.append("\n".join(lines[start:end]))
    return "\n".join(blocks)


class PreChartOwnershipTests(unittest.TestCase):
    def test_pre_packager_has_exactly_five_supporting_charts(self):
        script = (PRE_ROOT / "scripts" / "package-charts.sh").read_text()
        self.assertIn("CHARTS=(postgresql valkey seaweedfs labelstudio cvat)", script)
        self.assertIn('[[ "${#CHARTS[@]}" -eq 5 ]]', script)
        self.assertIn('[[ "${#package_files[@]}" -eq 5 ]]', script)
        self.assertNotIn("charts/nereidforge-", script)
        self.assertNotIn("infra/charts/kubeflow", script)
        self.assertIn('package_args+=(--version "$CHART_VERSION_OVERRIDE")', script)
        self.assertNotIn("--app-version", script)

    def test_pre_makefile_exposes_local_only_chart_packaging(self):
        makefile = (PRE_ROOT / "Makefile").read_text()
        self.assertIn("helm-package:", makefile)
        self.assertIn('bash "$(PRE_ROOT)/scripts/package-charts.sh"', makefile)
        self.assertNotIn("helm push", makefile)
        self.assertNotIn("oras push", makefile)


@unittest.skipUnless(WORKFLOWS.is_dir(), "requires a combined NereidForge checkout")
class PublicationWorkflowTests(unittest.TestCase):
    def read(self, name):
        return (WORKFLOWS / name).read_text()

    def execute_scope(self, workflow_name, paths, outputs):
        text = self.read(workflow_name)
        script = run_block(text, "id: scope")
        with tempfile.TemporaryDirectory(prefix="nf-path-scope-") as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "pre-tests@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Pre Tests"], check=True)
            (repo / ".baseline").write_text("base\n")
            subprocess.run(["git", "-C", str(repo), "add", ".baseline"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "base"], check=True)
            parent = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            for relative in paths:
                target = repo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("changed\n")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "change"], check=True)
            script = script.replace("${{ github.event.before }}", parent)
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
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            values = dict(
                line.split("=", 1)
                for line in output.read_text().splitlines()
                if "=" in line
            )
            self.assertEqual({key: values[key] for key in outputs}, outputs)

    def test_nf_release_matrix_has_exactly_four_nf_charts(self):
        text = self.read("publish-images.yml")
        matrix = text.split("  publish-charts:\n", 1)[1].split("  release-assets:\n", 1)[0]
        names = re.findall(r"^          - name: (nereidforge-[a-z-]+)$", matrix, re.MULTILINE)
        self.assertEqual(set(names), NF_CHARTS)
        self.assertEqual(len(names), 4)
        self.assertEqual(matrix.count("path: charts/"), 4)
        self.assertNotIn("deploy/pre-nereidforge", matrix)
        self.assertNotIn("version_mode: upstream", matrix)
        self.assertIn('test "${#charts[@]}" -eq 4', text)
        self.assertIn("grep -c '^oci://' \"$manifest\")\" -eq 4", text)

    def test_nf_release_has_no_pre_or_source_installer_assets(self):
        text = self.read("publish-images.yml")
        release = text.split("  release-assets:\n", 1)[1]
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
        self.assertIn('test "${#assets[@]}" -eq 4', self.read("pre-publish.yml"))

    def test_nf_manifest_export_has_exactly_23_release_images(self):
        self.assertEqual(len(NF_IMAGES), 23)
        with tempfile.TemporaryDirectory(prefix="nf-image-manifest-") as directory:
            manifest = Path(directory) / "images.txt"
            env = os.environ.copy()
            env.update(RELEASE_TAG="2.3.4", JWT_SECRET="local-manifest-test-only")
            result = subprocess.run(
                ["bash", "scripts/05_export-images.sh", "--manifest", str(manifest)],
                cwd=REPOSITORY,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            references = {}
            for line in manifest.read_text().splitlines():
                if line.startswith("ghcr.io/nereidai/"):
                    repository, tag = line.rsplit(":", 1)
                    references[repository.removeprefix("ghcr.io/nereidai/")] = tag
            self.assertEqual(references.keys(), NF_IMAGES)
            for repository, tag in references.items():
                expected = "py3.12" if repository == "prediction-jupyter-base" else "2.3.4"
                self.assertEqual(tag, expected, repository)

    def test_root_packager_has_exactly_four_nf_charts(self):
        script = (REPOSITORY / "scripts" / "03_package-charts.sh").read_text()
        entries = re.findall(r'^    "(charts/[^:]+):(nereidforge-[^:]+):version"$', script, re.MULTILINE)
        self.assertEqual({name for _, name in entries}, NF_CHARTS)
        self.assertEqual(len(entries), 4)
        self.assertNotIn("deploy/pre-nereidforge/charts", script)
        self.assertIn('[[ "${#CHARTS[@]}" -eq 4 ]]', script)
        self.assertIn('[[ "${#package_files[@]}" -eq 4 ]]', script)

    def test_pre_formal_trigger_and_version_validation(self):
        text = self.read("pre-publish.yml")
        self.assertIn("tags:\n      - 'pre-nereidforge/v*'", text)
        self.assertNotIn("workflow_dispatch", text)
        self.assertNotIn("uses: NereidAI/.github", text)
        self.assertNotIn("PRE_CHARTS_VERSION", text)
        script = run_block(text, "id: ver")
        valid = {
            "pre-nereidforge/v0.0.1": "0.0.1",
            "pre-nereidforge/v1.2.3-rc.1": "1.2.3-rc.1",
        }
        invalid = (
            "v0.0.1",
            "pre-nereidforge/v1.2",
            "pre-nereidforge/v0.0.1+build",
            "pre-nereidforge/v0.0.1-01",
        )
        with tempfile.TemporaryDirectory(prefix="pre-formal-version-") as directory:
            for tag, expected in valid.items():
                with self.subTest(tag=tag):
                    output = Path(directory) / "output"
                    output.write_text("")
                    env = os.environ.copy()
                    env.update(GITHUB_REF_NAME=tag, GITHUB_OUTPUT=str(output))
                    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_text().strip(), f"version={expected}")
            for tag in invalid:
                with self.subTest(tag=tag):
                    env = os.environ.copy()
                    env.update(GITHUB_REF_NAME=tag, GITHUB_OUTPUT=str(Path(directory) / "invalid-output"))
                    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
                    self.assertNotEqual(result.returncode, 0, tag)

    def test_pre_development_trigger_and_version_identity(self):
        text = self.read("pre-publish-dev.yml")
        self.assertIn("branches: [main]", text)
        for path in (
            "deploy/pre-nereidforge/**",
            ".github/workflows/pre-verify.yml",
            ".github/workflows/pre-publish.yml",
            ".github/workflows/pre-publish-dev.yml",
        ):
            self.assertIn(path, text)
        self.assertNotIn("gh release", text)
        self.assertNotIn("workflow_dispatch", text)
        self.assertIn("PRE_CHARTS_VERSION: ${{ steps.ver.outputs.immutable }}", text)
        self.assertIn("pre_publish_installer", text)
        self.assertNotIn('"ghcr.io/nereidai/pre-nereidforge-installer:${IMMUTABLE},${FLOATING}"', text)
        script = run_block(text, "id: ver")
        with tempfile.TemporaryDirectory(prefix="pre-dev-version-") as directory:
            output = Path(directory) / "output"
            env = os.environ.copy()
            env.update(
                GITHUB_EVENT_NAME="push",
                GITHUB_REF_TYPE="branch",
                GITHUB_REF_NAME="main",
                GITHUB_SHA="0123456789abcdef0123456789abcdef01234567",
                GITHUB_OUTPUT=str(output),
            )
            result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            values = dict(line.split("=", 1) for line in output.read_text().splitlines())
            self.assertEqual(values["immutable"], "0.0.0-dev-0123456")
            self.assertEqual(values["floating"], "0.0.0-dev")

    def test_pre_workflows_publish_no_nf_artifacts(self):
        for name in ("pre-publish.yml", "pre-publish-dev.yml"):
            with self.subTest(workflow=name):
                text = self.read(name)
                self.assertEqual(text.count("charts=(postgresql valkey seaweedfs labelstudio cvat)"), 2)
                for forbidden in (
                    "docker/build-push-action",
                    "uses: NereidAI/.github",
                    "charts/nereidforge-backend",
                    "charts/ml-backend",
                    "ghcr.io/nereidai/nereidforge-backend:",
                    "ghcr.io/nereidai/nereidforge-frontend:",
                ):
                    self.assertNotIn(forbidden, text)

    def test_anonymous_verification_uses_isolated_unauthenticated_configuration(self):
        for name in ("pre-publish.yml", "pre-publish-dev.yml"):
            with self.subTest(workflow=name):
                text = self.read(name)
                self.assertIn("HELM_REGISTRY_CONFIG: ${{ runner.temp }}/anonymous/helm/registry-config.json", text)
                self.assertIn("ORAS_REGISTRY_CONFIG: ${{ runner.temp }}/anonymous/oras-config.json", text)
                self.assertIn("helm pull", text)
                self.assertIn("helm template", text)
                self.assertIn("oras resolve --registry-config", text)
                anonymous = step_block(text, "- name: Anonymously")
                self.assertNotIn("GHCR_TOKEN", anonymous)
                self.assertNotIn("secrets.", anonymous)
                self.assertNotIn("registry login", anonymous)
                self.assertNotIn("oras login", anonymous)

    def test_nf_development_publication_requires_manual_main_dispatch(self):
        text = self.read("publish-dev.yml")
        self.assertIn("on:\n  workflow_dispatch:\n    inputs:\n      image_profile:", text)
        self.assertIn("        default: full\n", text)
        self.assertIn("          - cpu-xpu\n", text)
        for trigger in ("  push:\n", "  pull_request:\n", "  merge_group:\n"):
            self.assertNotIn(trigger, text)
        self.assertNotIn("path-scope:", text)

        script = run_block(text, "id: ver")
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
            result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            values = dict(line.split("=", 1) for line in output.read_text().splitlines())
            self.assertEqual(values["immutable"], "0.0.0-dev-0123456")
            self.assertEqual(values["floating"], "0.0.0-dev")
            self.assertEqual(values["image_profile"], "full")
            self.assertEqual(values["expected_count"], "23")

            cpu_output = Path(directory) / "cpu-output"
            env.update(IMAGE_PROFILE="cpu-xpu", GITHUB_OUTPUT=str(cpu_output))
            result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            cpu_values = dict(line.split("=", 1) for line in cpu_output.read_text().splitlines())
            self.assertEqual(cpu_values["image_profile"], "cpu-xpu")
            self.assertEqual(cpu_values["expected_count"], "18")
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
                    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
                    self.assertNotEqual(result.returncode, 0)

    def test_pre_only_paths_do_not_warm_nf_images(self):
        paths = (
            "deploy/pre-nereidforge/values/cvat.yaml",
            "deploy/pre-nereidforge/scripts/package.sh",
            ".github/workflows/pre-publish.yml",
        )
        self.execute_scope("ci.yml", paths, {"applications": "false", "core": "false", "vendor": "false"})

    def test_nf_cache_classifier_selects_expected_scopes_and_fails_safe(self):
        self.execute_scope(
            "ci.yml",
            ("python-backend/app/main.py",),
            {"applications": "true", "core": "false", "vendor": "false"},
        )
        self.execute_scope(
            "ci.yml",
            (".github/workflows/publish-images.yml",),
            {"applications": "true", "core": "true", "vendor": "true"},
        )
        self.execute_scope(
            "ci.yml",
            ("unknown/new-input.txt",),
            {"applications": "true", "core": "true", "vendor": "true"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
