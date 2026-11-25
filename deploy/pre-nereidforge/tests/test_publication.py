#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PRE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = PRE_ROOT.parents[1]
WORKFLOWS = REPOSITORY / ".github" / "workflows"
PRE_CHARTS = {"postgresql", "valkey", "seaweedfs", "labelstudio", "cvat"}


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

    def test_pre_formal_trigger_and_version_validation(self):
        text = self.read("pre-publish.yml")
        self.assertIn('test "${#assets[@]}" -eq 4', text)
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
                    "charts/nereidforge",
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

    def test_postgresql_render_never_needs_a_bootstrap_password_placeholder(self):
        for name in ("pre-publish.yml", "pre-publish-dev.yml"):
            with self.subTest(workflow=name):
                text = self.read(name)
                self.assertIn("values_args=(-f deploy/pre-nereidforge/charts/postgresql/values-prod.yaml)", text)
                self.assertNotIn("bootstrap.password", text)
                self.assertNotIn("ci-render-only-placeholder", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
