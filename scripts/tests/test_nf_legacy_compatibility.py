#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Compatibility contracts for legacy combined installers and root versioning.

These tests intentionally cross module boundaries while the legacy adapters remain
supported. They are not part of the standalone Pre-NereidForge verification gate.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PRE_ROOT = ROOT / "deploy" / "pre-nereidforge"
PRE_CONFIG = PRE_ROOT / "scripts" / "lib" / "config.sh"
ROOT_COMMON = ROOT / "scripts" / "lib" / "common.sh"
HELM = shutil.which("helm")
PRE_VERSIONS = {
    "cvat": ("2.73.0", "2.73.0"),
    "labelstudio": ("1.23.0", "1.23.0"),
    "postgresql": ("0.29.0", "16.4"),
    "seaweedfs": ("3.60.0", "4.34"),
    "valkey": ("1.0.0", "8.0"),
}


@unittest.skipUnless(HELM, "helm is required for legacy offline chart checks")
class LegacyChartCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="nf-legacy-charts-")
        self.addCleanup(self.tempdir.cleanup)
        self.base = Path(self.tempdir.name)
        self.wrapper = self.base / "helm-no-network"
        self.wrapper.write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            "if [[ \"$1 $2\" == 'repo add' ]]; then\n"
            "  exit 0\n"
            "fi\n"
            "if [[ \"$1 $2\" == 'dependency build' ]]; then\n"
            "  # Metadata-only CNPG placeholder: package tests never download dependencies.\n"
            "  if grep -q 'name: cloudnative-pg' \"$3/Chart.yaml\"; then\n"
            "    mkdir -p \"$3/charts/cloudnative-pg\"\n"
            "    printf 'apiVersion: v2\\nname: cloudnative-pg\\nversion: 0.29.0\\n' > \"$3/charts/cloudnative-pg/Chart.yaml\"\n"
            "  fi\n"
            "  exit 0\n"
            "fi\n"
            f"exec {HELM} \"$@\"\n"
        )
        self.wrapper.chmod(0o755)

    def test_legacy_offline_bundle_composes_independent_chart_trees(self):
        offline = self.base / "offline"
        env = os.environ.copy()
        env.update(
            HELM=str(self.wrapper),
            OFFLINE_DIR=str(offline),
            RELEASE_TAG="1.2.3",
            JWT_SECRET="local-offline-test-only",
        )
        result = subprocess.run(
            ["bash", "scripts/01_prepare-offline/package-charts.sh"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        archive = offline / "charts" / "charts-offline.tar.gz"
        self.assertTrue(archive.is_file())
        with tarfile.open(archive, "r:gz") as bundle:
            names = bundle.getnames()
        nf_definitions = [
            name for name in names
            if name.startswith("charts/") and name.count("/") == 2 and name.endswith("/Chart.yaml")
        ]
        pre_definitions = [
            name for name in names
            if name.startswith("deploy/pre-nereidforge/charts/")
            and name.count("/") == 4
            and name.endswith("/Chart.yaml")
        ]
        self.assertEqual(len(nf_definitions), 4)
        self.assertEqual(len(pre_definitions), 5)
        self.assertEqual({name.split("/")[3] for name in pre_definitions}, set(PRE_VERSIONS))
        self.assertFalse(any(name.startswith("infra/charts/kubeflow") for name in names))


class LegacyPreInterfaceCompatibilityTests(unittest.TestCase):
    def source_pre_config(self, extra_env, script):
        env = os.environ.copy()
        scrub = [
            "PRE_RELEASE_TAG", "SOURCE_COMMIT", "RELEASE_TAG", "PRE_ENV_FILE",
            "NF_ENV_FILE", "PRE_HANDOFF_DIR", "NF_HANDOFF_DIR", "JWT_SECRET",
            "NEREIDFORGE_AUTH_JWT_SECRET", "NF_BACKEND_IMAGE", "NF_FRONTEND_IMAGE",
            "NF_TRAINER_IMAGE", "IMG_ML_BACKEND", "IMG_XPU_BASE", "VALUES_FILE",
            "SECRETS_FILE", "SECRETS_TEMPLATE", "APP_CHARTS_DIR",
        ]
        for name in scrub:
            env.pop(name, None)
        env["PRE_RELEASE_TAG"] = "1.2.3"
        env["RELEASE_TAG"] = "9.9.9"
        env["DEPLOYMENT_ROOT"] = tempfile.mkdtemp(prefix="nf-legacy-pre-interface-")
        self.addCleanup(shutil.rmtree, env["DEPLOYMENT_ROOT"], True)
        env.update(extra_env)
        result = subprocess.run(
            ["bash", "-c", f'set -e; source "$1"; {script}', "_", str(PRE_CONFIG)],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_full_stack_site_and_handoff_paths_remain_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            values = Path(directory) / "values"
            handoff = Path(directory) / "handoff"
            site = Path(directory) / "root-site.env"
            offline = Path(directory) / "offline"
            site.write_text("BASE_DOMAIN=full-stack.example.test\n")
            result = self.source_pre_config(
                {
                    "PRE_ENV_FILE": "",
                    "NF_ENV_FILE": str(site),
                    "DEPLOY_VALUES_DIR": str(values),
                    "PRE_HANDOFF_DIR": "",
                    "NF_HANDOFF_DIR": str(handoff),
                    "OFFLINE_DIR": str(offline),
                },
                'export_all; printf "%s\\n%s\\n%s\\n%s\\n%s" "$PRE_ENV_FILE" "$NF_ENV_FILE" "$PRE_HANDOFF_DIR" "$NF_HANDOFF_DIR" "$OFFLINE_DIR"',
            )
        self.assertEqual(
            result.stdout.splitlines(),
            [str(site), str(site), str(handoff), str(handoff), str(offline)],
        )

    def test_root_full_stack_entry_points_still_address_pre_scripts(self):
        if not (ROOT / "scripts" / "00_deploy-all.sh").is_file():
            self.skipTest("full-stack compatibility is available only in a combined checkout")
        deploy_all = (ROOT / "scripts" / "00_deploy-all.sh").read_text(encoding="utf-8")
        verify = (ROOT / "scripts" / "60_verify.sh").read_text(encoding="utf-8")
        for relative in (
            "deploy/pre-nereidforge/scripts/10_infra.sh",
            "deploy/pre-nereidforge/scripts/30_install-kubeflow.sh",
        ):
            self.assertIn(relative, deploy_all)
            self.assertTrue((ROOT / relative).is_file())
        for relative in (
            "deploy/pre-nereidforge/scripts/60_verify/infra-check.sh",
            "deploy/pre-nereidforge/scripts/60_verify/kubeflow-check.sh",
            "deploy/pre-nereidforge/scripts/60_verify/mr-check.sh",
        ):
            self.assertIn(relative, verify)
            self.assertTrue((ROOT / relative).is_file())


@unittest.skipUnless(ROOT_COMMON.is_file(), "requires a combined NereidForge checkout")
class RootVersionIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.repo = Path(self.tempdir.name) / "repository"
        (self.repo / "scripts" / "lib").mkdir(parents=True)
        shutil.copy2(ROOT_COMMON, self.repo / "scripts" / "lib" / "common.sh")
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "nf-tests@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "NF Tests"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "scripts/lib/common.sh"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-m", "test fixture"], check=True)

    def resolve(self):
        env = os.environ.copy()
        env.pop("RELEASE_TAG", None)
        return subprocess.run(
            ["bash", "-c", 'source "$1/scripts/lib/common.sh"; nf_version', "_", str(self.repo)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_pre_tag_is_not_an_nf_release(self):
        subprocess.run(["git", "-C", str(self.repo), "tag", "pre-nereidforge/v1.2.3"], check=True)
        result = self.resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0.0.0-dev")

    def test_root_v_tag_behavior_is_unchanged(self):
        subprocess.run(["git", "-C", str(self.repo), "tag", "v1.2.3"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "tag", "pre-nereidforge/v9.8.7"], check=True)
        result = self.resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1.2.3")

    def test_unrelated_legacy_tag_behavior_is_unchanged(self):
        subprocess.run(["git", "-C", str(self.repo), "tag", "legacy-1.2.3"], check=True)
        result = self.resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "legacy-1.2.3")

    def test_all_root_exact_tag_build_entrypoints_exclude_the_pre_namespace(self):
        # These entry points build images and cannot be executed in this local
        # suite. Pin their narrow exclusion without invoking a build.
        entrypoints = (
            "Makefile",
            "builders/build.sh",
            "infra/charts/kubeflow/deploy-tools/build.sh",
            "kfp-pipelines/docker/build-evaluator.sh",
            "kfp-pipelines/docker/build-images.sh",
            "kfp-pipelines/docker/build-optimizer.sh",
            "kfp-pipelines/docker/resume-build.sh",
            "scripts/02_build-images.sh",
        )
        for relative in entrypoints:
            with self.subTest(entrypoint=relative):
                lines = (ROOT / relative).read_text(encoding="utf-8").splitlines()
                lookups = [line for line in lines if "describe --tags --exact-match" in line]
                self.assertTrue(lookups)
                for line in lookups:
                    self.assertIn("--exclude 'pre-nereidforge/v*'", line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
