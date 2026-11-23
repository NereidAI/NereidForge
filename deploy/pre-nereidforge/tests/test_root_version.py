#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PRE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = PRE_ROOT.parents[1]
ROOT_COMMON = REPOSITORY / "scripts" / "lib" / "common.sh"


@unittest.skipUnless(ROOT_COMMON.is_file(), "requires a combined NereidForge checkout")
class RootVersionIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.repo = Path(self.tempdir.name) / "repository"
        (self.repo / "scripts" / "lib").mkdir(parents=True)
        shutil.copy2(ROOT_COMMON, self.repo / "scripts" / "lib" / "common.sh")
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "pre-tests@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "Pre Tests"], check=True)
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
                lines = (REPOSITORY / relative).read_text().splitlines()
                lookups = [line for line in lines if "describe --tags --exact-match" in line]
                self.assertTrue(lookups)
                for line in lookups:
                    self.assertIn("--exclude 'pre-nereidforge/v*'", line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
