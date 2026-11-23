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
COMMON = PRE_ROOT / "scripts" / "lib" / "common.sh"


class PreVersionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.base = Path(self.tempdir.name)

    def make_module(self, parent, git=False):
        module = parent / "deploy" / "pre-nereidforge"
        (module / "scripts" / "lib").mkdir(parents=True)
        shutil.copy2(COMMON, module / "scripts" / "lib" / "common.sh")
        (module / "scripts" / "10_infra.sh").write_text("# tracked module marker\n")
        if git:
            subprocess.run(["git", "init", "-q", str(parent)], check=True)
            subprocess.run(["git", "-C", str(parent), "config", "user.email", "pre-tests@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(parent), "config", "user.name", "Pre Tests"], check=True)
            subprocess.run(["git", "-C", str(parent), "add", "deploy/pre-nereidforge"], check=True)
            subprocess.run(["git", "-C", str(parent), "commit", "-q", "-m", "test fixture"], check=True)
        return module

    def resolve(self, module, extra_env=None):
        env = os.environ.copy()
        for name in ("PRE_RELEASE_TAG", "SOURCE_COMMIT", "RELEASE_TAG"):
            env.pop(name, None)
        env["RELEASE_TAG"] = "8.7.6"
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", "-c", 'source "$1/scripts/lib/common.sh"; pre_version', "_", str(module)],
            text=True,
            capture_output=True,
            env=env,
        )

    def assert_version(self, module, expected, extra_env=None):
        result = self.resolve(module, extra_env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), expected, result.stderr)

    def tag(self, repo, name):
        subprocess.run(["git", "-C", str(repo), "tag", name], check=True)

    def test_explicit_tag_wins_over_release_file_and_git_tag(self):
        repo = self.base / "explicit"
        module = self.make_module(repo, git=True)
        self.tag(repo, "pre-nereidforge/v1.2.3")
        (module / "release.env").write_text("PRE_RELEASE_TAG=2.3.4\nSOURCE_COMMIT=generated\n")
        self.assert_version(module, "9.8.7", {"PRE_RELEASE_TAG": "9.8.7"})

    def test_generated_release_env_supplies_version_and_source_commit(self):
        module = self.make_module(self.base / "generated")
        (module / "release.env").write_text(
            "PRE_RELEASE_TAG=2.3.4\nSOURCE_COMMIT=0123456789abcdef\n"
        )
        self.assert_version(module, "2.3.4")
        result = subprocess.run(
            ["bash", "-c", 'source "$1/scripts/lib/common.sh"; printf "%s" "$SOURCE_COMMIT"', "_", str(module)],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "0123456789abcdef")

    def test_exact_pre_semver_tag_supplies_bare_version(self):
        repo = self.base / "pre-tag"
        module = self.make_module(repo, git=True)
        self.tag(repo, "pre-nereidforge/v1.2.3-rc.1")
        self.assert_version(module, "1.2.3-rc.1")

    def test_nf_root_tag_never_leaks_into_pre_version(self):
        repo = self.base / "nf-tag"
        module = self.make_module(repo, git=True)
        self.tag(repo, "v7.6.5")
        self.assert_version(module, "0.0.0-dev")

    def test_non_semver_pre_tag_uses_development_fallback(self):
        repo = self.base / "bad-pre-tag"
        module = self.make_module(repo, git=True)
        self.tag(repo, "pre-nereidforge/v1.2")
        self.assert_version(module, "0.0.0-dev")

    def test_unversioned_copied_source_archive_is_rejected(self):
        module = self.make_module(self.base / "copy")
        result = self.resolve(module)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported unversioned source archive", result.stderr)

    def test_nested_worktree_without_module_git_file_is_supported(self):
        repo = self.base / "worktree"
        module = self.make_module(repo, git=True)
        self.assertFalse((module / ".git").exists())
        self.assertTrue((repo / ".git").exists())
        self.assert_version(module, "0.0.0-dev")


if __name__ == "__main__":
    unittest.main(verbosity=2)
