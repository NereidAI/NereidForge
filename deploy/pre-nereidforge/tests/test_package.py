#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


PRE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHARTS = {
    "cvat": ("2.73.0", "2.73.0"),
    "labelstudio": ("1.23.0", "1.23.0"),
    "postgresql": ("0.29.0", "16.4"),
    "seaweedfs": ("3.60.0", "4.34"),
    "valkey": ("1.0.0", "8.0"),
}


class PrePackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inside_git = subprocess.run(
            ["git", "-C", str(PRE_ROOT), "rev-parse", "--show-toplevel"],
            text=True,
            capture_output=True,
        )
        if inside_git.returncode != 0:
            raise unittest.SkipTest("building a package requires a Git worktree")
        cls.tempdir = tempfile.TemporaryDirectory(prefix="pre-package-tests-")
        cls.base = Path(cls.tempdir.name)
        cls.packages = {}
        for channel, version in (("formal", "0.0.1"), ("development", "0.0.0-dev")):
            output = cls.base / channel / "output"
            env = os.environ.copy()
            env.pop("PRE_RELEASE_TAG", None)
            env.pop("RELEASE_TAG", None)
            env["SOURCE_COMMIT"] = "inherited-metadata-must-not-win"
            command = ["make", "package", f"PRE_OUTPUT_DIR={output}"]
            if channel == "formal":
                command.append("PRE_RELEASE_TAG=0.0.1")
            result = subprocess.run(
                command,
                cwd=PRE_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=180,
            )
            if result.returncode != 0:
                raise AssertionError(
                    f"{channel} package failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )
            archive = output / f"pre-nereidforge-installer-{version}.tar.gz"
            cls.packages[channel] = (version, archive)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_archive_checksum_contents_and_independent_reparse(self):
        for channel, (version, archive) in self.packages.items():
            with self.subTest(channel=channel):
                self.assertTrue(archive.is_file(), archive)
                checksum = Path(str(archive) + ".sha256")
                self.assertTrue(checksum.is_file())
                result = subprocess.run(
                    ["sha256sum", "-c", checksum.name],
                    cwd=archive.parent,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                prefix = f"pre-nereidforge-installer-{version}"
                with tarfile.open(archive, "r:gz") as bundle:
                    names = bundle.getnames()
                    self.assertTrue(names)
                    for name in names:
                        path = Path(name)
                        self.assertFalse(path.is_absolute())
                        self.assertNotIn("..", path.parts)
                        self.assertEqual(path.parts[0], prefix)
                    required = [
                        "Makefile",
                        "README.md",
                        "release.env",
                        "install-lock.json",
                        "scripts/lib/common.sh",
                        "scripts/lib/config.sh",
                        "scripts/lib/generate_nf_handoff.py",
                        "scripts/40_finalize-handoff.sh",
                        "scripts/40_check-nf-handoff.sh",
                        "scripts/lib/provision_s3_buckets.py",
                        "scripts/lib/s3_credentials.py",
                        "scripts/lib/publication.sh",
                        "scripts/package.sh",
                        "scripts/package-charts.sh",
                        "platform/kubeflow-patches/apply.sh",
                        "platform/kubeflow-manifests/common/dex/base/dex-passwords.yaml",
                    ] + [f"charts/{name}/Chart.yaml" for name in EXPECTED_CHARTS]
                    for relative in required:
                        self.assertIn(f"{prefix}/{relative}", names, relative)
                    forbidden = [
                        "scripts/00_deploy-all.sh",
                        "values-prod.yaml",
                        "secrets.yaml.template",
                        "images-manifest.txt",
                        "charts/nereidforge-backend/Chart.yaml",
                        ".git",
                    ]
                    for relative in forbidden:
                        self.assertNotIn(f"{prefix}/{relative}", names, relative)
                    release_env = bundle.extractfile(f"{prefix}/release.env").read().decode()
                    self.assertIn(f"PRE_RELEASE_TAG={version}\n", release_env)
                    internal_lock = bundle.extractfile(f"{prefix}/install-lock.json").read()
                    external_lock = archive.parent / f"pre-nereidforge-installer-{version}.lock.json"
                    self.assertEqual(internal_lock, external_lock.read_bytes())
                    bundle.extractall(self.base / channel / "reparse", filter="data")

                extracted = self.base / channel / "reparse" / prefix
                env = os.environ.copy()
                for name in ("PRE_RELEASE_TAG", "RELEASE_TAG", "SOURCE_COMMIT", "GIT_DIR", "GIT_WORK_TREE"):
                    env.pop(name, None)
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'set -e; source "$1/scripts/lib/common.sh"; test "$(pre_version)" = "$2"; source "$1/scripts/lib/config.sh"; test "$PRE_ROOT" = "$1"; test "$CHARTS_DIR" = "$1/charts"; test "$MANIFESTS_DIR" = "$1/platform/kubeflow-manifests"; printf "%s" "$(pre_version)"',
                        "_",
                        str(extracted),
                        version,
                    ],
                    cwd=self.base,
                    env=env,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, version)

                # The extracted module must generate and validate the NF handoff
                # without Git, repository-root charts, or NF environment values.
                handoff = self.base / channel / "generated-handoff"
                generated = handoff / "values.generated.yaml"
                operator = handoff / "operator-values.yaml"
                operator.parent.mkdir(parents=True, exist_ok=True)
                operator.write_text("{}\n", encoding="utf-8")
                generate = subprocess.run(
                    [
                        "python3", str(extracted / "scripts/lib/generate_nf_handoff.py"),
                        "generate", "--output", str(generated), "--pre-version", version,
                        "--source-commit", "archive-test", "--base-domain", "example.test",
                        "--storage-class", "nfs-csi", "--pg-cluster", "nereidforge-postgresql",
                        "--pg-database", "nereidforge", "--pg-user", "nereidforge",
                        "--dex-email", "admin@example.test",
                    ],
                    cwd=self.base,
                    env=env,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(generate.returncode, 0, generate.stderr)
                validate = subprocess.run(
                    [
                        "python3", str(extracted / "scripts/lib/generate_nf_handoff.py"),
                        "validate", "--values", str(generated), "--operator-values", str(operator),
                    ],
                    cwd=self.base,
                    env=env,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(validate.returncode, 0, validate.stderr)

    def test_five_chart_lock_and_kubeflow_provenance(self):
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PRE_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        for channel, (version, archive) in self.packages.items():
            with self.subTest(channel=channel):
                lock = json.loads(
                    (archive.parent / f"pre-nereidforge-installer-{version}.lock.json").read_text()
                )
                self.assertEqual(lock["schemaVersion"], 1)
                self.assertEqual(lock["version"], version)
                self.assertEqual(lock["sourceCommit"], source_commit)
                charts = {chart["name"]: chart for chart in lock["charts"]}
                self.assertEqual(set(charts), set(EXPECTED_CHARTS))
                self.assertEqual(len(lock["charts"]), 5)
                for name, (chart_version, app_version) in EXPECTED_CHARTS.items():
                    self.assertEqual(charts[name]["version"], chart_version)
                    self.assertEqual(charts[name]["appVersion"], app_version)
                    self.assertEqual(charts[name]["path"], f"charts/{name}")
                baseline = lock["kubeflow"]["baseline"]
                self.assertEqual(
                    baseline,
                    {
                        "repository": "https://github.com/kubeflow/community-distribution",
                        "tag": "26.03.1",
                        "commit": "f09f3ee",
                    },
                )
                patches = lock["kubeflow"]["patches"]
                self.assertEqual(patches["path"], "platform/kubeflow-patches")
                files = {item["path"]: item["sha256"] for item in patches["files"]}
                patch_diffs = [path for path in files if path.endswith("/patch.diff")]
                self.assertEqual(len(patch_diffs), 18)
                self.assertIn("platform/kubeflow-patches/apply.sh", files)
                self.assertIn("platform/kubeflow-patches/removals.txt", files)
                for relative, expected_digest in files.items():
                    self.assertRegex(expected_digest, r"^[0-9a-f]{64}$")
                    digest = hashlib.sha256((PRE_ROOT / relative).read_bytes()).hexdigest()
                    self.assertEqual(expected_digest, digest, relative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
