#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time
import unittest


PRE_ROOT = Path(__file__).resolve().parents[1]
HELM = shutil.which("helm")
PRE_VERSIONS = {
    "cvat": ("2.73.0", "2.73.0"),
    "labelstudio": ("1.23.0", "1.23.0"),
    "postgresql": ("0.29.0", "16.4"),
    "seaweedfs": ("3.60.0", "4.34"),
    "valkey": ("1.0.0", "8.0"),
}


def metadata(helm, chart):
    result = subprocess.run(
        [helm, "show", "chart", str(chart)],
        text=True,
        capture_output=True,
        check=True,
    )
    values = {}
    for key in ("name", "version", "appVersion"):
        line = next(line for line in result.stdout.splitlines() if line.startswith(key + ":"))
        values[key] = line.split(":", 1)[1].strip().strip('"')
    return values


@unittest.skipUnless(HELM, "requires Helm")
class ChartPackagingTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="chart-packaging-tests-")
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

    def run_packager(self, command, cwd, extra_env):
        env = os.environ.copy()
        env["HELM"] = str(self.wrapper)
        env.update(extra_env)
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def package_pre(self, channel, version=None):
        output = self.base / channel
        archive = self.base / f"{channel}.tar.gz"
        command = [
            "make",
            "helm-package",
            f"PRE_CHARTS_OUTPUT_DIR={output}",
            f"PRE_CHARTS_ARCHIVE={archive}",
        ]
        if version:
            command.append(f"PRE_CHARTS_VERSION={version}")
        self.run_packager(command, PRE_ROOT, {})
        return output, archive

    def test_pre_formal_packages_preserve_five_independent_versions(self):
        sources = {
            name: hashlib.sha256((PRE_ROOT / "charts" / name / "Chart.yaml").read_bytes()).hexdigest()
            for name in PRE_VERSIONS
        }
        output, archive = self.package_pre("pre-formal")
        packages = sorted(output.glob("*.tgz"))
        self.assertEqual(len(packages), 5)
        actual = {metadata(HELM, package)["name"]: metadata(HELM, package) for package in packages}
        self.assertEqual(set(actual), set(PRE_VERSIONS))
        for name, (version, app_version) in PRE_VERSIONS.items():
            self.assertEqual(actual[name]["version"], version)
            self.assertEqual(actual[name]["appVersion"], app_version)
            source = PRE_ROOT / "charts" / name / "Chart.yaml"
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), sources[name])
        with tarfile.open(archive, "r:gz") as bundle:
            names = bundle.getnames()
            self.assertEqual(len([name for name in names if name.endswith(".tgz")]), 5)
            self.assertFalse(any("nereidforge-backend" in name for name in names))

    def test_pre_development_packages_override_only_chart_version(self):
        output, _ = self.package_pre("pre-development", "0.0.0-dev-abcdef1")
        packages = sorted(output.glob("*.tgz"))
        self.assertEqual(len(packages), 5)
        actual = {metadata(HELM, package)["name"]: metadata(HELM, package) for package in packages}
        self.assertEqual(set(actual), set(PRE_VERSIONS))
        for name, (_, app_version) in PRE_VERSIONS.items():
            self.assertEqual(actual[name]["version"], "0.0.0-dev-abcdef1")
            self.assertEqual(actual[name]["appVersion"], app_version)

    def test_repackaging_one_development_commit_is_byte_identical(self):
        version = "0.0.0-dev-abcdef1"
        first, _ = self.package_pre("pre-deterministic-first", version)
        time.sleep(2)
        second, _ = self.package_pre("pre-deterministic-second", version)
        first_packages = {path.name: path.read_bytes() for path in first.glob("*.tgz")}
        second_packages = {path.name: path.read_bytes() for path in second.glob("*.tgz")}
        self.assertEqual(len(first_packages), 5)
        self.assertEqual(first_packages, second_packages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
