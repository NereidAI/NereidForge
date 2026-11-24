#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "publish-dev.yml"
EXPECTED_IMAGES = {
    "inference-model-server-openvino-xpu",
    "ml-backend",
    "ml-backend-custom",
    "nereidforge-backend",
    "nereidforge-frontend",
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


def job_block(text, name):
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [a-z][a-z0-9-]*:\n|\Z)",
        text,
    )
    if not match:
        raise AssertionError(f"workflow job not found: {name}")
    return match.group(1)


def matrix_names(text, job_name):
    return set(re.findall(r"(?m)^          - name: ([a-z0-9-]+)\s*$", job_block(text, job_name)))


class DevelopmentPublicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW_PATH.read_text(encoding="utf-8")

    def test_complete_image_snapshot_is_published_before_the_four_charts(self):
        self.assertEqual(len(EXPECTED_IMAGES), 23)
        published = matrix_names(self.text, "publish-base") | matrix_names(self.text, "publish-dependent")
        self.assertEqual(published, EXPECTED_IMAGES)
        self.assertIn("needs: alias-images", job_block(self.text, "publish-charts"))
        self.assertEqual(
            matrix_names(self.text, "publish-charts"),
            {
                "nereidforge-backend",
                "nereidforge-frontend",
                "nereidforge-ml-backend",
                "nereidforge-tenant-check",
            },
        )

    def test_existing_images_are_checked_before_any_build(self):
        preflight = job_block(self.text, "preflight")
        self.assertIn("EXPECTED_REVISION: ${{ github.sha }}", preflight)
        self.assertIn('_pre_oras manifest fetch "${reference}@${PRE_ORAS_DIGEST}"', preflight)
        self.assertIn("actual_revision != revision or actual_version != immutable", preflight)
        self.assertIn("immutable collision", preflight)
        self.assertLess(self.text.index("  preflight:\n"), self.text.index("  publish-base:\n"))

    def test_floating_images_move_only_after_anonymous_immutable_verification(self):
        for job_name in ("publish-base", "publish-dependent"):
            job = job_block(self.text, job_name)
            self.assertEqual(job.count("uses: docker/build-push-action@v6"), 1)
            self.assertIn("contains(fromJSON(needs.preflight.outputs.existing)", job)
            self.assertIn(
                "tags: ghcr.io/nereidai/${{ matrix.name }}:${{ needs.preflight.outputs.immutable }}",
                job,
            )
        alias = job_block(self.text, "alias-images")
        immutable_index = alias.index("Anonymously verify every immutable image and record its digest")
        alias_index = alias.index("Alias floating tags only to anonymously verified immutable digests")
        floating_index = alias.index("Anonymously verify all 23 floating aliases against immutable digests")
        self.assertLess(immutable_index, alias_index)
        self.assertLess(alias_index, floating_index)
        self.assertIn('digest = oras(["resolve"], reference)', alias)
        self.assertIn('manifest = json.loads(oras(["manifest", "fetch"], reference))', alias)
        self.assertNotIn("gh release", self.text)
        self.assertNotIn("0.0.1", self.text)

    def test_backend_notice_change_cannot_be_excluded_as_documentation(self):
        scope = job_block(self.text, "path-scope")
        self.assertLess(scope.index("THIRD_PARTY_NOTICES.md)"), scope.index("|docs/*|*.md)"))


@unittest.skipUnless(shutil.which("helm"), "helm is required for chart packaging tests")
class ReproducibleChartPackagingTest(unittest.TestCase):
    def package(self, root):
        output = root / "charts"
        environment = os.environ.copy()
        environment.update(
            CHART_VERSION="0.0.0-dev-abcdef1",
            CHARTS_OUTPUT_DIR=str(output),
            CHARTS_ARCHIVE=str(root / "charts.tar.gz"),
        )
        subprocess.run(
            ["bash", "scripts/03_package-charts.sh"],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return {path.name: path.read_bytes() for path in output.glob("*.tgz")}

    def test_repackaging_one_development_commit_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = self.package(Path(first_dir))
            time.sleep(2)
            second = self.package(Path(second_dir))
        self.assertEqual(len(first), 4)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
