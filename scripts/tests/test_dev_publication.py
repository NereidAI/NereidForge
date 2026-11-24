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
STAMP_SCRIPT_PATH = ROOT / "scripts" / "stamp-dev-image.sh"
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


def output_block(text, name):
    match = re.search(
        rf"(?ms)^{re.escape(name)}<<([A-Z0-9_]+)\n(.*?)^\1$",
        text,
    )
    if not match:
        raise AssertionError(f"multiline output not found: {name}")
    return match.group(2).splitlines()


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

    def stamp(self, platforms):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "github-output"
            environment = os.environ.copy()
            environment.update(
                IMMUTABLE="0.0.0-dev-abcdef1",
                EXTRA_BUILD_ARGS="BASE_IMAGE=example/base:@IMMUTABLE@",
                PLATFORMS=platforms,
                GITHUB_SHA="abcdef1234567890",
                GITHUB_OUTPUT=str(output),
            )
            subprocess.run(
                ["bash", str(STAMP_SCRIPT_PATH)],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            return output.read_text(encoding="utf-8")

    def test_stamp_exports_platform_appropriate_annotations(self):
        single = self.stamp("linux/amd64")
        annotations = output_block(single, "annotations")
        self.assertEqual(len(annotations), 5)
        self.assertTrue(all(line.startswith("manifest:") for line in annotations))
        self.assertFalse(any(line.startswith("index") for line in annotations))
        self.assertIn(
            "BASE_IMAGE=example/base:0.0.0-dev-abcdef1",
            output_block(single, "build_args"),
        )

        multiple = self.stamp("linux/amd64,linux/arm64")
        self.assertTrue(
            all(
                line.startswith("index,manifest:")
                for line in output_block(multiple, "annotations")
            )
        )
        for job_name in ("publish-base", "publish-dependent"):
            job = job_block(self.text, job_name)
            self.assertIn("run: bash scripts/stamp-dev-image.sh", job)
            self.assertIn("annotations: ${{ steps.stamp.outputs.annotations }}", job)
            self.assertNotIn("index,manifest:org.opencontainers", job)

    def test_cuda_build_checks_do_not_require_a_gpu_on_the_runner(self):
        for relative_path in (
            "builders/trainer-torch/Dockerfile.cu",
            "builders/trainer-llm-cu/Dockerfile",
        ):
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("torch._C._cuda_getArchFlags().split()", text)
            self.assertIn("'sm_120'", text)
            self.assertNotIn("arches=torch.cuda.get_arch_list()", text)

    def test_paddle_cuda_uses_the_official_package_index_not_a_file_url(self):
        text = (ROOT / "builders/trainer-paddle/Dockerfile.cu").read_text(encoding="utf-8")
        self.assertIn('--index-url "${PADDLE_GPU_INDEX}"', text)
        self.assertIn('"paddlepaddle-gpu==3.3.1"', text)
        self.assertNotIn("PADDLE_GPU_WHEEL", text)

    def test_large_rocm_build_uses_the_independent_temporary_disk(self):
        job = job_block(self.text, "publish-base")
        self.assertIn("if: matrix.name == 'trainer-torch-rocm'", job)
        self.assertIn("Move ROCm BuildKit storage to temporary disk", job)
        self.assertIn('mount --bind "/mnt/nf-docker" /var/lib/docker', job)
        self.assertIn('mount --bind "/mnt/nf-containerd" /var/lib/containerd', job)
        self.assertLess(
            job.index("Move ROCm BuildKit storage to temporary disk"),
            job.index("uses: docker/setup-buildx-action@v4"),
        )

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
