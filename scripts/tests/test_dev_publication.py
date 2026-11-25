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
ROCM_STORAGE_SCRIPT_PATH = ROOT / "scripts" / "prepare-rocm-buildkit-storage.sh"
AMD_NVIDIA_IMAGES = {
    "trainer-llm-cu",
    "trainer-llm-rocm",
    "trainer-paddle-cu",
    "trainer-torch-cu",
    "trainer-torch-rocm",
}
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

    def test_complete_publication_requires_manual_dispatch_on_main(self):
        trigger = re.search(r"(?ms)^on:\n(.*?)(?=^permissions:\n)", self.text)
        self.assertIsNotNone(trigger)
        self.assertIn("  workflow_dispatch:\n", trigger.group(1))
        self.assertIn("      image_profile:\n", trigger.group(1))
        self.assertIn("        default: full\n", trigger.group(1))
        self.assertIn("          - full\n", trigger.group(1))
        self.assertIn("          - cpu-xpu\n", trigger.group(1))
        self.assertIn("          - backend-frontend\n", trigger.group(1))
        self.assertNotIn("  path-scope:\n", self.text)
        self.assertNotIn("github.event.before", self.text)

        preflight = job_block(self.text, "preflight")
        self.assertIn('test "$GITHUB_EVENT_NAME" = workflow_dispatch', preflight)
        self.assertIn('test "$GITHUB_REF_NAME" = main', preflight)
        for name in ("publish-base", "publish-dependent"):
            self.assertNotIn("path-scope", job_block(self.text, name))

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

    def test_cpu_xpu_profile_selects_18_images_and_excludes_only_amd_nvidia_variants(self):
        self.assertEqual(len(EXPECTED_IMAGES - AMD_NVIDIA_IMAGES), 18)
        preflight = job_block(self.text, "preflight")
        self.assertIn('case "$IMAGE_PROFILE" in', preflight)
        self.assertIn('full) expected_count=23', preflight)
        self.assertIn('cpu-xpu) expected_count=18', preflight)
        self.assertIn('if [[ "$IMAGE_PROFILE" == full ]]; then', preflight)
        for name in AMD_NVIDIA_IMAGES:
            self.assertIn(name, preflight)
        self.assertIn('test "${#images[@]}" -eq "$EXPECTED_COUNT"', preflight)
        self.assertIn('echo "selected=$selected_json" >> "$GITHUB_OUTPUT"', preflight)
        self.assertIn('echo "expected_count=$expected_count" >> "$GITHUB_OUTPUT"', preflight)

        for job_name in ("publish-base", "publish-dependent"):
            job = job_block(self.text, job_name)
            self.assertIn("contains(fromJSON(needs.preflight.outputs.selected), matrix.name)", job)
        base = job_block(self.text, "publish-base")
        self.assertIn("if: ${{ !contains(fromJSON(needs.preflight.outputs.selected), matrix.name) }}", base)
        self.assertIn("if: matrix.name == 'trainer-torch-rocm' && contains(fromJSON(needs.preflight.outputs.selected), matrix.name)", base)

        alias = job_block(self.text, "alias-images")
        self.assertIn('profile = os.environ["IMAGE_PROFILE"]', alias)
        self.assertIn('if profile == "full":', alias)
        self.assertIn('expected_count = int(os.environ["EXPECTED_COUNT"])', alias)
        self.assertIn('assert len(images) == expected_count', alias)
        self.assertIn('assert len(digests) == int(os.environ["EXPECTED_COUNT"])', alias)
        self.assertNotIn("assert len(images) == 23", alias)
        self.assertNotIn("assert len(digests) == 23", alias)
        self.assertNotIn("All 23 immutable images", alias)

        shell_base = re.search(r"(?ms)^          images=\(\n(.*?)^          \)", preflight)
        shell_extra = re.search(r"(?ms)^            images\+=\(\n(.*?)^            \)", preflight)
        python_base = re.search(r"(?ms)^          images = \[\n(.*?)^          \]", alias)
        python_extra = re.search(r"(?ms)^              images \+= \[\n(.*?)^              \]", alias)
        for match in (shell_base, shell_extra, python_base, python_extra):
            self.assertIsNotNone(match)
        shell_names = lambda match: set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", match.group(1)))
        python_names = lambda match: set(re.findall(r'"([a-z0-9-]+)"', match.group(1)))
        expected_selected = EXPECTED_IMAGES - AMD_NVIDIA_IMAGES
        self.assertEqual(shell_names(shell_base), expected_selected)
        self.assertEqual(python_names(python_base), expected_selected)
        self.assertEqual(shell_names(shell_extra), AMD_NVIDIA_IMAGES)
        self.assertEqual(python_names(python_extra), AMD_NVIDIA_IMAGES)

    def test_backend_frontend_profile_runs_only_two_application_image_builds(self):
        """The narrow profile skips builder/dependent matrices and all charts."""
        applications = {"nereidforge-backend", "nereidforge-frontend"}
        preflight = job_block(self.text, "preflight")
        self.assertIn("backend-frontend) expected_count=2", preflight)
        self.assertIn(
            'if [[ "$IMAGE_PROFILE" == backend-frontend ]]; then\n'
            "            # Application-only publication must not even inventory builder tags.\n"
            "            images=(nereidforge-backend nereidforge-frontend)",
            preflight,
        )

        application_job = job_block(self.text, "publish-applications")
        self.assertIn("if: inputs.image_profile == 'backend-frontend'", application_job)
        self.assertEqual(matrix_names(self.text, "publish-applications"), applications)
        for name in applications:
            self.assertEqual(
                application_job.count(f"- name: {name}\n"),
                1,
                "each application must occur once in the narrow matrix",
            )
        for job_name in ("publish-base", "publish-dependent"):
            job = job_block(self.text, job_name)
            self.assertIn("if: inputs.image_profile != 'backend-frontend'", job)

        alias = job_block(self.text, "alias-images")
        self.assertIn(
            "needs: [preflight, publish-applications, publish-base, publish-dependent]",
            alias,
        )
        self.assertIn("needs.publish-applications.result == 'success'", alias)
        self.assertIn(
            'if profile == "backend-frontend":\n'
            '              images = ["nereidforge-backend", "nereidforge-frontend"]',
            alias,
        )
        charts = job_block(self.text, "publish-charts")
        self.assertIn("if: inputs.image_profile != 'backend-frontend'", charts)

    def test_gpu_images_are_not_static_chart_install_dependencies(self):
        chart_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "charts").rglob("*.yaml")
        )
        for name in AMD_NVIDIA_IMAGES:
            self.assertNotIn(name, chart_text)

    def test_existing_images_are_checked_before_any_build(self):
        preflight = job_block(self.text, "preflight")
        self.assertIn("EXPECTED_REVISION: ${{ github.sha }}", preflight)
        self.assertIn('_pre_oras manifest fetch "${reference}@${PRE_ORAS_DIGEST}"', preflight)
        self.assertIn("actual_revision != revision or actual_version != immutable", preflight)
        self.assertIn("immutable collision", preflight)
        self.assertLess(self.text.index("  preflight:\n"), self.text.index("  publish-base:\n"))

    def test_floating_images_move_only_after_anonymous_immutable_verification(self):
        for job_name in ("publish-applications", "publish-base", "publish-dependent"):
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
        floating_index = alias.index("Anonymously verify selected floating aliases against immutable digests")
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
        for job_name in ("publish-applications", "publish-base", "publish-dependent"):
            job = job_block(self.text, job_name)
            self.assertIn("run: bash scripts/stamp-dev-image.sh", job)
            self.assertIn("annotations: ${{ steps.stamp.outputs.annotations }}", job)
            self.assertNotIn("index,manifest:org.opencontainers", job)

    def test_xpu_dockerfile_copy_sources_are_present_in_clean_checkout(self):
        """A scratch build checks COPY inputs without pulling or installing dependencies."""
        if not shutil.which("docker"):
            self.skipTest("docker with buildx is required for the build-context check")
        available = subprocess.run(
            ["docker", "buildx", "version"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if available.returncode != 0:
            self.skipTest("docker with buildx is required for the build-context check")

        source = (ROOT / "builders/trainer-torch/Dockerfile.xpu").read_text(encoding="utf-8")
        copies = re.findall(r"(?ms)^COPY[ \t]+.*?(?=^[A-Z#]|\Z)", source)
        self.assertTrue(copies)
        with tempfile.TemporaryDirectory() as directory:
            dockerfile = Path(directory) / "Dockerfile.copy-context"
            dockerfile.write_text("FROM scratch\n" + "".join(copies), encoding="utf-8")
            result = subprocess.run(
                [
                    "docker", "buildx", "build", "--file", str(dockerfile),
                    "--progress=plain", "builders",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_xpu_base_torch_version_matches_vision_trainer_constraint(self):
        base = (ROOT / "builders/trainer-xpu/Dockerfile").read_text(encoding="utf-8")
        trainer = (ROOT / "builders/trainer-torch/Dockerfile.xpu").read_text(encoding="utf-8")
        pattern = r"(?m)^ARG TORCH_VERSION=([0-9.]+)$"
        base_version = re.search(pattern, base)
        trainer_version = re.search(pattern, trainer)
        self.assertIsNotNone(base_version, "the base must pin its torch version")
        self.assertIsNotNone(trainer_version)
        self.assertEqual(base_version.group(1), trainer_version.group(1))
        self.assertIn("torch==${TORCH_VERSION}+xpu", base)
        self.assertIn("importlib.metadata.version('torch')", base)

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

    def test_paddle_cuda_build_uses_compat_driver_only_during_builds(self):
        text = (ROOT / "builders/trainer-paddle/Dockerfile.cu").read_text(encoding="utf-8")
        compat = 'export LD_LIBRARY_PATH="/usr/local/cuda/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}";'
        self.assertGreaterEqual(text.count(compat), 3)
        self.assertNotIn("ENV LD_LIBRARY_PATH=/usr/local/cuda/compat", text)
        self.assertIn("pip uninstall -y torch torchvision torchaudio torchelastic triton", text)
        self.assertEqual(text.count("pip check"), 2)
        self.assertIn("rm -rf /opt/nf-trainer-sdk && pip check", text)
        self.assertIn('repo_dir="$(python -c', text)
        self.assertIn('os.path.dirname(paddlex.__file__),"repo_manager","repos"', text)
        self.assertNotIn("mkdir -p /usr/local/lib/python3.11/site-packages/paddlex/repo_manager/repos", text)
        opencv = '"opencv-python==4.5.5.64"'
        opencv_contrib = '"opencv-contrib-python==4.10.0.84"'
        self.assertIn(opencv, text)
        self.assertIn(opencv_contrib, text)
        self.assertLess(text.index(opencv), text.index(opencv_contrib))
        self.assertIn('hasattr(cv2, "ximgproc")', text)

    def prepare_rocm_storage(self, root_source, temporary_source, available_kb, strict_findmnt=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary_dir = root / "bin"
            binary_dir.mkdir()
            commands = {
                "findmnt": """#!/usr/bin/env bash
for target do :; done
if [[ "$FAKE_STRICT_FINDMNT" == true && "$target" == /mnt &&
      "$FAKE_TEMPORARY_SOURCE" == "$FAKE_ROOT_SOURCE" && " $* " != *" --target "* ]]; then
  exit 1
fi
case "$target" in
  /) printf '%s\\n' "$FAKE_ROOT_SOURCE" ;;
  /mnt) printf '%s\\n' "$FAKE_TEMPORARY_SOURCE" ;;
  *) exit 1 ;;
esac
""",
                "df": """#!/usr/bin/env bash
if [[ "${1:-}" == -Pk ]]; then
  echo 'Filesystem 1024-blocks Used Available Capacity Mounted on'
  printf 'filesystem 209715200 1024 %s 1%% %s\\n' "$FAKE_AVAILABLE_KB" "$2"
else
  echo 'Filesystem Size Used Avail Use% Mounted on'
  echo 'filesystem 145G 32G 113G 22% /'
fi
""",
                "sudo": """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_SUDO_LOG"
""",
            }
            for name, content in commands.items():
                path = binary_dir / name
                path.write_text(content, encoding="utf-8")
                path.chmod(0o755)
            sudo_log = root / "sudo.log"
            environment = os.environ.copy()
            environment.update(
                PATH=f"{binary_dir}:{environment['PATH']}",
                FAKE_ROOT_SOURCE=root_source,
                FAKE_TEMPORARY_SOURCE=temporary_source,
                FAKE_AVAILABLE_KB=str(available_kb),
                FAKE_STRICT_FINDMNT=str(strict_findmnt).lower(),
                FAKE_SUDO_LOG=str(sudo_log),
            )
            result = subprocess.run(
                ["bash", str(ROCM_STORAGE_SCRIPT_PATH)],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            operations = sudo_log.read_text(encoding="utf-8").splitlines()
            return result, operations

    def test_rocm_storage_accepts_a_large_shared_root_filesystem(self):
        result, operations = self.prepare_rocm_storage(
            "/dev/mapper/cow-root",
            "/dev/mapper/cow-root",
            113 * 1024 * 1024,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("/mnt shares the root filesystem", result.stdout)
        self.assertEqual(operations, ["docker image prune --all --force"])

    def test_rocm_storage_resolves_a_shared_mnt_by_target_path(self):
        result, operations = self.prepare_rocm_storage(
            "/dev/root",
            "/dev/root",
            110 * 1024 * 1024,
            strict_findmnt=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("/mnt shares the root filesystem", result.stdout)
        self.assertEqual(operations, ["docker image prune --all --force"])

    def test_rocm_storage_moves_to_an_independent_temporary_disk(self):
        result, operations = self.prepare_rocm_storage(
            "/dev/mapper/cow-root",
            "/dev/sdb1",
            48 * 1024 * 1024,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Using the independent /mnt filesystem", result.stdout)
        self.assertIn("systemctl stop docker.socket docker.service containerd.service", operations)
        self.assertIn("mount --bind /mnt/nf-docker /var/lib/docker", operations)
        self.assertIn("mount --bind /mnt/nf-containerd /var/lib/containerd", operations)
        self.assertIn("systemctl start containerd.service docker.service", operations)

    def test_rocm_storage_fails_closed_without_48_gib_on_the_selected_filesystem(self):
        for label, temporary_source, storage_path in (
            ("shared root", "/dev/mapper/cow-root", "/"),
            ("independent temporary", "/dev/sdb1", "/mnt"),
        ):
            with self.subTest(layout=label):
                result, operations = self.prepare_rocm_storage(
                    "/dev/mapper/cow-root",
                    temporary_source,
                    48 * 1024 * 1024 - 1,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"at least 48 GiB free at {storage_path}", result.stdout)
                self.assertEqual(operations, ["docker image prune --all --force"])

    def test_large_rocm_build_prepares_storage_before_buildx(self):
        job = job_block(self.text, "publish-base")
        self.assertIn("if: matrix.name == 'trainer-torch-rocm'", job)
        self.assertIn("Prepare ROCm BuildKit storage", job)
        self.assertIn("run: bash scripts/prepare-rocm-buildkit-storage.sh", job)
        self.assertLess(
            job.index("Prepare ROCm BuildKit storage"),
            job.index("uses: docker/setup-buildx-action@v4"),
        )


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
