# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Builder-owned Dockerfile and no-publication build-environment contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILDERS = ROOT / "builders"
DEVELOPMENT_WORKFLOW = ROOT / ".github" / "workflows" / "publish-dev.yml"
STAMP_SCRIPT = ROOT / "scripts" / "stamp-dev-image.sh"
ROCM_STORAGE_SCRIPT = ROOT / "scripts" / "prepare-rocm-buildkit-storage.sh"


def job_block(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [a-z][a-z0-9-]*:\n|\Z)",
        text,
    )
    if not match:
        raise AssertionError(f"workflow job not found: {name}")
    return match.group(1)


def output_block(text: str, name: str) -> list[str]:
    match = re.search(
        rf"(?ms)^{re.escape(name)}<<([A-Z0-9_]+)\n(.*?)^\1$",
        text,
    )
    if not match:
        raise AssertionError(f"multiline output not found: {name}")
    return match.group(2).splitlines()


def xpu_copy_instructions() -> list[str]:
    source = (BUILDERS / "trainer-torch" / "Dockerfile.xpu").read_text(
        encoding="utf-8"
    )
    copies = re.findall(r"(?ms)^COPY[ \t]+.*?(?=^[A-Z#]|\Z)", source)
    if not copies:
        raise AssertionError("the XPU trainer must declare COPY inputs")
    return copies


class BuilderDockerfileContractTest(unittest.TestCase):
    def test_xpu_copy_sources_exist_without_docker(self):
        """The offline gate catches missing PR #34 COPY inputs before Buildx runs."""
        for instruction in xpu_copy_instructions():
            tokens = shlex.split(instruction.replace("\\\n", " "))
            self.assertEqual("COPY", tokens[0])
            self.assertGreaterEqual(len(tokens), 3, instruction)
            for source in tokens[1:-1]:
                self.assertFalse(source.startswith("--"), instruction)
                self.assertTrue((BUILDERS / source).exists(), source)

    def test_xpu_dockerfile_copy_sources_build_from_scratch_in_clean_checkout(self):
        """Enforce PR #34 with Buildx, without pulling a base or installing packages."""
        if os.environ.get("NF_REQUIRE_BUILDX") != "1":
            self.skipTest("run make -C builders verify-buildx for the mandatory Buildx check")
        if not shutil.which("docker"):
            self.fail("docker with buildx is required for the clean-checkout context check")
        available = subprocess.run(
            ["docker", "buildx", "version"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if available.returncode != 0:
            self.fail(
                "docker with buildx is required for the clean-checkout context check: "
                + available.stdout
                + available.stderr
            )

        with tempfile.TemporaryDirectory() as directory:
            dockerfile = Path(directory) / "Dockerfile.copy-context"
            dockerfile.write_text(
                "FROM scratch\n" + "".join(xpu_copy_instructions()), encoding="utf-8"
            )
            result = subprocess.run(
                [
                    "docker",
                    "buildx",
                    "build",
                    "--file",
                    str(dockerfile),
                    "--progress=plain",
                    "--pull=false",
                    "builders",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_xpu_base_torch_version_pin_and_metadata_match_trainer(self):
        """Preserve PR #35: one +xpu pin and consistent runtime/distribution metadata."""
        base = (BUILDERS / "trainer-xpu" / "Dockerfile").read_text(encoding="utf-8")
        trainer = (BUILDERS / "trainer-torch" / "Dockerfile.xpu").read_text(
            encoding="utf-8"
        )
        pattern = r"(?m)^ARG TORCH_VERSION=([0-9.]+)$"
        base_version = re.search(pattern, base)
        trainer_version = re.search(pattern, trainer)
        self.assertIsNotNone(base_version, "the base must pin its torch version")
        self.assertIsNotNone(trainer_version, "the trainer must constrain the same torch")
        self.assertEqual(base_version.group(1), trainer_version.group(1))
        self.assertIn("torch==${TORCH_VERSION}+xpu", base)
        self.assertIn("torch==${TORCH_VERSION}+xpu", trainer)
        self.assertIn("importlib.metadata.version('torch')", base)
        self.assertIn("expected = '${TORCH_VERSION}+xpu'", base)
        self.assertIn(
            "assert importlib.metadata.version('torch') == expected",
            base,
        )

    def test_builder_legacy_build_entrypoints_exclude_pre_release_tags(self):
        """Docker-only changes must retain NF/pre version-channel isolation."""
        entrypoints = (
            "kfp-pipelines/docker/build-evaluator.sh",
            "kfp-pipelines/docker/build-images.sh",
            "kfp-pipelines/docker/build-optimizer.sh",
            "kfp-pipelines/docker/resume-build.sh",
        )
        for relative in entrypoints:
            with self.subTest(entrypoint=relative):
                lines = (ROOT / relative).read_text(encoding="utf-8").splitlines()
                lookups = [line for line in lines if "describe --tags --exact-match" in line]
                self.assertTrue(lookups)
                for line in lookups:
                    self.assertIn("--exclude 'pre-nereidforge/v*'", line)

    def test_cuda_build_checks_do_not_require_a_gpu_on_the_runner(self):
        for relative_path in (
            "trainer-torch/Dockerfile.cu",
            "trainer-llm-cu/Dockerfile",
        ):
            text = (BUILDERS / relative_path).read_text(encoding="utf-8")
            self.assertIn("torch._C._cuda_getArchFlags().split()", text)
            self.assertIn("'sm_120'", text)
            self.assertNotIn("arches=torch.cuda.get_arch_list()", text)

    def test_paddle_cuda_uses_the_official_package_index_not_a_file_url(self):
        text = (BUILDERS / "trainer-paddle" / "Dockerfile.cu").read_text(
            encoding="utf-8"
        )
        self.assertIn('--index-url "${PADDLE_GPU_INDEX}"', text)
        self.assertIn('"paddlepaddle-gpu==3.3.1"', text)
        self.assertNotIn("PADDLE_GPU_WHEEL", text)

    def test_paddle_cuda_build_uses_compat_driver_only_during_builds(self):
        text = (BUILDERS / "trainer-paddle" / "Dockerfile.cu").read_text(
            encoding="utf-8"
        )
        compat = 'export LD_LIBRARY_PATH="/usr/local/cuda/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}";'
        self.assertGreaterEqual(text.count(compat), 3)
        self.assertNotIn("ENV LD_LIBRARY_PATH=/usr/local/cuda/compat", text)
        self.assertIn("pip uninstall -y torch torchvision torchaudio torchelastic triton", text)
        self.assertEqual(text.count("pip check"), 2)
        self.assertIn("rm -rf /opt/nf-trainer-sdk && pip check", text)
        self.assertIn('repo_dir="$(python -c', text)
        self.assertIn('os.path.dirname(paddlex.__file__),"repo_manager","repos"', text)
        self.assertNotIn(
            "mkdir -p /usr/local/lib/python3.11/site-packages/paddlex/repo_manager/repos",
            text,
        )
        opencv = '"opencv-python==4.5.5.64"'
        opencv_contrib = '"opencv-contrib-python==4.10.0.84"'
        self.assertIn(opencv, text)
        self.assertIn(opencv_contrib, text)
        self.assertLess(text.index(opencv), text.index(opencv_contrib))
        self.assertIn('hasattr(cv2, "ximgproc")', text)

    def stamp(self, platforms: str) -> str:
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
                ["bash", str(STAMP_SCRIPT)],
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
        workflow = DEVELOPMENT_WORKFLOW.read_text(encoding="utf-8")
        for job_name in ("publish-base", "publish-dependent"):
            job = job_block(workflow, job_name)
            self.assertIn("run: bash scripts/stamp-dev-image.sh", job)
            self.assertIn("annotations: ${{ steps.stamp.outputs.annotations }}", job)
            self.assertNotIn("index,manifest:org.opencontainers", job)

    def test_release_dockerfiles_retain_standard_oci_annotation_labels(self):
        inventory = json.loads(
            (BUILDERS / "image-inventory.json").read_text(encoding="utf-8")
        )
        for image in inventory["images"]:
            with self.subTest(image=image["name"]):
                text = (ROOT / image["dockerfile"]).read_text(encoding="utf-8")
                for key in ("version", "revision", "created", "source"):
                    self.assertIn(f"org.opencontainers.image.{key}", text)
                self.assertIn('org.opencontainers.image.version="${VERSION}"', text)
                self.assertIn('org.opencontainers.image.revision="${GIT_SHA}"', text)
                self.assertIn('org.opencontainers.image.created="${BUILD_DATE}"', text)

    def prepare_rocm_storage(
        self,
        root_source,
        temporary_source,
        available_kb,
        strict_findmnt=False,
    ):
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
                ["bash", str(ROCM_STORAGE_SCRIPT)],
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

    def test_rocm_storage_fails_closed_without_48_gib_on_selected_filesystem(self):
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
        workflow = DEVELOPMENT_WORKFLOW.read_text(encoding="utf-8")
        job = job_block(workflow, "publish-base")
        self.assertIn("if: matrix.name == 'trainer-torch-rocm'", job)
        self.assertIn("Prepare ROCm BuildKit storage", job)
        self.assertIn("run: bash scripts/prepare-rocm-buildkit-storage.sh", job)
        self.assertLess(
            job.index("Prepare ROCm BuildKit storage"),
            job.index("uses: docker/setup-buildx-action@v4"),
        )


if __name__ == "__main__":
    unittest.main()
