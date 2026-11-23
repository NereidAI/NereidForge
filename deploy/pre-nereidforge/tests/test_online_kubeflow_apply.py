# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Local regression tests for fail-closed ordered Kubeflow installation."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


PRE_ROOT = Path(__file__).resolve().parents[1]


class KubeflowApplyTest(unittest.TestCase):
    """Exercise only the ordered build/apply stage with local kubectl fakes."""

    def run_apply_stage(self, mode):
        source = (PRE_ROOT / "scripts/30_install-kubeflow.sh").read_text(encoding="utf-8")
        start = source.index("tapply() {")
        end = source.index("# ---- post-apply customizations ----")
        apply_stage = source[start:end]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            kubectl = fake_bin / "kubectl"
            kubectl.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/bash
                    set -u
                    bump() {
                        local file="$1" count=0
                        if [[ -f "$file" ]]; then read -r count < "$file"; fi
                        count=$((count+1))
                        printf '%s\\n' "$count" > "$file"
                    }
                    case "${1:-}" in
                        get)
                            exit 0
                            ;;
                        kustomize)
                            bump "$NF_TEST_KUSTOMIZE_COUNT"
                            read -r count < "$NF_TEST_KUSTOMIZE_COUNT"
                            if [[ "$NF_TEST_KF_MODE" == render-fail && "$count" == 1 ]] \
                                || [[ "$NF_TEST_KF_MODE" == notebooks-render-fail && "$count" == 15 ]]; then
                                exit 1
                            fi
                            printf '%s\\n' 'apiVersion: v1' 'kind: ConfigMap'
                            ;;
                        apply)
                            bump "$NF_TEST_APPLY_COUNT"
                            read -r count < "$NF_TEST_APPLY_COUNT"
                            if [[ "$NF_TEST_KF_MODE" == apply-fail && "$count" == 1 ]]; then
                                printf '%s\\n' 'apply failed by test' >&2
                                exit 1
                            fi
                            if [[ "$NF_TEST_KF_MODE" == server-retry* && "$count" == 1 ]]; then
                                printf '%s\\n' 'annotation exceeds 262144 bytes: Too long' >&2
                                exit 1
                            fi
                            if [[ "$NF_TEST_KF_MODE" == server-retry-fail && "$count" == 2 ]]; then
                                printf '%s\\n' 'server-side apply failed by test' >&2
                                exit 1
                            fi
                            exit 0
                            ;;
                        *) exit 65 ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            kubectl.chmod(0o755)
            harness = root / "apply-stage.sh"
            harness.write_text(
                textwrap.dedent(
                    """\
                    set -euo pipefail
                    W="$NF_TEST_W"
                    KUBECTL="$NF_TEST_KUBECTL"
                    MANIFESTS_DIR=/test/manifests
                    log_info() { printf 'INFO %s\\n' "$*"; }
                    log_ok() { printf 'OK %s\\n' "$*"; }
                    log_warn() { printf 'WARN %s\\n' "$*"; }
                    log_err() { printf 'ERR %s\\n' "$*" >&2; }
                    """
                )
                + apply_stage,
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update(
                NF_TEST_W=str(root / "work"),
                NF_TEST_KUBECTL=str(kubectl),
                NF_TEST_KF_MODE=mode,
                NF_TEST_KUSTOMIZE_COUNT=str(root / "kustomize-count"),
                NF_TEST_APPLY_COUNT=str(root / "apply-count"),
            )
            return subprocess.run(
                ["bash", str(harness)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

    def test_render_and_apply_failures_have_accurate_counts_and_nonzero_status(self):
        """Former warning-only target failures must fail the ordered install."""
        scenarios = {
            "render-fail": "istio-crds",
            "notebooks-render-fail": "notebooks",
            "apply-fail": "istio-crds",
            "server-retry-fail": "istio-crds",
        }
        for mode, target in scenarios.items():
            with self.subTest(mode=mode):
                result = self.run_apply_stage(mode)
                output = result.stdout + result.stderr
                self.assertNotEqual(0, result.returncode, output)
                self.assertRegex(output, r"17 targets applied, 1 (?:target )?failed")
                self.assertNotIn("Kubeflow kustomize deployment complete", output)
                self.assertIn(target, output)
                self.assertNotIn(f"Applied: {target}", output)

    def test_success_and_annotation_retry_preserve_all_target_counts(self):
        """Normal applies and the intentional server-side retry still succeed."""
        for mode in ("success", "server-retry"):
            with self.subTest(mode=mode):
                result = self.run_apply_stage(mode)
                output = result.stdout + result.stderr
                self.assertEqual(0, result.returncode, output)
                self.assertIn("18 targets applied", output)
                self.assertIn("Applied: notebooks", output)
                if mode == "server-retry":
                    self.assertIn("Applied (server-side): istio-crds", output)


class KubeflowArtifactSecretCopyTest(unittest.TestCase):
    """Exercise the tenant Secret copy with fail-closed shell semantics."""

    def run_copy_stage(self, mode):
        source = (PRE_ROOT / "scripts/30_install-kubeflow.sh").read_text(encoding="utf-8")
        start = source.index("# Shared S3 credential secret")
        end = source.index("cat > /tmp/kfp-tenant-cm.yaml", start)
        copy_stage = source[start:end]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            kubectl = fake_bin / "kubectl"
            kubectl.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/bash
                    set -u
                    case "${1:-}" in
                        get)
                            if [[ "$NF_TEST_COPY_MODE" == get-fail ]]; then
                                printf '%s\\n' 'get failed by test' >&2
                                exit 1
                            fi
                            printf '%s\\n' \
                                '{' \
                                '  "metadata": {' \
                                '    "namespace": "kubeflow",' \
                                '    "resourceVersion": "1",' \
                                '    "name": "mlpipeline-minio-artifact"' \
                                '  }' \
                                '}'
                            ;;
                        apply)
                            cat > "$NF_TEST_APPLY_CAPTURE"
                            if [[ "$NF_TEST_COPY_MODE" == apply-fail ]]; then
                                printf '%s\\n' 'apply failed by test' >&2
                                exit 1
                            fi
                            ;;
                        *) exit 65 ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            kubectl.chmod(0o755)
            sed = fake_bin / "sed"
            sed.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/bash
                    set -u
                    if [[ "$NF_TEST_COPY_MODE" == sed-fail ]]; then
                        cat >/dev/null
                        exit 1
                    fi
                    exec "$NF_TEST_REAL_SED" "$@"
                    """
                ),
                encoding="utf-8",
            )
            sed.chmod(0o755)
            harness = root / "copy-stage.sh"
            harness.write_text(
                textwrap.dedent(
                    """\
                    set -euo pipefail
                    KUBECTL="$NF_TEST_KUBECTL"
                    TENANT_NS=nereidforge
                    log_ok() { printf 'OK %s\\n' "$*"; }
                    log_err() { printf 'ERR %s\\n' "$*" >&2; }
                    """
                )
                + copy_stage
                + "printf '%s\\n' 'POST_APPLY_CONTINUED'\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update(
                PATH=f"{fake_bin}:{env['PATH']}",
                NF_TEST_KUBECTL=str(kubectl),
                NF_TEST_COPY_MODE=mode,
                NF_TEST_REAL_SED=shutil.which("sed"),
                NF_TEST_APPLY_CAPTURE=str(root / "applied.json"),
            )
            result = subprocess.run(
                ["bash", str(harness)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            applied_path = root / "applied.json"
            applied = applied_path.read_text(encoding="utf-8") if applied_path.exists() else ""
            return result, applied

    def test_copy_failures_stop_post_apply_execution_without_success(self):
        for mode in ("get-fail", "sed-fail", "apply-fail"):
            with self.subTest(mode=mode):
                result, _ = self.run_copy_stage(mode)
                output = result.stdout + result.stderr
                self.assertNotEqual(result.returncode, 0, output)
                self.assertNotIn("OK mlpipeline-minio-artifact secret =>", output)
                self.assertNotIn("POST_APPLY_CONTINUED", output)

    def test_healthy_copy_succeeds_and_continues(self):
        result, applied = self.run_copy_stage("success")
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("mlpipeline-minio-artifact secret => nereidforge", output)
        self.assertIn("POST_APPLY_CONTINUED", output)
        self.assertIn('"namespace": "nereidforge"', applied)
        self.assertNotIn("resourceVersion", applied)


if __name__ == "__main__":
    unittest.main()
