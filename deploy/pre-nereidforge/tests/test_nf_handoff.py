# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Local, fake-kubectl tests for the non-secret NF Helm handoff."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


PRE_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = PRE_ROOT / "scripts/lib/generate_nf_handoff.py"
FINALIZE = PRE_ROOT / "scripts/40_finalize-handoff.sh"
PREFLIGHT = PRE_ROOT / "scripts/40_check-nf-handoff.sh"
SENTINELS = (
    "ls-token-must-not-enter-values",
    "cvat-token-must-not-enter-values",
    "jwt-must-not-enter-values",
    "sso-must-not-enter-values",
    "admin-password-must-not-enter-values",
)


def generator_args(output):
    return [
        "python3", str(GENERATOR), "generate", "--output", str(output),
        "--pre-version", "0.0.0-dev-test", "--source-commit", "abc1234",
        "--base-domain", "example.test", "--storage-class", "nfs-csi",
        "--pg-cluster", "nereidforge-postgresql", "--pg-database", "nereidforge",
        "--pg-user", "nereidforge", "--dex-email", "admin@example.test",
    ]


class HandoffGeneratorTest(unittest.TestCase):
    def test_generation_is_deterministic_and_excludes_all_secret_sentinels(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.yaml"
            second = Path(directory) / "second.yaml"
            env = os.environ.copy()
            env.update({
                "NEREIDFORGE_LABELSTUDIO_TOKEN": SENTINELS[0],
                "NEREIDFORGE_CVAT_TOKEN": SENTINELS[1],
                "NEREIDFORGE_AUTH_JWT_SECRET": SENTINELS[2],
                "NEREIDFORGE_SSO_NS_CLIENT_SECRET": SENTINELS[3],
                "NF_ADMIN_PASSWORD": SENTINELS[4],
            })
            for output in (first, second):
                result = subprocess.run(
                    generator_args(output), env=env, text=True, capture_output=True,
                )
                self.assertEqual(0, result.returncode, result.stderr)
            content = first.read_text(encoding="utf-8")
            self.assertEqual(content, second.read_text(encoding="utf-8"))
            self.assertTrue(content.startswith("# pre-nereidforge-handoff/v1"))
            for sentinel in SENTINELS:
                self.assertNotIn(sentinel, content)
                self.assertNotIn(sentinel, result.stdout + result.stderr)
            for expected in (
                'schemaVersion: 1', 'create: false', 'createSecret: false',
                'existingSecret: "nereidforge-service-credentials"',
                'existingSecret: "pg-nereidforge-postgresql-credentials"',
                'lsTokenKey: "NEREIDFORGE_LABELSTUDIO_TOKEN"',
                'path: "/notebook-servers"', 'rotatePassword: false',
            ):
                self.assertIn(expected, content)

    def test_validation_rejects_literal_credentials_in_both_values_files(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "values.generated.yaml"
            operator = Path(directory) / "operator-values.yaml"
            subprocess.run(generator_args(generated), check=True, capture_output=True)
            operator.write_text("{}\n", encoding="utf-8")
            valid = subprocess.run(
                ["python3", str(GENERATOR), "validate", "--values", str(generated),
                 "--operator-values", str(operator)],
                text=True, capture_output=True,
            )
            self.assertEqual(0, valid.returncode, valid.stderr)
            for target, literal in (
                (operator, 'secrets:\n  auth:\n    jwt_secret: "literal-jwt"\n'),
                (operator, 'config:\n  database:\n    password: "literal-password"\n'),
                (generated, 'lsToken: "literal-token"\n'),
            ):
                original = target.read_text(encoding="utf-8")
                target.write_text(original + literal, encoding="utf-8")
                rejected = subprocess.run(
                    ["python3", str(GENERATOR), "validate", "--values", str(generated),
                     "--operator-values", str(operator)],
                    text=True, capture_output=True,
                )
                self.assertNotEqual(0, rejected.returncode)
                self.assertNotIn("literal-", rejected.stdout + rejected.stderr)
                target.write_text(original, encoding="utf-8")


class FakeClusterTest(unittest.TestCase):
    def environment(self, root, kubectl):
        env = os.environ.copy()
        env.update({
            "KUBECTL": str(kubectl),
            "PRE_ENV_FILE": str(root / "missing-site.env"),
            "PRE_HANDOFF_DIR": str(root / "handoff"),
            "DEPLOYMENT_ROOT": str(root / "deployment"),
            "PRE_RELEASE_TAG": "0.0.0-dev-test",
            "SOURCE_COMMIT": "abc1234",
            "BASE_DOMAIN": "example.test",
            "STORAGE_CLASS": "nfs-csi",
            "PG_CLUSTER_NAME": "nereidforge-postgresql",
            "PG_DB_NAME": "nereidforge",
            "PG_DB_USER": "nereidforge",
            "DEX_STATIC_EMAIL": "admin@example.test",
            "NF_TEST_LOG": str(root / "kubectl.log"),
            "NF_TEST_CAPTURE_DIR": str(root / "captures"),
            "PATH": os.environ["PATH"],
        })
        for name in ("NF_ENV_FILE", "NF_HANDOFF_DIR", "RELEASE_TAG"):
            env.pop(name, None)
        return env

    def write_kubectl(self, root, body):
        fake_bin = root / "bin"
        fake_bin.mkdir(exist_ok=True)
        kubectl = fake_bin / "kubectl"
        kubectl.write_text("#!/bin/bash\nset -u\n" + textwrap.dedent(body), encoding="utf-8")
        kubectl.chmod(0o755)
        return kubectl


class FinalizeHandoffTest(FakeClusterTest):
    def test_retained_secret_uses_ssa_and_values_never_contain_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kubectl = self.write_kubectl(root, r'''
                mkdir -p "$NF_TEST_CAPTURE_DIR"
                printf '%s\n' "$*" >> "$NF_TEST_LOG"
                case "${1:-}" in
                    apply)
                        count=0
                        [[ ! -f "$NF_TEST_CAPTURE_DIR/count" ]] || read -r count < "$NF_TEST_CAPTURE_DIR/count"
                        count=$((count+1)); printf '%s\n' "$count" > "$NF_TEST_CAPTURE_DIR/count"
                        cat > "$NF_TEST_CAPTURE_DIR/apply-$count"
                        [[ "${NF_TEST_MODE:-}" != apply-fail ]] || exit 1
                        ;;
                    get)
                        printf '%s\n' 'Q0EtQ0VSVElGSUNBVEU='
                        ;;
                    create)
                        cat >/dev/null
                        printf '%s\n' 'apiVersion: v1' 'kind: Secret'
                        ;;
                    *) exit 64 ;;
                esac
            ''')
            handoff = root / "handoff"
            handoff.mkdir()
            (handoff / "service-tokens.env").write_text(
                f"NEREIDFORGE_LABELSTUDIO_TOKEN={SENTINELS[0]}\n"
                f"NEREIDFORGE_CVAT_TOKEN={SENTINELS[1]}\n",
                encoding="utf-8",
            )
            env = self.environment(root, kubectl)
            first = subprocess.run(["bash", str(FINALIZE)], env=env, text=True, capture_output=True)
            self.assertEqual(0, first.returncode, first.stdout + first.stderr)
            values = handoff / "values.generated.yaml"
            initial = values.read_bytes()
            second = subprocess.run(["bash", str(FINALIZE)], env=env, text=True, capture_output=True)
            self.assertEqual(0, second.returncode, second.stdout + second.stderr)
            self.assertEqual(initial, values.read_bytes())
            combined_output = first.stdout + first.stderr + second.stdout + second.stderr
            for sentinel in SENTINELS[:2]:
                self.assertNotIn(sentinel, combined_output)
                self.assertNotIn(sentinel, initial.decode())
            arguments = (root / "kubectl.log").read_text(encoding="utf-8")
            self.assertIn("--field-manager=pre-nereidforge", arguments)
            self.assertIn("--force-conflicts", arguments)
            for sentinel in SENTINELS[:2]:
                self.assertNotIn(sentinel, arguments)
            applied = (root / "captures/apply-1").read_text(encoding="utf-8")
            manifest = json.loads(applied)
            self.assertEqual(
                set(manifest["stringData"]),
                {"NEREIDFORGE_LABELSTUDIO_TOKEN", "NEREIDFORGE_CVAT_TOKEN"},
            )
            self.assertNotIn("NEREIDFORGE_AUTH_JWT_SECRET", manifest["stringData"])

    def test_secret_apply_failure_does_not_generate_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kubectl = self.write_kubectl(root, r'''
                printf '%s\n' "$*" >> "$NF_TEST_LOG"
                [[ "${1:-}" == apply ]] || exit 0
                cat >/dev/null
                exit 1
            ''')
            handoff = root / "handoff"
            handoff.mkdir()
            (handoff / "service-tokens.env").write_text(
                f"NEREIDFORGE_LABELSTUDIO_TOKEN={SENTINELS[0]}\n"
                f"NEREIDFORGE_CVAT_TOKEN={SENTINELS[1]}\n",
                encoding="utf-8",
            )
            env = self.environment(root, kubectl)
            env["NF_TEST_MODE"] = "apply-fail"
            result = subprocess.run(["bash", str(FINALIZE)], env=env, text=True, capture_output=True)
            self.assertNotEqual(0, result.returncode)
            self.assertFalse((handoff / "values.generated.yaml").exists())
            for sentinel in SENTINELS[:2]:
                self.assertNotIn(sentinel, result.stdout + result.stderr)


class PreflightTest(FakeClusterTest):
    def prepare_values(self, root):
        handoff = root / "handoff"
        handoff.mkdir()
        subprocess.run(generator_args(handoff / "values.generated.yaml"), check=True, capture_output=True)
        (handoff / "operator-values.yaml").write_text("{}\n", encoding="utf-8")

    def run_preflight(self, root, mode="success"):
        kubectl = self.write_kubectl(root, r'''
            printf '%s\n' "$*" >> "$NF_TEST_LOG"
            if [[ "${1:-}" != get ]]; then exit 64; fi
            if [[ "$NF_TEST_MODE" == missing-resource && "$*" == *clusterissuer* ]]; then exit 1; fi
            if [[ "$*" == *deployment* ]]; then
                if [[ "$NF_TEST_MODE" == not-ready && "$*" == *ml-pipeline* ]]; then
                    printf '0'; exit 0
                fi
                printf '1'; exit 0
            fi
            if [[ "$*" == *secret* ]]; then
                if [[ "$NF_TEST_MODE" == missing-jwt && "$*" == *NEREIDFORGE_AUTH_JWT_SECRET* ]]; then
                    exit 0
                fi
                if [[ "$NF_TEST_MODE" == literal-jsonpath && "$*" == *nereid-internal-ca* && "$*" != *'jsonpath={.data.ca\.crt}'* ]]; then
                    exit 0
                fi
                printf 'ZmFrZQ=='; exit 0
            fi
            exit 0
        ''')
        env = self.environment(root, kubectl)
        env["NF_TEST_MODE"] = mode
        return subprocess.run(["bash", str(PREFLIGHT)], env=env, text=True, capture_output=True)

    def test_success_and_fail_closed_contracts(self):
        for mode, expected_error in (
            ("missing-resource", "Missing required resource"),
            ("not-ready", "Deployment is not ready"),
            ("missing-jwt", "NEREIDFORGE_AUTH_JWT_SECRET"),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.prepare_values(root)
                result = self.run_preflight(root, mode)
                self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
                self.assertIn(expected_error, result.stdout + result.stderr)
                self.assertNotIn("preflight passed", result.stdout + result.stderr)
                self.assertNotIn("ZmFrZQ==", result.stdout + result.stderr)

    def test_healthy_preflight_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare_values(root)
            result = self.run_preflight(root)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertIn("preflight passed", result.stdout + result.stderr)
            arguments = (root / "kubectl.log").read_text(encoding="utf-8")
            self.assertIn("get secret nereidforge-service-credentials", arguments)
            self.assertNotIn(" apply ", f" {arguments} ")
            self.assertNotIn(" create ", f" {arguments} ")
            self.assertNotIn(" patch ", f" {arguments} ")

    def test_dotted_secret_key_is_read_as_a_literal_jsonpath_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare_values(root)
            result = self.run_preflight(root, "literal-jsonpath")
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertIn("preflight passed", result.stdout + result.stderr)
            arguments = (root / "kubectl.log").read_text(encoding="utf-8")
            self.assertIn(r"jsonpath={.data.ca\.crt}", arguments)

    def test_missing_operator_overlay_fails_before_cluster_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare_values(root)
            (root / "handoff/operator-values.yaml").unlink()
            result = self.run_preflight(root)
            self.assertNotEqual(0, result.returncode)
            self.assertFalse((root / "kubectl.log").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
