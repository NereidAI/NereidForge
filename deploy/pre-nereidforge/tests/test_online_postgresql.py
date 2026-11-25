# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for retained PostgreSQL credentials and preflight."""

import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
import textwrap
import unittest


PRE_ROOT = Path(__file__).resolve().parents[1]
SECRET_NAME = "pg-nereidforge-postgresql-credentials"
MUTATING_VERBS = {"apply", "annotate", "create", "delete", "patch", "run", "wait"}


class PostgreSQLPreflightTest(unittest.TestCase):
    """Run the real operator stage against local recording command fakes."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self.state = self.root / "state"
        self.state.mkdir()
        self.kubectl_args = self.root / "kubectl-args.jsonl"
        self.helm_args = self.root / "helm-args.jsonl"
        self.openssl_args = self.root / "openssl-args.jsonl"
        self._write_fakes()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_fakes(self):
        kubectl = self.fake_bin / "kubectl"
        kubectl.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import base64
                import fcntl
                import json
                import os
                from pathlib import Path
                import subprocess
                import sys
                import time

                args = sys.argv[1:]
                with open(os.environ["TEST_KUBECTL_ARGS"], "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(args) + "\\n")
                state = Path(os.environ["TEST_STATE"])
                secret_path = state / "pg-secret.json"
                process_id = os.environ.get("TEST_PROCESS_ID", "single")

                def wait_for_pair(marker):
                    (state / f"{marker}-{process_id}").touch()
                    deadline = time.monotonic() + 10
                    while len(list(state.glob(f"{marker}-*"))) < 2:
                        if time.monotonic() >= deadline:
                            print(f"timed out waiting for {marker}", file=sys.stderr)
                            sys.exit(70)
                        time.sleep(0.01)

                if args[:3] == ["get", "secret", "pg-nereidforge-postgresql-credentials"]:
                    if not secret_path.exists():
                        if os.environ.get("TEST_INITIAL_BARRIER") == "1":
                            wait_for_pair("initial-checked")
                        print("Error from server (NotFound): secret not found", file=sys.stderr)
                        sys.exit(1)
                    resource = json.loads(secret_path.read_text(encoding="utf-8"))
                    if "jsonpath={.data.password}" in args:
                        print(base64.b64decode(resource["data"]["password"]).decode(), end="")
                    elif "jsonpath={.data.username}" in args:
                        print(base64.b64decode(resource["data"]["username"]).decode(), end="")
                    else:
                        print(json.dumps(resource))
                    sys.exit(0)
                if args[:2] in (["get", "secret"], ["get", "configmap"]):
                    print("Error from server (NotFound): credential source not found", file=sys.stderr)
                    sys.exit(1)
                if args == ["create", "-f", "-"]:
                    manifest = json.loads(sys.stdin.read())
                    Path(os.environ["TEST_CREATE_CAPTURE"]).write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    if os.environ.get("TEST_CREATE_BARRIER") == "1":
                        wait_for_pair("create-checked")
                    with open(state / "lock", "w", encoding="utf-8") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        result_path = Path(os.environ["TEST_CREATE_RESULT"])
                        if secret_path.exists():
                            (state / f"already-exists-secret-{process_id}.json").write_bytes(
                                secret_path.read_bytes()
                            )
                            result_path.write_text("already-exists", encoding="utf-8")
                            print(
                                'Error from server (AlreadyExists): secrets "'
                                'pg-nereidforge-postgresql-credentials" already exists',
                                file=sys.stderr,
                            )
                            sys.exit(1)
                        resource = {
                            "apiVersion": "v1",
                            "kind": "Secret",
                            "metadata": manifest["metadata"],
                            "type": manifest["type"],
                            "data": {
                                key: base64.b64encode(value.encode()).decode()
                                for key, value in manifest["stringData"].items()
                            },
                        }
                        pending = state / f"pg-secret-{process_id}.tmp"
                        pending.write_text(json.dumps(resource), encoding="utf-8")
                        os.replace(pending, secret_path)
                        result_path.write_text("created", encoding="utf-8")
                        (state / "winner").write_text(process_id, encoding="utf-8")
                    sys.exit(0)
                if args[:2] == ["get", "ns"]:
                    sys.exit(1)
                if args[:2] == ["get", "deploy"]:
                    sys.exit(0)
                if args[0] == "run":
                    password = sys.stdin.buffer.read()
                    Path(os.environ["TEST_PROBE_INPUT_CAPTURE"]).write_bytes(password)
                    command = args[args.index("--") + 1:]
                    result = subprocess.run(command, input=password, check=False)
                    sys.exit(result.returncode)
                sys.exit(0)
                """
            ),
            encoding="utf-8",
        )
        kubectl.chmod(0o755)

        helm = self.fake_bin / "helm"
        helm.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['TEST_HELM_ARGS'], 'a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n",
            encoding="utf-8",
        )
        helm.chmod(0o755)

        psql = self.fake_bin / "psql"
        psql.write_text(
            "#!/usr/bin/env python3\n"
            "import os\n"
            "from pathlib import Path\n"
            "Path(os.environ['TEST_REMOTE_PASSWORD_CAPTURE']).write_bytes(\n"
            "    os.environ['PGPASSWORD'].encode('utf-8')\n"
            ")\n",
            encoding="utf-8",
        )
        psql.chmod(0o755)

        openssl = self.fake_bin / "openssl"
        openssl.write_text(
            "#!/bin/bash\n"
            "printf '%s\\n' \"$*\" >> \"$TEST_OPENSSL_ARGS\"\n"
            "exec \"$REAL_OPENSSL\" \"$@\"\n",
            encoding="utf-8",
        )
        openssl.chmod(0o755)

        python3 = self.fake_bin / "python3"
        python3.write_text(
            "#!/bin/bash\n"
            "if [[ \"$*\" == '-c import boto3' ]]; then\n"
            "  [[ \"${TEST_BOTO3_AVAILABLE:-1}\" == 1 ]]\n"
            "  exit $?\n"
            "fi\n"
            "exec \"$REAL_PYTHON\" \"$@\"\n",
            encoding="utf-8",
        )
        python3.chmod(0o755)

    def environment(self, password=None):
        env = os.environ.copy()
        env.update(
            PATH=f"{self.fake_bin}:{env['PATH']}",
            KUBECTL=str(self.fake_bin / "kubectl"),
            PRE_ENV_FILE=str(self.root / "absent-site.env"),
            DEPLOYMENT_ROOT=str(self.root / "deployment"),
            PRE_RELEASE_TAG="0.0.1",
            TEST_STATE=str(self.state),
            TEST_PROCESS_ID="single",
            TEST_CREATE_CAPTURE=str(self.root / "created.json"),
            TEST_CREATE_RESULT=str(self.root / "create-result"),
            TEST_INITIAL_BARRIER="0",
            TEST_CREATE_BARRIER="0",
            TEST_KUBECTL_ARGS=str(self.kubectl_args),
            TEST_HELM_ARGS=str(self.helm_args),
            TEST_OPENSSL_ARGS=str(self.openssl_args),
            TEST_PROBE_INPUT_CAPTURE=str(self.root / "probe-input"),
            TEST_REMOTE_PASSWORD_CAPTURE=str(self.root / "remote-password"),
            REAL_OPENSSL=shutil.which("openssl"),
            REAL_PYTHON=shutil.which("python3"),
            TEST_BOTO3_AVAILABLE="1",
        )
        for name in (
            "PG_ADMIN_PASSWORD",
            "NF_ENV_FILE",
            "SEAWEEDFS_S3_ACCESS_KEY",
            "SEAWEEDFS_S3_SECRET_KEY",
        ):
            env.pop(name, None)
        if password is not None:
            env["PG_ADMIN_PASSWORD"] = password
        return env

    def invocations(self, path):
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def seed_secret(self, username="nereidforge", password="retained-pg-password"):
        resource = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": SECRET_NAME, "namespace": "nereidforge"},
            "type": "kubernetes.io/basic-auth",
            "data": {
                "username": base64.b64encode(username.encode()).decode(),
                "password": base64.b64encode(password.encode()).decode(),
            },
        }
        (self.state / "pg-secret.json").write_text(json.dumps(resource), encoding="utf-8")

    def assert_no_mutation_or_helm(self):
        invocations = self.invocations(self.kubectl_args)
        self.assertFalse(
            [args for args in invocations if args and args[0] in MUTATING_VERBS],
            invocations,
        )
        self.assertFalse(self.helm_args.exists() and self.helm_args.read_text(encoding="utf-8"))
        self.assertFalse(self.openssl_args.exists() and self.openssl_args.read_text(encoding="utf-8"))

    def test_direct_stage_rejects_missing_first_use_password_before_mutation(self):
        result = subprocess.run(
            ["bash", str(PRE_ROOT / "scripts/10_infra/postgresql.sh")],
            cwd=PRE_ROOT,
            env=self.environment(),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("PG_ADMIN_PASSWORD", output)
        self.assert_no_mutation_or_helm()

    def test_infra_entry_rejects_missing_first_use_password_before_namespace(self):
        result = subprocess.run(
            ["bash", str(PRE_ROOT / "scripts/10_infra.sh")],
            cwd=PRE_ROOT,
            env=self.environment(),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("PG_ADMIN_PASSWORD", output)
        self.assert_no_mutation_or_helm()

    def test_first_use_creates_retained_secret_without_exposing_password(self):
        password = 'first-pg-keep-":#{}'
        result = subprocess.run(
            ["bash", str(PRE_ROOT / "scripts/10_infra/postgresql.sh")],
            cwd=PRE_ROOT,
            env=self.environment(password=password),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        created_path = self.root / "created.json"
        self.assertTrue(created_path.exists(), output)
        created = json.loads(created_path.read_text(encoding="utf-8"))
        self.assertEqual(created["kind"], "Secret")
        self.assertEqual(created["metadata"]["name"], SECRET_NAME)
        self.assertEqual(created["metadata"]["annotations"]["helm.sh/resource-policy"], "keep")
        self.assertEqual(created["type"], "kubernetes.io/basic-auth")
        self.assertEqual(created["stringData"], {"username": "nereidforge", "password": password})

        kubectl_text = self.kubectl_args.read_text(encoding="utf-8")
        helm_text = self.helm_args.read_text(encoding="utf-8")
        self.assertNotIn(password, output)
        self.assertNotIn(password, kubectl_text)
        self.assertNotIn(password, helm_text)
        self.assertNotIn("bootstrap.password", helm_text)

    def test_infra_entry_rejects_missing_first_use_s3_input_before_namespace(self):
        env = self.environment(password="retained-postgresql-candidate")
        result = subprocess.run(
            ["bash", str(PRE_ROOT / "scripts/10_infra.sh")],
            cwd=PRE_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("SEAWEEDFS_S3_ACCESS_KEY", output)
        self.assertIn("SEAWEEDFS_S3_SECRET_KEY", output)
        self.assert_no_mutation_or_helm()

    def test_infra_entry_rejects_missing_boto3_before_namespace(self):
        env = self.environment(password="retained-postgresql-candidate")
        env.update(
            SEAWEEDFS_S3_ACCESS_KEY="first-s3-access",
            SEAWEEDFS_S3_SECRET_KEY="first-s3-secret",
            TEST_BOTO3_AVAILABLE="0",
        )
        result = subprocess.run(
            ["bash", str(PRE_ROOT / "scripts/10_infra.sh")],
            cwd=PRE_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("boto3", output)
        self.assert_no_mutation_or_helm()

    def test_existing_secret_allows_missing_site_password_without_recreation(self):
        retained = "retained-pg-password-without-site-copy"
        self.seed_secret(password=retained)
        result = subprocess.run(
            ["bash", str(PRE_ROOT / "scripts/10_infra/postgresql.sh")],
            cwd=PRE_ROOT,
            env=self.environment(),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("Reusing the existing PostgreSQL credentials Secret", output)
        self.assertFalse((self.root / "created.json").exists())
        self.assertNotIn(retained, output)
        self.assertNotIn(retained, self.kubectl_args.read_text(encoding="utf-8"))
        self.assertNotIn(retained, self.helm_args.read_text(encoding="utf-8"))

    def test_conflicting_site_password_fails_before_secret_or_helm_mutation(self):
        self.seed_secret(password="retained-pg-password")
        result = subprocess.run(
            ["bash", str(PRE_ROOT / "scripts/10_infra/postgresql.sh")],
            cwd=PRE_ROOT,
            env=self.environment(password="conflicting-site-password"),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("differs from the existing PostgreSQL Secret", output)
        self.assertNotIn("retained-pg-password", output)
        self.assertNotIn("conflicting-site-password", output)
        self.assert_no_mutation_or_helm()

    def test_multiline_password_round_trips_through_secret_and_probe(self):
        password = "first line\nsecond ' \"$`\\ line\n\n"
        self.seed_secret(password=password)
        result = subprocess.run(
            ["bash", str(PRE_ROOT / "scripts/10_infra/postgresql.sh")],
            cwd=PRE_ROOT,
            env=self.environment(password=password),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertEqual(0, result.returncode, output)
        self.assertEqual(
            password.encode(), (self.root / "probe-input").read_bytes()
        )
        self.assertEqual(
            password.encode(), (self.root / "remote-password").read_bytes()
        )
        self.assertNotIn(password, output)
        self.assertNotIn(password, self.kubectl_args.read_text(encoding="utf-8"))
        self.assertNotIn(password, self.helm_args.read_text(encoding="utf-8"))

    def test_nul_in_existing_secret_fails_before_mutation(self):
        self.seed_secret(password="retained\x00secret")
        result = subprocess.run(
            ["bash", str(PRE_ROOT / "scripts/10_infra/postgresql.sh")],
            cwd=PRE_ROOT,
            env=self.environment(),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertNotEqual(0, result.returncode, output)
        self.assertIn("Existing PostgreSQL credentials Secret is malformed", output)
        self.assertNotIn("retained", output)
        self.assert_no_mutation_or_helm()

    def test_credential_helper_rejects_nul_in_new_or_retained_credentials(self):
        helper = runpy.run_path(str(PRE_ROOT / "scripts/lib/pg_credentials.py"))
        for field in ("username", "password"):
            with self.subTest(operation="manifest", field=field):
                values = {"username": "nereidforge", "password": "candidate"}
                values[field] += "\x00"
                with self.assertRaises(helper["CredentialError"]):
                    helper["secret_manifest"](
                        SECRET_NAME, "nereidforge", values["username"], values["password"]
                    )
            with self.subTest(operation="exports", field=field):
                values = {"username": "nereidforge", "password": "retained"}
                values[field] += "\x00"
                resource = {
                    "data": {
                        key: base64.b64encode(value.encode()).decode()
                        for key, value in values.items()
                    }
                }
                with self.assertRaises(helper["CredentialError"]):
                    helper["credentials_from_secret"](resource)

    def test_concurrent_conflicting_creator_loses_without_secret_or_helm_mutation(self):
        passwords = {"a": "race-password-a", "b": "race-password-b"}

        def run_stage(process_id):
            env = self.environment(password=passwords[process_id])
            env.update(
                TEST_PROCESS_ID=process_id,
                TEST_CREATE_CAPTURE=str(self.root / f"created-{process_id}.json"),
                TEST_CREATE_RESULT=str(self.root / f"create-result-{process_id}"),
                TEST_INITIAL_BARRIER="1",
                TEST_CREATE_BARRIER="1",
                TEST_KUBECTL_ARGS=str(self.root / f"kubectl-{process_id}.jsonl"),
                TEST_HELM_ARGS=str(self.root / f"helm-{process_id}.jsonl"),
                TEST_OPENSSL_ARGS=str(self.root / f"openssl-{process_id}.jsonl"),
                TEST_PROBE_INPUT_CAPTURE=str(self.root / f"probe-input-{process_id}"),
                TEST_REMOTE_PASSWORD_CAPTURE=str(self.root / f"remote-password-{process_id}"),
            )
            return subprocess.run(
                ["bash", str(PRE_ROOT / "scripts/10_infra/postgresql.sh")],
                cwd=PRE_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=40,
                check=False,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                process_id: executor.submit(run_stage, process_id)
                for process_id in passwords
            }
            results = {
                process_id: future.result()
                for process_id, future in futures.items()
            }

        winner = (self.state / "winner").read_text(encoding="utf-8")
        loser = "b" if winner == "a" else "a"
        winner_output = results[winner].stdout + results[winner].stderr
        loser_output = results[loser].stdout + results[loser].stderr
        self.assertEqual(0, results[winner].returncode, winner_output)
        self.assertNotEqual(0, results[loser].returncode, loser_output)
        self.assertIn("Reusing the concurrently created", loser_output)
        self.assertIn("differs from the existing PostgreSQL Secret", loser_output)
        self.assertEqual(
            "created", (self.root / f"create-result-{winner}").read_text(encoding="utf-8")
        )
        self.assertEqual(
            "already-exists",
            (self.root / f"create-result-{loser}").read_text(encoding="utf-8"),
        )

        secret_path = self.state / "pg-secret.json"
        snapshot = self.state / f"already-exists-secret-{loser}.json"
        self.assertEqual(snapshot.read_bytes(), secret_path.read_bytes())
        retained = json.loads(secret_path.read_text(encoding="utf-8"))
        self.assertEqual(
            passwords[winner], base64.b64decode(retained["data"]["password"]).decode()
        )
        loser_invocations = self.invocations(self.root / f"kubectl-{loser}.jsonl")
        self.assertFalse(
            [
                args for args in loser_invocations
                if args and args[0] in {"annotate", "delete", "patch", "run", "wait"}
            ],
            loser_invocations,
        )
        loser_helm = self.root / f"helm-{loser}.jsonl"
        self.assertFalse(loser_helm.exists() and loser_helm.read_text(encoding="utf-8"))
        for result in results.values():
            output = result.stdout + result.stderr
            self.assertNotIn(passwords["a"], output)
            self.assertNotIn(passwords["b"], output)


@unittest.skipUnless(shutil.which("helm"), "helm is required for chart rendering tests")
class PostgreSQLChartRenderingTest(unittest.TestCase):
    def test_chart_references_external_secret_without_rendering_credentials(self):
        sentinel = "legacy-bootstrap-password-must-not-render"
        with tempfile.TemporaryDirectory() as temporary:
            chart = Path(temporary) / "postgresql"
            shutil.copytree(PRE_ROOT / "charts/postgresql", chart)
            dependency = chart / "charts/cloudnative-pg"
            dependency.mkdir(parents=True)
            (dependency / "Chart.yaml").write_text(
                "apiVersion: v2\nname: cloudnative-pg\nversion: 0.29.0\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "helm",
                    "template",
                    "postgresql",
                    str(chart),
                    "--namespace",
                    "nereidforge",
                    "--set",
                    "cnpg.enabled=false",
                    "--set",
                    f"bootstrap.password={sentinel}",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("kind: Cluster", result.stdout)
        self.assertIn("name: pg-nereidforge-credentials", result.stdout)
        self.assertNotIn("kind: Secret", result.stdout)
        self.assertNotIn("stringData", result.stdout)
        self.assertNotIn(sentinel, result.stdout)


if __name__ == "__main__":
    unittest.main()
