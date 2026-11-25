# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Local regression tests for NF frontend routing and fail-closed verification."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]


class FrontendRoutingTest(unittest.TestCase):
    """Exercise the installer with a recording Helm stub."""

    def test_frontend_preserves_chart_notebook_proxy_route(self):
        """The explicit path list retains every API route, including WebSockets."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script_dir = root / "scripts/40_nereidforge"
            lib_dir = root / "scripts/lib"
            chart_dir = root / "charts/nereidforge-frontend"
            fake_bin = root / "bin"
            script_dir.mkdir(parents=True)
            lib_dir.mkdir(parents=True)
            chart_dir.mkdir(parents=True)
            fake_bin.mkdir()
            script = script_dir / "frontend.sh"
            shutil.copy2(ROOT / "scripts/40_nereidforge/frontend.sh", script)
            (lib_dir / "common.sh").write_text(
                "log_info() { :; }\n"
                "log_ok() { :; }\n"
                "validate_required() { return 0; }\n",
                encoding="utf-8",
            )
            (lib_dir / "config.sh").write_text(
                ": \"${FRONTEND_HOST:=forge.example.test}\"\n"
                ": \"${NF_FRONTEND_IMAGE:=ghcr.io/nereidai/nereidforge-frontend:test}\"\n",
                encoding="utf-8",
            )
            args_file = root / "helm-args"
            helm = fake_bin / "helm"
            helm.write_text(
                "#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$HELM_ARGS_FILE\"\n",
                encoding="utf-8",
            )
            helm.chmod(0o755)
            env = os.environ.copy()
            env.update(PATH=f"{fake_bin}:{env['PATH']}", HELM_ARGS_FILE=str(args_file))

            result = subprocess.run(
                ["bash", str(script)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            args = args_file.read_text(encoding="utf-8").splitlines()
            settings = [args[index + 1] for index, arg in enumerate(args) if arg == "--set"]
            routes = {}
            for setting in settings:
                match = re.fullmatch(
                    r"ingress\.paths\[(\d+)]\.(path|serviceName|servicePort)=(.*)", setting
                )
                if match:
                    index, field, value = match.groups()
                    routes.setdefault(int(index), {})[field] = value
            actual = [routes[index] for index in sorted(routes)]
            self.assertEqual(
                [
                    {"path": "/", "serviceName": "nereidforge-frontend", "servicePort": "80"},
                    {"path": "/api", "serviceName": "nereidforge-api", "servicePort": "8080"},
                    {
                        "path": "/auth/ns-callback",
                        "serviceName": "nereidforge-api",
                        "servicePort": "8080",
                    },
                    {
                        "path": "/v1/integration/ns",
                        "serviceName": "nereidforge-api",
                        "servicePort": "8080",
                    },
                    {
                        "path": "/notebook-servers",
                        "serviceName": "nereidforge-api",
                        "servicePort": "8080",
                    },
                ],
                actual,
            )
            self.assertIn("ingress.host=forge.example.test", settings)
            self.assertIn("env[1].value=https://forge.example.test/api", settings)
            self.assertIn("env[2].value=wss://forge.example.test", settings)


class NereidForgeVerificationTest(unittest.TestCase):
    """Run the real verifier without cluster access or real credentials."""

    def run_check(self, overrides=None, admin_password="test-password"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script_dir = root / "60_verify"
            lib_dir = root / "lib"
            fake_bin = root / "bin"
            handoff_dir = root / "handoff"
            script_dir.mkdir()
            lib_dir.mkdir()
            fake_bin.mkdir()
            handoff_dir.mkdir()
            script = script_dir / "nf-check.sh"
            shutil.copy2(ROOT / "scripts/60_verify/nf-check.sh", script)
            (lib_dir / "common.sh").write_text(
                textwrap.dedent(
                    """\
                    log_info() { printf 'INFO %s\\n' "$*"; }
                    log_ok() { printf 'OK %s\\n' "$*"; }
                    log_warn() { printf 'WARN %s\\n' "$*"; }
                    log_err() { printf 'ERR %s\\n' "$*" >&2; }
                    wait_deployment_ready() {
                        case "$1" in
                            nereidforge-api) return "${NF_TEST_API_DEPLOYMENT_EXIT:-0}" ;;
                            nereidforge-frontend) return "${NF_TEST_FRONTEND_DEPLOYMENT_EXIT:-0}" ;;
                            *) return 64 ;;
                        esac
                    }
                    IMG_BUSYBOX=busybox:test
                    IMG_CURL=curl:test
                    NF_HANDOFF_DIR="$NF_TEST_HANDOFF_DIR"
                    """
                ),
                encoding="utf-8",
            )
            (lib_dir / "config.sh").write_text(":\n", encoding="utf-8")
            kubectl = fake_bin / "kubectl"
            kubectl.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/bash
                    set -u
                    case "${1:-}" in
                        get)
                            if [[ "$*" == *"secret nereidforge-admin-credentials"* ]]; then
                                if [[ "${NF_TEST_ADMIN_SECRET:-0}" == 1 ]]; then
                                    printf '%s' 'dGVzdC1wYXNzd29yZA=='
                                else
                                    exit 1
                                fi
                            elif [[ "${NF_TEST_API_POD_EXIT:-0}" != 0 ]]; then
                                exit "$NF_TEST_API_POD_EXIT"
                            else
                                printf '%s\\n' "${NF_TEST_API_POD_OUT-api-pod}"
                            fi
                            ;;
                        exec)
                            if [[ "$*" == *"curl -s -o /dev/null -w %{http_code}"* ]]; then
                                case "$*" in
                                    *"/api/v1/training/jobs"*) probe=JOBS ;;
                                    *"/v1/integration/ns/tenants"*) probe=INT ;;
                                    *) exit 66 ;;
                                esac
                                exit_var="NF_TEST_${probe}_EXIT"
                                out_var="NF_TEST_${probe}_OUT"
                                exec_out_var="NF_TEST_EXEC_${probe}_OUT"
                                out="${!out_var-}"
                                if [[ -v $exec_out_var ]]; then
                                    out="${!exec_out_var}"
                                fi
                                if [[ "${!exit_var:-0}" != 0 ]]; then
                                    exit "${!exit_var}"
                                fi
                                printf '%s\\n' "$out"
                            else
                                if [[ "${NF_TEST_LOGIN_EXIT:-0}" != 0 ]]; then
                                    exit "$NF_TEST_LOGIN_EXIT"
                                fi
                                if [[ "${NF_TEST_LOGIN_CONTRACT_EXIT:-0}" != 0 ]]; then
                                    exit "$NF_TEST_LOGIN_CONTRACT_EXIT"
                                fi
                                printf '%s\\n' "${NF_TEST_LOGIN_OUT-LOGIN_OK}"
                            fi
                            ;;
                        run)
                            case "${4:-}" in
                                api-verify-*) probe=HEALTH ;;
                                ready-verify-*) probe=READY ;;
                                sso-verify-*) probe=SSO ;;
                                train-verify-*) probe=JOBS ;;
                                int-verify-*) probe=INT ;;
                                fe-verify-*) probe=FE ;;
                                *) exit 64 ;;
                            esac
                            exit_var="NF_TEST_${probe}_EXIT"
                            out_var="NF_TEST_${probe}_OUT"
                            run_out_var="NF_TEST_RUN_${probe}_OUT"
                            out="${!out_var-}"
                            if [[ -v $run_out_var ]]; then
                                out="${!run_out_var}"
                            fi
                            if [[ "${!exit_var:-0}" != 0 ]]; then
                                exit "${!exit_var}"
                            fi
                            printf '%s\\n' "$out"
                            ;;
                        *) exit 65 ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            kubectl.chmod(0o755)
            env = os.environ.copy()
            env.update(
                PATH=f"{fake_bin}:{env['PATH']}",
                NF_TEST_HANDOFF_DIR=str(handoff_dir),
                NF_ADMIN_PASSWORD=admin_password,
                NF_TEST_HEALTH_OUT='{"status":"ok"}',
                NF_TEST_READY_OUT='{"status":"ready"}',
                NF_TEST_LOGIN_OUT="LOGIN_OK",
                NF_TEST_SSO_OUT=(
                    '{"enabled":true,"mode":"ns",'
                    '"logout_url":"https://ns.example.test/logout"}'
                ),
                NF_TEST_JOBS_OUT="401",
                NF_TEST_INT_OUT="401401",
                NF_TEST_FE_OUT="<!DOCTYPE html>",
            )
            if overrides:
                env.update({key: str(value) for key, value in overrides.items()})
            return subprocess.run(
                ["bash", str(script), "nereidforge"],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

    def assert_failed_check(self, overrides):
        result = self.run_check(overrides)
        output = result.stdout + result.stderr
        self.assertNotEqual(0, result.returncode, output)
        self.assertNotIn("business layer verification done", output)
        return output

    def test_healthy_contracts_accept_authenticated_training_endpoint(self):
        """Exact and known doubled codes pass; a 401 challenge is reachable."""
        scenarios = (
            {},
            {"NF_TEST_JOBS_OUT": "200", "NF_TEST_INT_OUT": "401"},
            {"NF_TEST_JOBS_OUT": "200200", "NF_TEST_INT_OUT": "401401"},
            {"NF_TEST_JOBS_OUT": "401401", "NF_TEST_INT_OUT": "401"},
        )
        for overrides in scenarios:
            with self.subTest(overrides=overrides):
                result = self.run_check(overrides)
                output = result.stdout + result.stderr
                self.assertEqual(0, result.returncode, output)
                self.assertIn("Training jobs endpoint", output)
                self.assertIn("verification done", output)

    def test_http_contracts_ignore_obsolete_kubectl_run_cleanup_output(self):
        """Captured run --rm output must not hide otherwise healthy HTTP 401s."""
        result = self.run_check(
            {
                # k3s v1.36.2+k3s1 returned both of these real probe outputs:
                # one lost curl's status entirely, and one appended deletion text.
                "NF_TEST_RUN_JOBS_OUT": (
                    'pod "train-verify-1358715" deleted from nereidforge namespace'
                ),
                "NF_TEST_RUN_INT_OUT": (
                    '401401pod "int-verify-1358715" deleted from nereidforge namespace'
                ),
                # The stable kubectl exec transport returns only curl's status.
                "NF_TEST_JOBS_OUT": "401",
                "NF_TEST_INT_OUT": "401",
            }
        )
        output = result.stdout + result.stderr
        self.assertEqual(0, result.returncode, output)
        self.assertIn("Training jobs endpoint reachable", output)
        self.assertIn("NS integration tenant-list secret-gated", output)
        self.assertIn("verification done", output)

    def test_dependency_and_resource_failures_return_nonzero(self):
        scenarios = {
            "API deployment": {"NF_TEST_API_DEPLOYMENT_EXIT": 1},
            "frontend deployment": {"NF_TEST_FRONTEND_DEPLOYMENT_EXIT": 1},
            "API pod lookup": {"NF_TEST_API_POD_EXIT": 1},
            "missing API pod": {"NF_TEST_API_POD_OUT": ""},
        }
        for name, overrides in scenarios.items():
            with self.subTest(name=name):
                self.assert_failed_check(overrides)

    def test_core_probe_command_and_contract_failures_return_nonzero(self):
        """Every former warning-only acceptance gate now fails closed."""
        scenarios = {
            "health command": {"NF_TEST_HEALTH_EXIT": 1},
            "health contract": {"NF_TEST_HEALTH_OUT": '{"status":"error"}'},
            "readiness command": {"NF_TEST_READY_EXIT": 1},
            "readiness contract": {"NF_TEST_READY_OUT": '{"status":"not-ready"}'},
            "SSO command": {"NF_TEST_SSO_EXIT": 1},
            "SSO status": {"NF_TEST_SSO_OUT": '{"enabled":false}'},
            "training command": {"NF_TEST_JOBS_EXIT": 1},
            "training unreachable": {"NF_TEST_JOBS_OUT": "000"},
            "training server error": {"NF_TEST_JOBS_OUT": "500"},
            "training ambiguous status": {"NF_TEST_JOBS_OUT": "200401"},
            "integration command": {"NF_TEST_INT_EXIT": 1},
            "integration unreachable": {"NF_TEST_INT_OUT": "000"},
            "integration unprotected": {"NF_TEST_INT_OUT": "200"},
            "integration ambiguous status": {"NF_TEST_INT_OUT": "200401"},
            "frontend command": {"NF_TEST_FE_EXIT": 1},
            "frontend contract": {"NF_TEST_FE_OUT": "upstream error"},
            "login command": {"NF_TEST_LOGIN_EXIT": 1},
            "login rejected": {"NF_TEST_LOGIN_CONTRACT_EXIT": 1},
        }
        for name, overrides in scenarios.items():
            with self.subTest(name=name):
                self.assert_failed_check(overrides)

    def test_missing_handoff_credential_is_a_failed_acceptance_gate(self):
        """An absent administrator password cannot degrade to a skipped check."""
        result = self.run_check(admin_password="")
        output = result.stdout + result.stderr
        self.assertNotEqual(0, result.returncode, output)
        self.assertIn("credential", output.lower())
        self.assertNotIn("business layer verification done", output)

    def test_admin_password_can_come_from_the_retained_helm_secret(self):
        """Direct Helm acceptance needs no legacy local credential file."""
        result = self.run_check({"NF_TEST_ADMIN_SECRET": 1}, admin_password="")
        output = result.stdout + result.stderr
        self.assertEqual(0, result.returncode, output)
        self.assertIn("admin login OK", output)
        self.assertNotIn("test-password", output)

    def test_login_success_does_not_depend_on_stdout_marker(self):
        """A lost stdout marker cannot reject a successful in-pod validation."""
        result = self.run_check({"NF_TEST_LOGIN_OUT": ""})
        output = result.stdout + result.stderr
        self.assertEqual(0, result.returncode, output)
        self.assertIn("admin login OK", output)
        self.assertIn("verification done", output)

    def test_login_failure_does_not_leak_sensitive_response(self):
        """Failed login output never exposes tokens or the full response body."""
        token = "do-not-log-access-token"
        output = self.assert_failed_check(
            {
                "NF_TEST_LOGIN_CONTRACT_EXIT": 1,
                "NF_TEST_LOGIN_OUT": f'{{"access_token":"{token}"}}\nLOGIN_FAIL',
            }
        )
        self.assertNotIn(token, output)
        self.assertNotIn("access_token", output)


class LoginScriptContractTest(unittest.TestCase):
    """Execute the real in-pod login probe with a local curl fake."""

    def run_login(self, status, body, curl_exit=0):
        source = (ROOT / "scripts/60_verify/nf-check.sh").read_text(encoding="utf-8")
        login_script = source.split("LOGIN_SCRIPT='", 1)[1].split("\n'\nif [[", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            fake_bin = Path(tmp) / "bin"
            fake_bin.mkdir()
            curl = fake_bin / "curl"
            curl.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/bash
                    if [[ "$*" == *"/api/v1/auth/captcha"* ]]; then
                        printf '%s\n' '{"captcha_id":"captcha-id","expression":"1 + 2"}'
                        exit 0
                    fi
                    printf '%s\n' "$NF_TEST_LOGIN_BODY"
                    printf '%s\n' "$NF_TEST_LOGIN_STATUS"
                    exit "$NF_TEST_CURL_EXIT"
                    """
                ),
                encoding="utf-8",
            )
            curl.chmod(0o755)
            env = os.environ.copy()
            env.update(
                PATH=f"{fake_bin}:{env['PATH']}",
                NF_ADMIN_PASSWORD="test-password",
                NF_TEST_LOGIN_STATUS=str(status),
                NF_TEST_LOGIN_BODY=body,
                NF_TEST_CURL_EXIT=str(curl_exit),
            )
            return subprocess.run(
                ["sh", "-c", login_script],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

    def test_login_requires_http_200_and_nonempty_token_without_leaking_body(self):
        token = "token-shaped-error-body-must-not-pass"
        scenarios = (
            (200, '{"access_token":"valid-token"}', 0, 0),
            (401, f'{{"access_token":"{token}"}}', 0, 1),
            (500, f'{{"access_token":"{token}"}}', 0, 1),
            (200, '{"error":"missing token"}', 0, 1),
            (0, f'{{"access_token":"{token}"}}', 7, 1),
        )
        for status, body, curl_exit, expected_exit in scenarios:
            with self.subTest(status=status, curl_exit=curl_exit, body=body):
                result = self.run_login(status, body, curl_exit)
                self.assertEqual(expected_exit, result.returncode, result.stdout + result.stderr)
                self.assertEqual("", result.stdout)
                self.assertNotIn(token, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
