# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for create-only SeaweedFS credential initialization."""

import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


PRE_ROOT = Path(__file__).resolve().parents[1]


class SeaweedFSInitializationTest(unittest.TestCase):
    """Run the real stage with an atomic, local-only Kubernetes fake."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = self.root / "state"
        self.state.mkdir()
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self.fake_python = self.root / "python"
        (self.fake_python / "botocore").mkdir(parents=True)
        self._write_kubectl()
        self._write_command_fakes()
        self._write_boto3()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_kubectl(self):
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
                import sys
                import time

                args = sys.argv[1:]
                with open(os.environ["TEST_KUBECTL_ARGS"], "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(args) + "\\n")
                state = Path(os.environ["TEST_STATE"])
                canonical = state / "canonical.json"
                process_id = os.environ["TEST_PROCESS_ID"]

                def encoded_secret(manifest):
                    return {
                        "apiVersion": "v1",
                        "kind": "Secret",
                        "metadata": manifest["metadata"],
                        "data": {
                            key: base64.b64encode(value.encode()).decode()
                            for key, value in manifest["stringData"].items()
                        },
                    }

                def store(manifest, create_only):
                    with open(state / "lock", "w", encoding="utf-8") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        if create_only and canonical.exists():
                            return False
                        rendered = json.dumps(encoded_secret(manifest))
                        pending = state / f"canonical-{process_id}.tmp"
                        pending.write_text(rendered, encoding="utf-8")
                        os.replace(pending, canonical)
                        if create_only:
                            (state / "winner").write_text(process_id, encoding="utf-8")
                        return True

                if args[:2] == ["get", "ns"]:
                    sys.exit(0)
                if args[:3] == ["get", "secret", "seaweedfs-s3-credentials"]:
                    if "-o" not in args and os.environ.get("TEST_INITIAL_BARRIER") == "1":
                        if canonical.exists():
                            sys.exit(0)
                        (state / f"checked-{process_id}").touch()
                        deadline = time.monotonic() + 10
                        while len(list(state.glob("checked-*"))) < 2:
                            if time.monotonic() >= deadline:
                                print("timed out waiting for concurrent initial checks", file=sys.stderr)
                                sys.exit(70)
                            time.sleep(0.01)
                        sys.exit(1)
                    if canonical.exists():
                        print(canonical.read_text(encoding="utf-8"))
                        sys.exit(0)
                    sys.exit(1)
                if args[:2] in (["get", "secret"], ["get", "configmap"]):
                    sys.exit(1)
                if args[:2] == ["get", "svc"]:
                    print("10.23.45.67")
                    sys.exit(0)
                if args[:2] == ["get", "sts"]:
                    print("1")
                    sys.exit(0)
                if args and args[0] == "rollout":
                    sys.exit(0)
                if args and args[0] == "run":
                    joined = " ".join(args)
                    if "sw-test2-" in joined:
                        print("SeaweedFS Filer")
                    elif "sw-test3-" in joined:
                        print("ListAllMyBucketsResult")
                    else:
                        print("SeaweedFS")
                    sys.exit(0)
                if args in (["create", "-f", "-"], ["apply", "-f", "-"]):
                    payload = sys.stdin.read()
                    if os.environ.get("TEST_CREATE_FAILURE") == "1":
                        print("forced create failure; rejected manifest follows:", file=sys.stderr)
                        print(payload, file=sys.stderr)
                        sys.exit(42)
                    created = store(json.loads(payload), create_only=args[0] == "create")
                    result = "created" if created else "already-exists"
                    Path(os.environ["TEST_CREATE_RESULT"]).write_text(result, encoding="utf-8")
                    if not created:
                        print(
                            'Error from server (AlreadyExists): secrets '
                            '"seaweedfs-s3-credentials" already exists',
                            file=sys.stderr,
                        )
                        sys.exit(1)
                    sys.exit(0)
                print("unexpected kubectl invocation: " + json.dumps(args), file=sys.stderr)
                sys.exit(64)
                """
            ),
            encoding="utf-8",
        )
        kubectl.chmod(0o755)

    def _write_command_fakes(self):
        helm = self.fake_bin / "helm"
        helm.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['TEST_HELM_ARGS'], 'a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n",
            encoding="utf-8",
        )
        helm.chmod(0o755)
        sleep = self.fake_bin / "sleep"
        sleep.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        sleep.chmod(0o755)

    def _write_boto3(self):
        (self.fake_python / "boto3.py").write_text(
            textwrap.dedent(
                """\
                import json
                import os

                class FakeS3:
                    def _record(self, operation, bucket):
                        with open(os.environ["TEST_BUCKET_CALLS"], "a", encoding="utf-8") as stream:
                            stream.write(f"{operation} {bucket}\\n")

                    def head_bucket(self, Bucket):
                        self._record("head", Bucket)
                        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

                    def list_objects_v2(self, Bucket, MaxKeys):
                        self._record("list", Bucket)
                        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

                def client(*args, **kwargs):
                    kwargs.pop("config", None)
                    with open(os.environ["TEST_BOTO_CAPTURE"], "w", encoding="utf-8") as stream:
                        json.dump(kwargs, stream)
                    return FakeS3()
                """
            ),
            encoding="utf-8",
        )
        (self.fake_python / "botocore/__init__.py").write_text("", encoding="utf-8")
        (self.fake_python / "botocore/client.py").write_text(
            "class Config:\n    def __init__(self, **kwargs):\n        self.kwargs = kwargs\n",
            encoding="utf-8",
        )

    def environment(self, process_id, credentials, barrier=False, create_failure=False):
        access_key, secret_key = credentials
        env = os.environ.copy()
        env.update(
            PATH=f"{self.fake_bin}:{env['PATH']}",
            PYTHONPATH=str(self.fake_python),
            KUBECTL=str(self.fake_bin / "kubectl"),
            SEAWEEDFS_S3_ACCESS_KEY=access_key,
            SEAWEEDFS_S3_SECRET_KEY=secret_key,
            TEST_STATE=str(self.state),
            TEST_PROCESS_ID=process_id,
            TEST_KUBECTL_ARGS=str(self.root / f"{process_id}-kubectl-args.jsonl"),
            TEST_CREATE_RESULT=str(self.root / f"{process_id}-create-result"),
            TEST_HELM_ARGS=str(self.root / f"{process_id}-helm-args.jsonl"),
            TEST_BOTO_CAPTURE=str(self.root / f"{process_id}-boto.json"),
            TEST_BUCKET_CALLS=str(self.root / f"{process_id}-bucket-calls"),
            TEST_INITIAL_BARRIER="1" if barrier else "0",
            TEST_CREATE_FAILURE="1" if create_failure else "0",
        )
        return env

    def run_installer(self, env):
        return subprocess.run(
            ["bash", str(PRE_ROOT / "scripts/10_infra/seaweedfs.sh")],
            cwd=PRE_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def assert_credentials_not_logged_or_passed(self, process_id, output, credentials):
        arguments = (self.root / f"{process_id}-kubectl-args.jsonl").read_text(encoding="utf-8")
        helm_path = self.root / f"{process_id}-helm-args.jsonl"
        helm_arguments = helm_path.read_text(encoding="utf-8") if helm_path.exists() else ""
        for credential in credentials:
            self.assertNotIn(credential, output)
            self.assertNotIn(credential, arguments)
            self.assertNotIn(credential, helm_arguments)

    def test_concurrent_creation_keeps_first_identity_for_both_processes(self):
        candidates = {
            "first": ("first-candidate-access", "first-candidate-secret"),
            "second": ("second-candidate-access", "second-candidate-secret"),
        }
        environments = {
            process_id: self.environment(process_id, credentials, barrier=True)
            for process_id, credentials in candidates.items()
        }
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = dict(pool.map(lambda item: (item[0], self.run_installer(item[1])), environments.items()))

        for process_id, result in results.items():
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("All required S3 buckets are provisioned", output)
            self.assert_credentials_not_logged_or_passed(process_id, output, candidates[process_id])

        canonical = json.loads((self.state / "canonical.json").read_text(encoding="utf-8"))
        stored = tuple(
            base64.b64decode(canonical["data"][key]).decode()
            for key in ("accessKey", "secretKey")
        )
        winner = (self.state / "winner").read_text(encoding="utf-8")
        self.assertEqual(stored, candidates[winner])
        self.assertEqual(
            {(self.root / f"{process_id}-create-result").read_text(encoding="utf-8") for process_id in candidates},
            {"created", "already-exists"},
        )
        for process_id in candidates:
            boto = json.loads((self.root / f"{process_id}-boto.json").read_text(encoding="utf-8"))
            self.assertEqual(boto["aws_access_key_id"], stored[0])
            self.assertEqual(boto["aws_secret_access_key"], stored[1])
            invocations = [
                json.loads(line)
                for line in (self.root / f"{process_id}-kubectl-args.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertNotIn("apply", [invocation[0] for invocation in invocations])
            for credentials in candidates.values():
                self.assert_credentials_not_logged_or_passed(
                    process_id, results[process_id].stdout + results[process_id].stderr, credentials
                )

    def test_other_create_errors_fail_without_exposing_the_candidate_manifest(self):
        credentials = ("rejected-candidate-access", "rejected-candidate-secret")
        result = self.run_installer(self.environment("failure", credentials, create_failure=True))
        output = result.stdout + result.stderr

        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("Could not create the SeaweedFS credentials Secret", output)
        self.assertNotIn("SeaweedFS deployment completed", output)
        self.assertFalse((self.state / "canonical.json").exists())
        self.assert_credentials_not_logged_or_passed("failure", output, credentials)
        self.assertNotIn("stringData", output)
        helm_args = self.root / "failure-helm-args.jsonl"
        self.assertFalse(helm_args.exists() and helm_args.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
