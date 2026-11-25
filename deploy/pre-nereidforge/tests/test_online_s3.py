# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Self-contained regression tests for pre-NereidForge SeaweedFS S3 wiring."""

import base64
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


PRE_ROOT = Path(__file__).resolve().parents[1]
SECRET_NAME = "seaweedfs-s3-credentials"
FAKE_ACCESS_KEY = 'fake-access-":#{}\nsecond-line'
FAKE_SECRET_KEY = 'fake-secret-":#{}\nsecond-line'


def load_helper(name):
    path = PRE_ROOT / "scripts" / "lib" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"pre_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


provision_s3_buckets = load_helper("provision_s3_buckets")
s3_credentials = load_helper("s3_credentials")


def response(status=200):
    return {"ResponseMetadata": {"HTTPStatusCode": status}}


class FakeS3Error(Exception):
    def __init__(self, code, status):
        super().__init__(code)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class FakeS3:
    def __init__(
        self,
        existing=(),
        head_error=None,
        head_status=200,
        create_error=None,
        create_status=200,
        list_error=None,
        list_status=200,
    ):
        self.buckets = set(existing)
        self.head_error = head_error
        self.head_status = head_status
        self.create_error = create_error
        self.create_status = create_status
        self.list_error = list_error
        self.list_status = list_status
        self.calls = []

    def head_bucket(self, Bucket):
        self.calls.append(("head", Bucket))
        if self.head_error:
            raise self.head_error
        if Bucket not in self.buckets:
            raise FakeS3Error("NoSuchBucket", 404)
        return response(self.head_status)

    def create_bucket(self, Bucket):
        self.calls.append(("create", Bucket))
        if self.create_error:
            if self.create_error.response["Error"]["Code"] == "BucketAlreadyOwnedByYou":
                self.buckets.add(Bucket)
            raise self.create_error
        if self.create_status == 200:
            self.buckets.add(Bucket)
        return response(self.create_status)

    def list_objects_v2(self, Bucket, MaxKeys):
        self.calls.append(("list", Bucket, MaxKeys))
        if self.list_error:
            raise self.list_error
        if Bucket not in self.buckets:
            raise FakeS3Error("NoSuchBucket", 404)
        return response(self.list_status)


class BucketProvisioningTest(unittest.TestCase):
    def test_required_buckets_are_idempotent_and_authenticated(self):
        self.assertEqual(
            provision_s3_buckets.BUCKETS,
            ("mlpipeline", "nereidforge", "labelstudio", "cvat"),
        )
        s3 = FakeS3(existing={"nereidforge"})
        provision_s3_buckets.ensure_buckets(s3)
        provision_s3_buckets.ensure_buckets(s3)

        created = [call[1] for call in s3.calls if call[0] == "create"]
        listed = [call for call in s3.calls if call[0] == "list"]
        self.assertEqual(created, ["mlpipeline", "labelstudio", "cvat"])
        self.assertEqual(len(listed), 8)
        self.assertTrue(all(call[2] == 1 for call in listed))
        self.assertFalse(any(call[0] in ("put", "delete") for call in s3.calls))

    def test_forbidden_head_never_counts_as_existence(self):
        for error in (FakeS3Error("403", 403), FakeS3Error("NoSuchBucket", 403)):
            with self.subTest(error=error.response):
                s3 = FakeS3(head_error=error)
                with self.assertRaises(FakeS3Error):
                    provision_s3_buckets.ensure_bucket(s3, "mlpipeline")
                self.assertFalse(any(call[0] == "create" for call in s3.calls))

    def test_authentication_and_endpoint_errors_fail_closed(self):
        for error in (
            FakeS3Error("AccessDenied", 403),
            RuntimeError("fake unreachable endpoint"),
        ):
            with self.subTest(error=type(error).__name__):
                s3 = FakeS3(existing={"cvat"}, list_error=error)
                with self.assertRaises(type(error)):
                    provision_s3_buckets.ensure_bucket(s3, "cvat")

    def test_unsuccessful_http_results_fail_closed(self):
        scenarios = (
            (FakeS3(head_status=403, existing={"cvat"}), "cvat"),
            (FakeS3(create_status=500), "mlpipeline"),
            (FakeS3(list_status=500, existing={"cvat"}), "cvat"),
        )
        for s3, bucket in scenarios:
            with self.subTest(bucket=bucket, calls=s3.calls):
                with self.assertRaises(RuntimeError):
                    provision_s3_buckets.ensure_bucket(s3, bucket)

    def test_creation_race_still_requires_authenticated_verification(self):
        s3 = FakeS3(create_error=FakeS3Error("BucketAlreadyOwnedByYou", 409))
        provision_s3_buckets.ensure_bucket(s3, "mlpipeline")
        self.assertIn(("list", "mlpipeline", 1), s3.calls)


class CredentialSecretTest(unittest.TestCase):
    def test_manifest_preserves_credentials_and_narrow_anonymous_access(self):
        manifest = s3_credentials.secret_manifest(FAKE_ACCESS_KEY, FAKE_SECRET_KEY)
        self.assertEqual(manifest["kind"], "Secret")
        self.assertEqual(manifest["metadata"]["name"], SECRET_NAME)
        self.assertEqual(manifest["metadata"]["annotations"]["helm.sh/resource-policy"], "keep")
        self.assertEqual(manifest["stringData"]["accessKey"], FAKE_ACCESS_KEY)
        self.assertEqual(manifest["stringData"]["secretKey"], FAKE_SECRET_KEY)

        config_text = manifest["stringData"]["s3-config.json"]
        for legacy in (
            {"kind": "Secret", "stringData": {"s3-config.json": config_text}},
            {"kind": "ConfigMap", "data": {"s3-config.json": config_text}},
        ):
            with self.subTest(kind=legacy["kind"]):
                self.assertEqual(
                    s3_credentials.credentials_from_config(legacy),
                    (FAKE_ACCESS_KEY, FAKE_SECRET_KEY),
                )
        config = json.loads(config_text)
        anonymous = next(item for item in config["identities"] if item["name"] == "anonymous")
        self.assertEqual(
            anonymous["actions"],
            ["Read:nereidforge/models/*", "Read:nereidforge/optimized/*"],
        )

    def test_existing_secret_round_trip_does_not_rotate_identity(self):
        original = s3_credentials.secret_manifest(FAKE_ACCESS_KEY, FAKE_SECRET_KEY)
        encoded = {
            "data": {
                key: base64.b64encode(value.encode()).decode()
                for key, value in original["stringData"].items()
            }
        }
        first = s3_credentials.credentials_from_secret(encoded)
        second = s3_credentials.credentials_from_secret(encoded)
        self.assertEqual(first, (FAKE_ACCESS_KEY, FAKE_SECRET_KEY))
        self.assertEqual(second, first)
        self.assertEqual(s3_credentials.secret_manifest(*first)["stringData"], original["stringData"])

    def test_shell_exports_preserve_special_bytes_without_eval_injection(self):
        exports = s3_credentials.shell_exports(FAKE_ACCESS_KEY, FAKE_SECRET_KEY)
        result = subprocess.run(
            ["bash", "-c", exports + 'printf "%s\\0%s\\0" "$SEAWEEDFS_S3_ACCESS_KEY" "$SEAWEEDFS_S3_SECRET_KEY"'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(
            result.stdout.split(b"\0")[:2],
            [FAKE_ACCESS_KEY.encode(), FAKE_SECRET_KEY.encode()],
        )

    def test_checksum_contains_no_credential_and_missing_keys_fail(self):
        manifest = s3_credentials.secret_manifest(FAKE_ACCESS_KEY, FAKE_SECRET_KEY)
        encoded = {
            "data": {
                key: base64.b64encode(value.encode()).decode()
                for key, value in manifest["stringData"].items()
            }
        }
        digest = s3_credentials.secret_checksum(encoded)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertNotIn(FAKE_ACCESS_KEY, digest)
        self.assertNotIn(FAKE_SECRET_KEY, digest)
        with self.assertRaises(s3_credentials.CredentialError):
            s3_credentials.credentials_from_secret({"data": {}})


@unittest.skipUnless(shutil.which("helm"), "helm is required for chart rendering tests")
class PreChartRenderingTest(unittest.TestCase):
    def render(self, chart, values=None, values_file=None, release_name="test", extra_args=()):
        command = ["helm", "template", release_name, str(chart), "--namespace", "nereidforge"]
        command.extend(extra_args)
        temporary = None
        if values is not None:
            temporary = tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8")
            json.dump(values, temporary)
            temporary.flush()
            command.extend(("-f", temporary.name))
        if values_file is not None:
            command.extend(("-f", str(values_file)))
        try:
            return subprocess.run(
                command,
                cwd=PRE_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        finally:
            if temporary is not None:
                temporary.close()

    def test_seaweedfs_mounts_existing_secret_without_rendering_credentials(self):
        chart = PRE_ROOT / "charts/seaweedfs"
        rendered = self.render(
            chart,
            release_name="seaweedfs",
            values={"s3": {"existingSecret": SECRET_NAME, "configKey": "s3-config.json"}},
        )
        self.assertIn(f"secretName: {SECRET_NAME}", rendered)
        self.assertIn("key: s3-config.json", rendered)
        self.assertIn("path: s3-config.json", rendered)
        self.assertIn("checksum/s3-credentials:", rendered)
        self.assertNotIn("kind: Secret", rendered)
        self.assertNotIn("kind: ConfigMap\nmetadata:\n  name: seaweedfs-s3-config", rendered)
        self.assertNotIn("accessKey: nereidforge", rendered)
        self.assertNotIn("secretKey: nereidforge", rendered)
        with self.assertRaises(subprocess.CalledProcessError):
            self.render(chart, values={"s3": {"existingSecret": ""}})

    def test_labelstudio_consumes_the_shared_secret(self):
        chart = PRE_ROOT / "charts/labelstudio"
        rendered = self.render(chart, values={"s3": {"existingSecret": SECRET_NAME}})
        self.assertIn(f"name: {SECRET_NAME}", rendered)
        self.assertIn("checksum/s3-credentials:", rendered)
        for name in ("S3_ACCESS_KEY", "S3_SECRET_KEY"):
            self.assertIn(f"- name: {name}", rendered)
        self.assertIn("key: accessKey", rendered)
        self.assertIn("key: secretKey", rendered)
        self.assertNotIn("accessKey: nereidforge", rendered)
        self.assertNotIn("secretKey: nereidforge", rendered)
        with self.assertRaises(subprocess.CalledProcessError):
            self.render(chart, values={"s3": {"existingSecret": ""}})

    def test_cvat_uses_secret_references_not_hardcoded_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            chart = Path(temporary) / "cvat"
            shutil.copytree(PRE_ROOT / "charts/cvat", chart)
            (chart / "charts").mkdir()
            for archive in chart.glob("*.tgz"):
                (chart / "charts" / archive.name).symlink_to(archive.resolve())
            rendered = self.render(
                chart,
                values_file=PRE_ROOT / "values/cvat.yaml",
                extra_args=(
                    "--set-string",
                    "cvat.backend.annotations.checksum/s3-credentials=fake-checksum",
                ),
            )
        self.assertIn("AWS_ACCESS_KEY_ID", rendered)
        self.assertIn("AWS_SECRET_ACCESS_KEY", rendered)
        self.assertIn(f"name: {SECRET_NAME}", rendered)
        self.assertIn("key: accessKey", rendered)
        self.assertIn("key: secretKey", rendered)
        self.assertIn("checksum/s3-credentials: fake-checksum", rendered)
        self.assertNotIn('value: "nereidforge"', rendered)


class PreInstallerRuntimeTest(unittest.TestCase):
    def test_rerun_reuses_existing_identity_without_leaking_or_rotating_it(self):
        """Run the real installer stage entirely against local recording fakes."""
        existing_access = "existing-access-keep-me"
        existing_secret = "existing-secret-keep-me"
        replacement_access = "replacement-access-must-not-win"
        replacement_secret = "replacement-secret-must-not-win"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_python = root / "python"
            fake_bin.mkdir()
            (fake_python / "botocore").mkdir(parents=True)

            manifest = s3_credentials.secret_manifest(existing_access, existing_secret)
            existing_resource = {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": manifest["metadata"],
                "data": {
                    key: base64.b64encode(value.encode()).decode()
                    for key, value in manifest["stringData"].items()
                },
            }
            existing_file = root / "existing.json"
            existing_file.write_text(json.dumps(existing_resource))
            apply_capture = root / "applied.json"
            helm_capture = root / "helm-args"
            boto_capture = root / "boto.json"
            bucket_calls = root / "bucket-calls"

            kubectl = fake_bin / "kubectl"
            kubectl.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/bash
                    set -euo pipefail
                    case "$*" in
                        "get ns "*) exit 0 ;;
                        "get secret seaweedfs-s3-credentials"*) cat "$TEST_EXISTING_SECRET" ;;
                        "apply -f -") cat > "$TEST_APPLY_CAPTURE" ;;
                        "rollout status "*) exit 0 ;;
                        "get sts "*) printf '1\\n' ;;
                        "get svc "*) printf '10.23.45.67\\n' ;;
                        "run "*)
                            case "$*" in
                                *sw-test2-*) printf 'SeaweedFS Filer\\n' ;;
                                *sw-test3-*) printf 'ListAllMyBucketsResult\\n' ;;
                                *) printf 'SeaweedFS\\n' ;;
                            esac
                            ;;
                        *) printf 'unexpected kubectl invocation: %s\\n' "$*" >&2; exit 64 ;;
                    esac
                    """
                )
            )
            kubectl.chmod(0o755)
            helm = fake_bin / "helm"
            helm.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\" >> \"$TEST_HELM_ARGS\"\n")
            helm.chmod(0o755)
            sleep = fake_bin / "sleep"
            sleep.write_text("#!/bin/bash\nexit 0\n")
            sleep.chmod(0o755)

            (fake_python / "boto3.py").write_text(
                textwrap.dedent(
                    """\
                    import json
                    import os

                    class FakeS3:
                        def _record(self, operation, bucket):
                            with open(os.environ["TEST_BUCKET_CALLS"], "a") as stream:
                                stream.write(f"{operation} {bucket}\\n")

                        def head_bucket(self, Bucket):
                            self._record("head", Bucket)
                            return {"ResponseMetadata": {"HTTPStatusCode": 200}}

                        def list_objects_v2(self, Bucket, MaxKeys):
                            self._record("list", Bucket)
                            return {"ResponseMetadata": {"HTTPStatusCode": 200}}

                    def client(*args, **kwargs):
                        kwargs.pop("config", None)
                        with open(os.environ["TEST_BOTO_CAPTURE"], "w") as stream:
                            json.dump(kwargs, stream)
                        return FakeS3()
                    """
                )
            )
            (fake_python / "botocore/__init__.py").write_text("")
            (fake_python / "botocore/client.py").write_text(
                "class Config:\n    def __init__(self, **kwargs):\n        self.kwargs = kwargs\n"
            )

            env = os.environ.copy()
            env.update(
                PATH=f"{fake_bin}:{env['PATH']}",
                PYTHONPATH=str(fake_python),
                KUBECTL=str(kubectl),
                SEAWEEDFS_S3_ACCESS_KEY=replacement_access,
                SEAWEEDFS_S3_SECRET_KEY=replacement_secret,
                TEST_EXISTING_SECRET=str(existing_file),
                TEST_APPLY_CAPTURE=str(apply_capture),
                TEST_HELM_ARGS=str(helm_capture),
                TEST_BOTO_CAPTURE=str(boto_capture),
                TEST_BUCKET_CALLS=str(bucket_calls),
            )
            result = subprocess.run(
                ["bash", str(PRE_ROOT / "scripts/10_infra/seaweedfs.sh")],
                cwd=PRE_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertNotIn(existing_access, output)
            self.assertNotIn(existing_secret, output)
            self.assertNotIn(replacement_access, output)
            self.assertNotIn(replacement_secret, output)
            self.assertIn("Reusing the existing SeaweedFS credentials Secret", output)
            self.assertIn("All required S3 buckets are provisioned", output)

            applied = json.loads(apply_capture.read_text())
            self.assertEqual(applied["stringData"]["accessKey"], existing_access)
            self.assertEqual(applied["stringData"]["secretKey"], existing_secret)
            self.assertNotIn(replacement_access, apply_capture.read_text())
            self.assertNotIn(replacement_secret, apply_capture.read_text())
            helm_args = helm_capture.read_text()
            self.assertIn("s3.existingSecret=seaweedfs-s3-credentials", helm_args)
            for credential in (existing_access, existing_secret, replacement_access, replacement_secret):
                self.assertNotIn(credential, helm_args)

            boto = json.loads(boto_capture.read_text())
            self.assertEqual(boto["aws_access_key_id"], existing_access)
            self.assertEqual(boto["aws_secret_access_key"], existing_secret)
            calls = bucket_calls.read_text().splitlines()
            for bucket in provision_s3_buckets.BUCKETS:
                self.assertIn(f"head {bucket}", calls)
                self.assertIn(f"list {bucket}", calls)
            self.assertEqual(len(calls), 8)


class PreInstallerStaticTest(unittest.TestCase):
    def test_tracked_pre_values_have_no_working_default_s3_identity(self):
        for relative in (
            "charts/seaweedfs/values.yaml",
            "charts/seaweedfs/values-prod.yaml",
            "charts/labelstudio/values.yaml",
            "values/cvat.yaml",
        ):
            with self.subTest(path=relative):
                text = (PRE_ROOT / relative).read_text()
                self.assertNotIn("accessKey: nereidforge", text)
                self.assertNotIn("secretKey: nereidforge", text)
                self.assertNotIn('value: "nereidforge"', text)

    def test_online_scripts_never_pass_s3_secrets_through_helm_or_values(self):
        seaweedfs = (PRE_ROOT / "scripts/10_infra/seaweedfs.sh").read_text()
        labelstudio = (PRE_ROOT / "scripts/10_infra/labelstudio.sh").read_text()
        for line in seaweedfs.splitlines():
            if "--set" in line:
                self.assertNotIn("SEAWEEDFS_S3_ACCESS_KEY", line)
                self.assertNotIn("SEAWEEDFS_S3_SECRET_KEY", line)
        self.assertNotIn("S3_VALUES_FILE", seaweedfs)
        self.assertIn('s3.existingSecret="$S3_CREDENTIALS_SECRET"', seaweedfs)
        self.assertNotIn("accessKey: $SEAWEEDFS_S3_ACCESS_KEY", labelstudio)
        self.assertNotIn("secretKey: $SEAWEEDFS_S3_SECRET_KEY", labelstudio)

    def test_reruns_reuse_and_retain_the_existing_secret(self):
        script = (PRE_ROOT / "scripts/10_infra/seaweedfs.sh").read_text()
        self.assertIn('get secret "$S3_CREDENTIALS_SECRET"', script)
        self.assertLess(script.index('get secret "$S3_CREDENTIALS_SECRET"'), script.index("read_password()"))
        self.assertIn("s3_credentials.py\" manifest", script)
        self.assertIn("load_seaweedfs_s3_credentials secret \"$S3_CREDENTIALS_SECRET\" secret", script)
        helper = (PRE_ROOT / "scripts/lib/s3_credentials.py").read_text()
        self.assertIn('"helm.sh/resource-policy": "keep"', helper)

    def test_bucket_provisioning_precedes_every_pre_consumer(self):
        infra = (PRE_ROOT / "scripts/10_infra.sh").read_text()
        seaweedfs = (PRE_ROOT / "scripts/10_infra/seaweedfs.sh").read_text()
        kubeflow = (PRE_ROOT / "scripts/30_install-kubeflow.sh").read_text()

        self.assertLess(infra.index("10_infra/seaweedfs.sh"), infra.index("10_infra/labelstudio.sh"))
        self.assertLess(infra.index("10_infra/seaweedfs.sh"), infra.index("10_infra/cvat.sh"))
        self.assertIn("provision_s3_buckets.py", seaweedfs)
        self.assertIn("load_seaweedfs_s3_credentials secret seaweedfs-s3-credentials secret", kubeflow)
        self.assertNotIn("assuming ok", kubeflow)
        self.assertNotIn("mlpipeline bucket not provisioned", kubeflow)


if __name__ == "__main__":
    unittest.main()
