# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Combined-checkout tests for NF integration with pre-managed S3 Secrets."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SECRET_NAME = "seaweedfs-s3-credentials"


@unittest.skipUnless(shutil.which("helm"), "helm is required for chart rendering tests")
class NFChartRenderingTest(unittest.TestCase):
    def render(self, chart, values=None, release_name="test", extra_args=()):
        command = ["helm", "template", release_name, str(chart), "--namespace", "nereidforge"]
        command.extend(extra_args)
        temporary = None
        if values is not None:
            temporary = tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8")
            json.dump(values, temporary)
            temporary.flush()
            command.extend(("-f", temporary.name))
        try:
            return subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        finally:
            if temporary is not None:
                temporary.close()

    def test_nf_charts_consume_the_pre_managed_shared_secret(self):
        cases = (
            ("charts/ml-backend", ("S3_ACCESS_KEY", "S3_SECRET_KEY")),
            (
                "charts/nereidforge",
                (
                    "NEREIDFORGE_STORAGE_S3_ACCESS_KEY",
                    "NEREIDFORGE_STORAGE_S3_SECRET_KEY",
                    "NEREIDFORGE_KFP_OBJECT_STORE_ACCESS_KEY",
                    "NEREIDFORGE_KFP_OBJECT_STORE_SECRET_KEY",
                ),
            ),
        )
        for chart, environment_names in cases:
            with self.subTest(chart=chart):
                rendered = self.render(
                    ROOT / chart,
                    values={
                        "s3": {"existingSecret": SECRET_NAME},
                        "frontend": {"ingress": {"host": "forge.example.test"}},
                    },
                )
                self.assertIn(f"name: {SECRET_NAME}", rendered)
                self.assertIn("secretKeyRef:", rendered)
                for name in environment_names:
                    self.assertIn(f"- name: {name}", rendered)
                self.assertNotIn("accessKey: nereidforge", rendered)
                self.assertNotIn("secretKey: nereidforge", rendered)
                with self.assertRaises(subprocess.CalledProcessError):
                    self.render(
                        ROOT / chart,
                        values={
                            "s3": {"existingSecret": ""},
                            "frontend": {"ingress": {"host": "forge.example.test"}},
                        },
                    )

    def test_backend_api_and_worker_both_receive_all_s3_references(self):
        rendered = self.render(
            ROOT / "charts/nereidforge",
            values={
                "s3": {"existingSecret": SECRET_NAME},
                "frontend": {"ingress": {"host": "forge.example.test"}},
            },
        )
        for name in (
            "NEREIDFORGE_STORAGE_S3_ACCESS_KEY",
            "NEREIDFORGE_STORAGE_S3_SECRET_KEY",
            "NEREIDFORGE_KFP_OBJECT_STORE_ACCESS_KEY",
            "NEREIDFORGE_KFP_OBJECT_STORE_SECRET_KEY",
        ):
            self.assertEqual(rendered.count(f"- name: {name}"), 2)
        self.assertEqual(rendered.count("checksum/s3-credentials:"), 2)

    def test_backend_secret_template_has_no_literal_s3_fallback(self):
        template = (ROOT / "charts/nereidforge/templates/configmap.yaml").read_text()
        values = (ROOT / "charts/nereidforge/values.yaml").read_text()
        self.assertNotIn("$secrets.kfp", template)
        self.assertNotIn("$secrets.storage", template)
        self.assertNotIn("object_store_access_key: \"\"  # required at install time", values)
        self.assertNotIn("s3_access_key: \"\"  # required at install time", values)


class CombinedInstallerStaticTest(unittest.TestCase):
    def test_tracked_nf_values_have_no_working_default_s3_identity(self):
        for relative in (
            "values-prod.yaml",
            "charts/ml-backend/values.yaml",
            "charts/nereidforge/config.rendered.yaml",
            "configs/unified/config.rendered.yaml",
            "configs/unified/config.yaml.j2",
        ):
            with self.subTest(path=relative):
                text = (ROOT / relative).read_text()
                self.assertNotIn("accessKey: nereidforge", text)
                self.assertNotIn("secretKey: nereidforge", text)
                self.assertNotIn('accessKey: "nereidforge"', text)
                self.assertNotIn('secretKey: "nereidforge"', text)

    def test_nf_scripts_do_not_pass_s3_secrets_through_helm_set(self):
        backend = (ROOT / "scripts/40_nereidforge/app.sh").read_text()
        ml_backend = (ROOT / "scripts/40_nereidforge/ml-backend.sh").read_text()
        for line in (backend + ml_backend).splitlines():
            if "--set" in line:
                self.assertNotIn("SEAWEEDFS_S3_ACCESS_KEY", line)
                self.assertNotIn("SEAWEEDFS_S3_SECRET_KEY", line)
        self.assertIn('s3.existingSecret="seaweedfs-s3-credentials"', backend)
        self.assertIn('s3.existingSecret=seaweedfs-s3-credentials', ml_backend)

    def test_pre_bucket_provisioning_precedes_kubeflow_and_nf(self):
        deploy_all = (ROOT / "scripts/00_deploy-all.sh").read_text()
        infra = (ROOT / "deploy/pre-nereidforge/scripts/10_infra.sh").read_text()
        seaweedfs = (ROOT / "deploy/pre-nereidforge/scripts/10_infra/seaweedfs.sh").read_text()
        kubeflow = (ROOT / "deploy/pre-nereidforge/scripts/30_install-kubeflow.sh").read_text()

        self.assertLess(infra.index("10_infra/seaweedfs.sh"), infra.index("10_infra/labelstudio.sh"))
        self.assertLess(infra.index("10_infra/seaweedfs.sh"), infra.index("10_infra/cvat.sh"))
        self.assertLess(
            deploy_all.index("deploy/pre-nereidforge/scripts/10_infra.sh"),
            deploy_all.index("deploy/pre-nereidforge/scripts/30_install-kubeflow.sh"),
        )
        self.assertLess(
            deploy_all.index("deploy/pre-nereidforge/scripts/30_install-kubeflow.sh"),
            deploy_all.index("40_nereidforge.sh"),
        )
        self.assertIn('"$LIB_DIR/provision_s3_buckets.py"', seaweedfs)
        self.assertNotIn("scripts/lib/provision_s3_buckets.py", seaweedfs.replace('"$LIB_DIR/provision_s3_buckets.py"', ""))
        self.assertNotIn("assuming ok", kubeflow)
        self.assertNotIn("mlpipeline bucket not provisioned", kubeflow)

    def test_custom_ml_backend_uses_the_same_secret(self):
        clients = types.ModuleType("app.clients")
        clients.cluster_ops = types.SimpleNamespace()
        images = types.ModuleType("app.core.images")
        images.nf_image = lambda name: "fake.invalid/" + name
        modules = {
            "app": types.ModuleType("app"),
            "app.clients": clients,
            "app.core": types.ModuleType("app.core"),
            "app.core.images": images,
        }
        path = ROOT / "python-backend/app/k8s/ml_backend_deployer.py"
        spec = importlib.util.spec_from_file_location("test_ml_backend_deployer", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)

        manifest = module._deployment_manifest(
            "fake-service", "fake-model", "v1", "s3://fake/model", "det", "fake"
        )
        environment = {
            item["name"]: item
            for item in manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        for name, key in (("S3_ACCESS_KEY", "accessKey"), ("S3_SECRET_KEY", "secretKey")):
            with self.subTest(environment=name):
                self.assertEqual(
                    environment[name]["valueFrom"]["secretKeyRef"],
                    {"name": SECRET_NAME, "key": key, "optional": False},
                )
                self.assertNotIn("value", environment[name])


if __name__ == "__main__":
    unittest.main()
