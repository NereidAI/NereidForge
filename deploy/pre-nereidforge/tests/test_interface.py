#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


PRE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PRE_ROOT / "scripts" / "lib" / "config.sh"


class PreInterfaceTests(unittest.TestCase):
    def source_config(self, extra_env=None, script="export_all; env -0"):
        env = os.environ.copy()
        scrub = [
            "PRE_RELEASE_TAG", "SOURCE_COMMIT", "RELEASE_TAG", "PRE_ENV_FILE",
            "NF_ENV_FILE", "PRE_HANDOFF_DIR", "NF_HANDOFF_DIR", "JWT_SECRET",
            "NEREIDFORGE_AUTH_JWT_SECRET", "NF_BACKEND_IMAGE", "NF_FRONTEND_IMAGE",
            "NF_TRAINER_IMAGE", "IMG_ML_BACKEND", "IMG_XPU_BASE", "VALUES_FILE",
            "SECRETS_FILE", "SECRETS_TEMPLATE", "APP_CHARTS_DIR",
        ]
        for name in scrub:
            env.pop(name, None)
        env["PRE_RELEASE_TAG"] = "1.2.3"
        env["RELEASE_TAG"] = "9.9.9"
        env["DEPLOYMENT_ROOT"] = tempfile.mkdtemp(prefix="pre-interface-")
        self.addCleanup(__import__("shutil").rmtree, env["DEPLOYMENT_ROOT"], True)
        if extra_env:
            env.update(extra_env)
        result = subprocess.run(
            ["bash", "-c", f'set -e; source "$1"; {script}', "_", str(CONFIG)],
            text=True,
            capture_output=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_required_pre_inputs_and_defaults_remain_available(self):
        result = self.source_config()
        values = dict(item.split("=", 1) for item in result.stdout.split("\0") if "=" in item)
        expected = {
            "PRE_RELEASE_TAG": "1.2.3",
            "BASE_DOMAIN": "",
            "STORAGE_CLASS": "nfs-csi",
            "IMG_POSTGRESQL": "ghcr.io/cloudnative-pg/postgresql:16.4",
            "IMG_VALKEY": "valkey/valkey:8.0-alpine",
            "IMG_SEAWEEDFS": "chrislusf/seaweedfs:4.34",
            "IMG_LABEL_STUDIO": "heartexlabs/label-studio:1.23.0",
            "IMG_BUSYBOX": "busybox:1.36",
            "PG_CLUSTER_NAME": "nereidforge-postgresql",
            "PG_INSTANCES": "3",
            "PG_STORAGE_SIZE": "50Gi",
            "PG_DB_NAME": "nereidforge",
            "PG_DB_USER": "nereidforge",
            "PG_ADMIN_PASSWORD": "",
            "SEAWEEDFS_MASTER_STORAGE": "10Gi",
            "SEAWEEDFS_VOLUME_STORAGE": "200Gi",
            "SEAWEEDFS_S3_ACCESS_KEY": "",
            "SEAWEEDFS_S3_SECRET_KEY": "",
            "LABEL_STUDIO_STORAGE": "20Gi",
            "DEX_STATIC_EMAIL": "",
            "DEX_STATIC_PASSWORD": "",
            "DEX_STATIC_PASSWORD_HASH": "",
            "POD_CIDR": "10.42.0.0/16",
            "SERVICE_CIDR": "10.43.0.0/16",
            "KUBECTL": "kubectl",
            "MANIFESTS_DIR": str(PRE_ROOT / "platform" / "kubeflow-manifests"),
            "CHARTS_DIR": str(PRE_ROOT / "charts"),
            "OFFLINE_DIR": str(PRE_ROOT / "offline-package"),
        }
        for name, value in expected.items():
            self.assertIn(name, values, name)
            self.assertEqual(values[name], value, name)

    def test_pre_env_file_is_loaded_without_nf_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / "site.env"
            site.write_text("BASE_DOMAIN=example.test\nSEAWEEDFS_S3_ACCESS_KEY=test-access\n")
            result = self.source_config(
                {"PRE_ENV_FILE": str(site)},
                script='export_all; printf "%s\\n%s\\n%s" "$BASE_DOMAIN" "$SEAWEEDFS_S3_ACCESS_KEY" "${JWT_SECRET:-}"',
            )
        self.assertEqual(result.stdout.split("\n"), ["example.test", "test-access", ""])
        self.assertNotIn("JWT", result.stderr)

    def test_nf_only_variables_and_root_value_paths_are_not_defined(self):
        result = self.source_config(
            script='for name in JWT_SECRET NEREIDFORGE_AUTH_JWT_SECRET NF_BACKEND_IMAGE NF_FRONTEND_IMAGE NF_TRAINER_IMAGE IMG_ML_BACKEND IMG_XPU_BASE VALUES_FILE SECRETS_FILE SECRETS_TEMPLATE APP_CHARTS_DIR; do if [[ -n "${!name+x}" ]]; then printf "%s\\n" "$name"; fi; done'
        )
        self.assertEqual(result.stdout, "")
        self.assertNotIn("JWT", result.stderr)

    def test_full_stack_site_and_handoff_paths_remain_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            values = Path(directory) / "values"
            handoff = Path(directory) / "handoff"
            site = Path(directory) / "root-site.env"
            offline = Path(directory) / "offline"
            site.write_text("BASE_DOMAIN=full-stack.example.test\n")
            result = self.source_config(
                {
                    "PRE_ENV_FILE": "",
                    "NF_ENV_FILE": str(site),
                    "DEPLOY_VALUES_DIR": str(values),
                    "PRE_HANDOFF_DIR": "",
                    "NF_HANDOFF_DIR": str(handoff),
                    "OFFLINE_DIR": str(offline),
                },
                script='export_all; printf "%s\\n%s\\n%s\\n%s\\n%s" "$PRE_ENV_FILE" "$NF_ENV_FILE" "$PRE_HANDOFF_DIR" "$NF_HANDOFF_DIR" "$OFFLINE_DIR"',
            )
        self.assertEqual(
            result.stdout.splitlines(),
            [str(site), str(site), str(handoff), str(handoff), str(offline)],
        )

    def test_root_full_stack_entry_points_still_address_pre_scripts(self):
        repository = PRE_ROOT.parents[1]
        if not (repository / "scripts" / "00_deploy-all.sh").is_file():
            self.skipTest("full-stack compatibility is available only in a combined checkout")
        deploy_all = (repository / "scripts" / "00_deploy-all.sh").read_text()
        verify = (repository / "scripts" / "60_verify.sh").read_text()
        for relative in (
            "deploy/pre-nereidforge/scripts/10_infra.sh",
            "deploy/pre-nereidforge/scripts/30_install-kubeflow.sh",
        ):
            self.assertIn(relative, deploy_all)
            self.assertTrue((repository / relative).is_file())
        for relative in (
            "deploy/pre-nereidforge/scripts/60_verify/infra-check.sh",
            "deploy/pre-nereidforge/scripts/60_verify/kubeflow-check.sh",
            "deploy/pre-nereidforge/scripts/60_verify/mr-check.sh",
        ):
            self.assertIn(relative, verify)
            self.assertTrue((repository / relative).is_file())

    def test_no_pre_script_sources_root_libraries_or_uses_root_paths(self):
        forbidden = {
            "root library path": re.compile(r"\.\./\.\./scripts/lib|(^|[\"'])\.\./scripts/lib"),
            "ROOT_DIR runtime variable": re.compile(r"(?<![A-Z_])ROOT_DIR"),
            "root production values": re.compile(r"(?:PRE_ROOT|REPO_ROOT)/(?:values-prod\.yaml|secrets\.yaml\.template)"),
        }
        failures = []
        for script in (PRE_ROOT / "scripts").rglob("*.sh"):
            text = script.read_text()
            for description, pattern in forbidden.items():
                if pattern.search(text):
                    failures.append(f"{script.relative_to(PRE_ROOT)}: {description}")
        self.assertEqual(failures, [])

    def test_pre_libraries_contain_no_nf_image_or_jwt_configuration(self):
        text = "\n".join(path.read_text() for path in (PRE_ROOT / "scripts" / "lib").glob("*.sh"))
        for token in ("nf_ref", "NF_BACKEND_IMAGE", "NF_FRONTEND_IMAGE", "IMG_ML_BACKEND", "JWT_SECRET"):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
