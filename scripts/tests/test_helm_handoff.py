# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Chart and post-install contracts for the direct OCI Helm handoff."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "deploy/pre-nereidforge/scripts/lib/generate_nf_handoff.py"


@unittest.skipUnless(shutil.which("helm"), "helm is required for chart rendering tests")
class HelmHandoffRenderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="nf-helm-handoff-")
        cls.values = Path(cls.temporary.name) / "values.generated.yaml"
        subprocess.run(
            [
                "python3", str(GENERATOR), "generate", "--output", str(cls.values),
                "--pre-version", "0.0.0-dev-test", "--source-commit", "abc1234",
                "--base-domain", "example.test", "--storage-class", "nfs-csi",
                "--pg-cluster", "nereidforge-postgresql", "--pg-database", "nereidforge",
                "--pg-user", "nereidforge", "--dex-email", "admin@example.test",
            ],
            check=True,
            capture_output=True,
        )
        # The single consolidated chart renders API, worker and frontend from
        # one release using the same generated handoff values.
        cls.rendered = subprocess.run(
            [
                "helm", "template", "nereidforge",
                str(ROOT / "charts/nereidforge"),
                "--namespace", "nereidforge", "-f", str(cls.values),
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def render(self, chart, release):
        return subprocess.run(
            [
                "helm", "template", release, str(ROOT / chart),
                "--namespace", "nereidforge", "-f", str(self.values),
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout

    def test_application_uses_only_retained_secrets_and_renders_ordered_jobs(self):
        rendered = self.rendered
        self.assertNotIn("kind: Namespace\n", rendered)
        self.assertNotIn("name: pg-nereidforge-credentials\n", rendered)
        self.assertIn("name: pg-nereidforge-postgresql-credentials", rendered)
        self.assertIn("name: nereidforge-service-credentials", rendered)
        # The chart renders no managed Secret when every credential is retained.
        self.assertNotIn("kind: Secret\n", rendered)
        self.assertNotIn("name: nereidforge-secrets\n", rendered)
        self.assertEqual(3, rendered.count("kind: Job\n"))
        self.assertEqual(5, rendered.count("mountPath: /etc/nereidforge/config.yaml"))
        # One ConfigMap resource plus API, worker, and all three hook Job volumes.
        self.assertEqual(6, rendered.count("name: nereidforge-config"))
        self.assertIn('component: admin-seed', rendered)
        self.assertIn('component: mlbackend-seed', rendered)
        self.assertIn('component: pipeline-upload', rendered)
        self.assertLess(
            rendered.index('"helm.sh/hook-weight": "0"'),
            rendered.index('"helm.sh/hook-weight": "5"'),
        )
        self.assertLess(
            rendered.index('"helm.sh/hook-weight": "5"'),
            rendered.index('"helm.sh/hook-weight": "10"'),
        )
        # The preset script ships in the post-install ConfigMap and the hook
        # must not render a URL env when the generator leaves it to config.yaml
        # (the script's own source text still mentions NF_MLBACKEND_URL).
        self.assertIn("seed_ml_backend.py: |", rendered)
        self.assertNotIn("- name: NF_MLBACKEND_URL", rendered)
        self.assertIn('NF_ADMIN_ROTATE', rendered)
        self.assertIn('value: "false"', rendered)
        self.assertIn('before-hook-creation', rendered)

    def test_deep_merged_config_preserves_secret_env_placeholders(self):
        rendered = self.rendered
        self.assertIn('host: nereidforge-postgresql-rw.nereidforge.svc.cluster.local', rendered)
        self.assertIn('base_url: https://nereidsphere.example.test', rendered)
        self.assertIn('namespace: nereidforge', rendered)
        # Empty placeholders are configuration keys, not credentials. They allow
        # NEREIDFORGE_* environment overrides sourced from retained Secrets.
        self.assertIn('password: ""', rendered)
        self.assertIn('s3_access_key: ""', rendered)
        self.assertIn('s3_secret_key: ""', rendered)
        self.assertIn('object_store_access_key: ""', rendered)
        self.assertIn('object_store_secret_key: ""', rendered)

    def test_ml_backend_selects_existing_token_secret_without_rendering_one(self):
        rendered = self.render("charts/ml-backend", "ml-backend")
        self.assertNotIn("kind: Secret\n", rendered)
        self.assertIn("name: nereidforge-service-credentials", rendered)
        self.assertIn("key: NEREIDFORGE_LABELSTUDIO_TOKEN", rendered)
        self.assertIn("name: seaweedfs-s3-credentials", rendered)
        self.assertNotIn("stringData:", rendered)

    def test_frontend_tier_contains_same_origin_urls_and_all_five_routes(self):
        # The frontend templates are part of the consolidated release; the
        # handoff frontend: block drives the ingress and SPA environment.
        rendered = self.rendered
        self.assertIn('value: https://nereidforge.example.test/api', rendered)
        self.assertIn('value: wss://nereidforge.example.test', rendered)
        positions = [rendered.index(f'path: "{path}"') for path in (
            "/", "/api", "/auth/ns-callback", "/v1/integration/ns", "/notebook-servers",
        )]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('secretName: nereidforge-frontend-tls', rendered)
        # The live object names are pinned despite the merged release.
        self.assertIn("name: nereidforge-frontend", rendered)
        for deployment in ("nereidforge-api", "nereidforge-worker", "nereidforge-frontend"):
            self.assertIn(f"name: {deployment}", rendered)


    def test_mlbackend_seed_hook_is_gated_on_its_own_flag(self):
        """Chart defaults render no ml-backend seed Job; enabling renders it."""
        chart = str(ROOT / "charts/nereidforge")
        base_args = [
            "helm", "template", "nereidforge", chart,
            "--namespace", "nereidforge",
            # The chart requires an ingress host even when hooks are disabled.
            "--set", "frontend.ingress.host=forge.example.test",
        ]
        default_render = subprocess.run(
            base_args, cwd=ROOT, check=True, text=True, capture_output=True,
        ).stdout
        self.assertNotIn("mlbackend-seed", default_render)
        self.assertNotIn("seed_ml_backend.py", default_render)

        enabled_render = subprocess.run(
            base_args + [
                "--set", "postInstall.mlBackend.enabled=true",
                "--set", "postgresql.existingSecret=pg-test-credentials",
            ],
            cwd=ROOT, check=True, text=True, capture_output=True,
        ).stdout
        self.assertIn("component: mlbackend-seed", enabled_render)
        self.assertIn("seed_ml_backend.py: |", enabled_render)
        self.assertIn("/opt/nereidforge-post-install/seed_ml_backend.py", enabled_render)
        # The admin hook stays off: only the ml-backend Job is rendered.
        self.assertNotIn("component: admin-seed", enabled_render)


class SeedAdminJobTest(unittest.TestCase):
    def run_seed(self, *, existing, rotate=False, password="retained-password"):
        calls = []

        class Username:
            def __eq__(self, other):
                return other

        class User:
            username = Username()

        class Query:
            def where(self, _criterion):
                return self

        class Database:
            def scalar(self, _query):
                return types.SimpleNamespace(id=42) if existing else None

            def close(self):
                calls.append("closed")

        class AdminService:
            def __init__(self, _db):
                pass

            def create_user(self, username, email, _display, _password, tenant, roles):
                calls.append(("create", username, email, tenant, roles))
                return {"username": username, "tenant_id": tenant}

            def update_user(self, user_id, values):
                calls.append(("update", user_id, sorted(values)))

        sqlalchemy = types.ModuleType("sqlalchemy")
        sqlalchemy.select = lambda _model: Query()
        modules = {
            "sqlalchemy": sqlalchemy,
            "app": types.ModuleType("app"),
            "app.db": types.ModuleType("app.db"),
            "app.db.session": types.SimpleNamespace(SessionLocal=Database),
            "app.models": types.ModuleType("app.models"),
            "app.models.orm": types.SimpleNamespace(User=User),
            "app.services": types.ModuleType("app.services"),
            "app.services.admin": types.SimpleNamespace(AdminService=AdminService),
        }
        path = ROOT / "charts/nereidforge/scripts/seed_admin.py"
        spec = importlib.util.spec_from_file_location("nf_seed_admin_test", path)
        module = importlib.util.module_from_spec(spec)
        environment = {
            "NF_ADMIN_PASSWORD": password,
            "NF_ADMIN_ROTATE": "true" if rotate else "false",
            "NF_ADMIN_EMAIL": "admin@example.test",
        }
        with patch.dict(sys.modules, modules), patch.dict(os.environ, environment, clear=False):
            spec.loader.exec_module(module)
            if not password:
                with self.assertRaises(RuntimeError):
                    module.main()
            else:
                module.main()
        return calls

    def test_seed_create_skip_and_explicit_rotation_are_idempotent(self):
        created = self.run_seed(existing=False)
        self.assertEqual(created[0][:2], ("create", "admin"))
        self.assertEqual(created[-1], "closed")

        skipped = self.run_seed(existing=True)
        self.assertEqual(skipped, ["closed"])

        rotated = self.run_seed(existing=True, rotate=True)
        self.assertEqual(rotated[0], ("update", 42, ["password"]))
        self.assertEqual(rotated[-1], "closed")

        missing = self.run_seed(existing=False, password="")
        self.assertEqual(missing, [])


class SeedMLBackendJobTest(unittest.TestCase):
    """Contract for the idempotent system-preset ml-backend seed hook."""

    CONFIGURED_URL = "http://ml-backend.nereidforge.svc.cluster.local:9090"

    def run_seed(self, *, existing: bool, configured_url: str = CONFIGURED_URL,
                 env: dict | None = None, expect_failure: bool = False):
        calls = []

        class Query:
            def where(self, *_criteria):
                return self

        class Database:
            def scalar(self, _query):
                return types.SimpleNamespace(id="mlb-existing-1") if existing else None

            def close(self):
                calls.append("closed")

        class MLBackend:
            # Column attributes are read by the select predicate at call time;
            # their values are irrelevant because the stub Query ignores them.
            tenant_id = "tenant_id"
            url = "url"

        class MLBackendService:
            def __init__(self, db):
                self.db = db

            def create(self, tenant, name, url, model_name, model_version,
                       model_type, confidence, created_by):
                calls.append((
                    "create", tenant, name, url, model_name, model_version,
                    model_type, confidence, created_by,
                ))
                return {"id": "mlb-new-1", "url": url}

        sqlalchemy = types.ModuleType("sqlalchemy")
        sqlalchemy.select = lambda _model: Query()
        labelstudio = types.SimpleNamespace(ml_backend_url=configured_url)
        modules = {
            "sqlalchemy": sqlalchemy,
            "app": types.ModuleType("app"),
            "app.db": types.ModuleType("app.db"),
            "app.db.session": types.SimpleNamespace(SessionLocal=Database),
            "app.models": types.ModuleType("app.models"),
            "app.models.orm": types.SimpleNamespace(MLBackend=MLBackend),
            "app.services": types.ModuleType("app.services"),
            "app.services.mlbackend": types.SimpleNamespace(MLBackendService=MLBackendService),
            "app.core": types.ModuleType("app.core"),
            "app.core.config": types.SimpleNamespace(
                config=types.SimpleNamespace(labelstudio=labelstudio)
            ),
        }
        path = ROOT / "charts/nereidforge/scripts/seed_ml_backend.py"
        spec = importlib.util.spec_from_file_location("nf_seed_ml_backend_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch.dict(os.environ, env or {}, clear=False):
            spec.loader.exec_module(module)
            if expect_failure:
                with self.assertRaises(RuntimeError):
                    module.main()
            else:
                module.main()
        return calls

    def test_seed_creates_preset_from_config_url_with_defaults(self):
        calls = self.run_seed(existing=False)
        self.assertEqual(calls[-1], "closed")
        created = calls[0]
        self.assertEqual(created[0], "create")
        self.assertEqual(created[1], "default")
        self.assertEqual(created[2], "system-ml-backend")
        self.assertEqual(created[3], self.CONFIGURED_URL)
        self.assertEqual(created[4:7], ("PP-YOLOE", "v1", "cnn"))
        self.assertEqual(created[7], 0.6)
        self.assertEqual(created[8], "system-seed")

    def test_seed_is_noop_when_preset_already_registered(self):
        self.assertEqual(self.run_seed(existing=True), ["closed"])

    def test_env_override_url_and_metadata_win_over_config(self):
        override_url = "http://ml-backend-custom.nereidforge:9090"
        calls = self.run_seed(existing=False, env={
            "NF_MLBACKEND_URL": override_url,
            "NF_MLBACKEND_NAME": "renamed-preset",
            "NF_MLBACKEND_MODEL_NAME": "OtherModel",
            "NF_MLBACKEND_CONFIDENCE": "0.42",
        })
        created = calls[0]
        self.assertEqual(created[3], override_url)
        self.assertEqual(created[2], "renamed-preset")
        self.assertEqual(created[4], "OtherModel")
        self.assertEqual(created[7], 0.42)

    def test_missing_url_everywhere_fails_before_any_write(self):
        calls = self.run_seed(existing=False, configured_url="", expect_failure=True)
        self.assertEqual(calls, [])


class DirectHelmDocumentationTest(unittest.TestCase):
    def test_readme_documents_oci_order_without_an_nf_source_installer(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        application = readme.index("helm upgrade --install nereidforge ")
        ml_backend = readme.index("helm upgrade --install ml-backend")
        self.assertLess(application, ml_backend)
        self.assertIn("40_check-nf-handoff.sh", readme)
        self.assertIn("values.generated.yaml", readme)
        self.assertIn("operator-values.yaml", readme)
        # The two retired standalone releases must not be documented anymore.
        self.assertNotIn("helm upgrade --install nereidforge-backend", readme)
        self.assertNotIn("helm upgrade --install nereidforge-frontend", readme)


if __name__ == "__main__":
    unittest.main(verbosity=2)
