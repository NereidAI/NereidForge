# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Create or update NF pipelines and persist their KFP registration.

Usage (from repo root, via the repo .venv):
    ./.venv/bin/python app/kfp-pipelines/upload_pipelines.py
    ./.venv/bin/python app/kfp-pipelines/upload_pipelines.py training

The host-side default expects a port-forward on localhost:18888. Deployment runs
the same file in the API pod with KFP_HOST/KFP_CONFIG_HOST set to the cluster
Service. Existing IDs are reused; missing IDs are created or adopted by name and
saved after each step so an interrupted run can safely resume.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from kfp import client as kfp_client
from kfp import compiler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
BACKEND_DIR = SCRIPT_DIR.parent / "python-backend"
if BACKEND_DIR.is_dir():
    sys.path.insert(0, str(BACKEND_DIR))

from pipelines.evaluation_pipeline import evaluation_pipeline
from pipelines.optimization_pipeline import optimization_pipeline
from pipelines.training_pipeline import training_pipeline

HOST = os.environ.get("KFP_HOST", "http://localhost:18888")
CONFIG_HOST = os.environ.get("KFP_CONFIG_HOST", "http://ml-pipeline.kubeflow:8888")
NAMESPACE = os.environ.get("KFP_NAMESPACE", "nereidforge")
USERID = os.environ.get("KFP_USERID", "admin@nereidforge.local")
EXPERIMENT_NAME = os.environ.get("KFP_EXPERIMENT_NAME", "NereidForge")

PIPELINES = {
    "training": training_pipeline,
    "optimization": optimization_pipeline,
    "evaluation": evaluation_pipeline,
}
PIPELINE_DISPLAY_NAMES = {
    name: f"nereidforge-{name}-pipeline" for name in PIPELINES
}


def load_system_config() -> dict[str, Any]:
    """Read the shared platform configuration through the backend owner."""
    from app.db.session import SessionLocal
    from app.services.admin import AdminService

    db = SessionLocal()
    try:
        return AdminService(db).get_system_config()
    finally:
        db.close()


def save_system_config(config: dict[str, Any]) -> None:
    """Merge registration updates without replacing unrelated site settings."""
    from app.db.session import SessionLocal
    from app.services.admin import AdminService

    db = SessionLocal()
    try:
        AdminService(db).update_system_config(config)
    finally:
        db.close()


def configure_sdk_identity() -> None:
    """Scope SDK calls to the tenant namespace and its Kubeflow user header."""
    context_dir = Path(os.environ.get(
        "KFP_CONTEXT_DIR", str(Path.home() / ".config" / "kfp")
    ))
    context_dir.mkdir(parents=True, exist_ok=True)
    context_file = context_dir / "context.json"
    context_file.write_text(json.dumps({
        "namespace": NAMESPACE,
        "client_authentication_header_name": "kubeflow-userid",
        "client_authentication_header_value": USERID,
    }))
    context_file.chmod(0o600)


def find_pipeline_id(kc: Any, display_name: str) -> str:
    """Adopt an already uploaded pipeline after a lost or partial DB hand-off."""
    page_token = ""
    while True:
        response = kc.list_pipelines(
            page_token=page_token, page_size=100, namespace=NAMESPACE,
        )
        for pipeline in response.pipelines or []:
            name = getattr(pipeline, "display_name", "") or getattr(pipeline, "name", "")
            if name == display_name:
                return pipeline.pipeline_id
        page_token = response.next_page_token or ""
        if not page_token:
            return ""


def synchronize_pipelines(
    kc: Any,
    config: dict[str, Any],
    *,
    only: str = "",
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Ensure the experiment and selected pipelines exist, then upload versions.

    Every newly discovered or created ID is persisted through on_update before
    the next external operation. Existing IDs never cause duplicate pipelines.
    """
    def persist() -> None:
        if on_update:
            on_update(config)

    if not config.get("kfp_experiment_id"):
        try:
            experiment = kc.get_experiment(
                experiment_name=EXPERIMENT_NAME, namespace=NAMESPACE,
            )
        except Exception:
            experiment = kc.create_experiment(
                name=EXPERIMENT_NAME, namespace=NAMESPACE,
            )
        config["kfp_experiment_id"] = experiment.experiment_id
        persist()
        print(f"[register] experiment {EXPERIMENT_NAME}: ready")

    version = time.strftime("v%Y%m%d-%H%M%S")
    with tempfile.TemporaryDirectory(prefix="nf-pipeline-upload-") as workdir:
        for name, fn in PIPELINES.items():
            if only and only not in name:
                continue
            path = str(Path(workdir) / f"{name}.yaml")
            compiler.Compiler().compile(pipeline_func=fn, package_path=path)
            key = f"kfp_{name}_pipeline_id"
            pipeline_id = str(config.get(key) or "")
            display_name = PIPELINE_DISPLAY_NAMES[name]

            if not pipeline_id:
                pipeline_id = find_pipeline_id(kc, display_name)
                if pipeline_id:
                    config[key] = pipeline_id
                    persist()
                    print(f"[register] {name}: adopted existing pipeline")

            if not pipeline_id:
                pipeline = kc.upload_pipeline(
                    pipeline_package_path=path,
                    pipeline_name=display_name,
                    namespace=NAMESPACE,
                )
                config[key] = pipeline.pipeline_id
                persist()
                print(f"[register] {name}: created initial pipeline version")
                continue

            uploaded = kc.upload_pipeline_version(
                pipeline_package_path=path,
                pipeline_version_name=version,
                pipeline_id=pipeline_id,
            )
            print(f"[upload] {name}: uploaded {uploaded.pipeline_version_id}")


def main() -> None:
    """Load site config, synchronize KFP, and save each registration update."""
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    config = load_system_config()
    bindings = {
        "kfp_host": CONFIG_HOST,
        "kfp_namespace": NAMESPACE,
        "kfp_userid": USERID,
    }
    if any(config.get(key) != value for key, value in bindings.items()):
        config.update(bindings)
        save_system_config(config)

    configure_sdk_identity()
    kc = kfp_client.Client(host=HOST, verify_ssl=False, namespace=NAMESPACE)
    synchronize_pipelines(kc, config, only=only, on_update=save_system_config)


if __name__ == "__main__":
    main()
