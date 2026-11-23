# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""End-to-end VLM LoRA training smoke test (contract-identical to the UI).

Submits an Intel XPU VLM fine-tuning job through the same service layer the
Web UI uses (JobService.create_job + start_job), so it exercises dataset
selection, KFP run creation, TrainJob creation, trainer pod logs/history and
auto model registration without needing the production admin password.

Run inside a deployed API pod with in-cluster KFP/S3/DB access. Copy this file
to /tmp/e2e_vlm_training.py and run it with PYTHONPATH=/app after setting the
required operator inputs below. Every required value must be non-empty; a
missing value stops the script before it creates a job.

Required environment variables:
- E2E_DATASET_ID: ID of an existing llm-sft dataset backed by external S3.
- E2E_DATASET_VERSION: existing version of that dataset.
- E2E_BASE_MODEL: S3 URI of a directory containing the compatible VLM base
  model files, for example s3://<bucket>/<base-model-prefix>/.
- E2E_NODE: name of the node to which the training job should be pinned.
- E2E_JOB_NAME: unique name for the training job to create.

E2E_TENANT optionally selects the tenant and defaults to "default". The
selected node must advertise gpu.intel.com/xe, and the cluster must have the
KFP TrainJob RBAC required by the existing training workflow.
"""
import os

from app.db.session import SessionLocal
from app.services import Services


def _required_env(name: str) -> str:
    """Return a non-empty operator input or stop before creating a job."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} to a non-empty value before running this E2E test")
    return value


DATASET_ID = _required_env("E2E_DATASET_ID")
DATASET_VERSION = _required_env("E2E_DATASET_VERSION")
BASE_MODEL = _required_env("E2E_BASE_MODEL")
NODE = _required_env("E2E_NODE")
JOB_NAME = _required_env("E2E_JOB_NAME")
TENANT = os.environ.get("E2E_TENANT", "default").strip() or "default"


def main() -> None:
    """Create and start one VLM LoRA job; print ids for later polling."""
    svc = Services(SessionLocal())
    params = {"dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION}
    cfg = {
        "task_type": "vlm",
        "accelerator": "intel",
        "epochs": 5,
        "batch": 2,
        "base_model": BASE_MODEL,
        "node": NODE,
        "llm": {
            "template": "qwen",
            "cutoff_len": 512,
            "num_train_epochs": 5,
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        },
    }
    job = svc.job.create_job(
        "training", TENANT, JOB_NAME, "VLM E2E live observability", "admin",
        params, cfg,
    )
    started = svc.job.start_job("training", TENANT, job["id"])
    print("JOB", job["id"], "RUN", started.get("kfp_run_id"))


if __name__ == "__main__":
    main()
