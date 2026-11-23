# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""End-to-end custom training script smoke test (I-11).

Proves the whole custom-script path against the live cluster, end to end:

  1. upload a tiny user training script to SeaweedFS,
  2. create + start a real *training* job pinned to a CPU ``custom_image``
     (the built-in cv image, so no new pull) plus ``custom_script_uri``,
  3. the KFP train step turns the selector into a fetch-and-run trainer
     command; the real TrainJob/trainer pod downloads the script over the
     filer anonymous HTTP endpoint and ``runpy``-executes it,
  4. the script uploads a dummy ``.pt`` artifact to ``$OUTPUT_DIR`` and writes
     **no** metrics.json, deliberately forcing the worker's
     ``_discover_artifact_uri`` registration fallback.

Run INSIDE the deployed api pod (in-cluster KFP/S3/K8s + DB access):

    APOD=$(sudo kubectl get pods -n nereidforge \
        --field-selector=status.phase=Running -o name \
        | grep '^pod/nereidforge-api' | sort | tail -1 | cut -d/ -f2)
    sudo kubectl cp -n nereidforge \
        python-backend/scripts/e2e_custom_training.py "$APOD:/tmp/"
    sudo kubectl exec -n nereidforge "$APOD" -- \
        env PYTHONPATH=/app python /tmp/e2e_custom_training.py

The script deliberately leaves the created model/version in place as evidence;
the job name is suffixed with a timestamp. Exits non-zero on any failed check.
"""
from __future__ import annotations

import sys
import time

import boto3
import botocore.config

from app.core.config import config
from app.core.images import nf_image
from app.db.session import SessionLocal
from app.services import Services

# CPU cv image = the custom image for this test (python + boto3, already
# cached on the nodes); the point is the custom command/script, not new bits.
CUSTOM_IMAGE = nf_image("trainer/torch-cpu")
SCRIPT_KEY = "e2e-probe/i11/custom_train.py"
# Marker the in-pod user script prints after it uploads its artifact.
ARTIFACT_MARKER = "I11_ARTIFACT_UPLOADED"

# The user-supplied training script. It performs NO real training: it just
# uploads a recognizable dummy weight file into $OUTPUT_DIR and exits. It does
# not emit *.metrics.json, so registration must come from artifact discovery.
USER_SCRIPT = f"""
import os, boto3, botocore.config
ep=os.environ["S3_ENDPOINT"]; ak=os.environ["S3_ACCESS_KEY"]
sk=os.environ["S3_SECRET_KEY"]; bucket=os.environ["S3_BUCKET"]
out=os.environ.get("OUTPUT_DIR","").rstrip("/")
rest=out[5:] if out.startswith("s3://") else out
seg=rest.split("/",1)
prefix=seg[1].strip("/") if len(seg)>1 else ""
aid=os.environ.get("ARTIFACT_ID") or "model"
key=(prefix+"/" if prefix else "")+aid+".pt"
s3=boto3.client("s3", endpoint_url=ep, aws_access_key_id=ak,
    aws_secret_access_key=sk, region_name="us-east-1",
    config=botocore.config.Config(signature_version="s3v4"))
body=b"NEREIDFORGE-I11-CUSTOM-ARTIFACT\\n"+os.urandom(2048)
s3.put_object(Bucket=bucket, Key=key, Body=body)
print("{ARTIFACT_MARKER}", "s3://%s/%s" % (bucket, key), flush=True)
""".lstrip()


def _s3() -> boto3.client:
    """Authenticated S3 client (S3 gateway), config-driven."""
    return boto3.client(
        "s3",
        endpoint_url=config.storage.s3_endpoint,
        aws_access_key_id=config.storage.s3_access_key,
        aws_secret_access_key=config.storage.s3_secret_key,
        region_name="us-east-1",
        config=botocore.config.Config(signature_version="s3v4"),
    )


def main() -> None:
    """Create/start the custom-script job, poll it, and assert registration."""
    bucket = config.storage.bucket or "nereidforge"
    s3 = _s3()

    # 1. Publish the user script.
    s3.put_object(Bucket=bucket, Key=SCRIPT_KEY, Body=USER_SCRIPT.encode())
    script_uri = f"s3://{bucket}/{SCRIPT_KEY}"
    print(f"uploaded user script -> {script_uri}")

    svc = Services(SessionLocal())
    suffix = int(time.time())
    name = f"i11-custom-{suffix}"
    params = {
        "custom_image": CUSTOM_IMAGE,
        "custom_script_uri": script_uri,
        # Passthrough s3 dataset keeps dataset_step instant and offline.
        "dataset_uri": f"s3://{bucket}/e2e-probe/i11/dummy/",
    }
    # Force the CPU vendor profile so the TrainJob requests no GPU and
    # schedules immediately; detection => pytorch/cv branch (not LLM).
    cfg = {
        "accelerator": "cpu",
        "task_type": "detection",
        "epochs": "1",
        "imgsz": "640",
        "batch": "1",
        "base_model": "ppyoloe_crn_l_300e_coco.pdparams",
    }

    # 2. Create + start through the real service lifecycle (same as the API).
    job = svc.job.create_job("training", "default", name, "I-11 e2e", "e2e", params, cfg)
    job_id = job["id"]
    print(f"created job {job_id} ({name})")
    svc.job.start_job("training", "default", job_id)
    print(f"started job {job_id}; polling ...")

    # 3. Poll until terminal.
    terminal = {"completed", "failed", "cancelled", "stopped"}
    deadline = time.time() + 900
    final = None
    while time.time() < deadline:
        final = svc.job.get_job_dict("training", "default", job_id)
        pods = svc.job.trainer_pods("default", job_id)
        print(
            f"status={final['status']} progress={final['progress']} "
            f"pods={[(p.get('name'), p.get('phase')) for p in pods]}"
        )
        if final["status"] in terminal:
            break
        time.sleep(10)

    assert final is not None, "no job state observed"
    if final["status"] != "completed":
        print("--- trainer log tail (failure) ---")
        print(svc.job.trainer_logs("default", job_id, tail=200)[-4000:])
        raise SystemExit(f"job did not complete: status={final['status']}")

    # finish_job flips the job to "completed" one write BEFORE it persists the
    # discovered/registered model link, so settle-poll until that second write
    # lands (bounded) instead of asserting on the first completed observation.
    for _ in range(12):
        if (final.get("results") or {}).get("registered_model"):
            break
        time.sleep(5)
        final = svc.job.get_job_dict("training", "default", job_id)

    # 4a. The trainer pod really ran the downloaded user script.
    logs = svc.job.trainer_logs("default", job_id, tail=400)
    print("--- trainer log tail ---")
    print(logs[-2000:])
    assert ARTIFACT_MARKER in logs, "custom-script marker missing from trainer logs"

    # 4b. The artifact exists in the job output prefix.
    artifact_key = f"models/{job_id}/{job_id}.pt"
    head = s3.head_object(Bucket=bucket, Key=artifact_key)
    print(f"artifact present: s3://{bucket}/{artifact_key} size={head['ContentLength']}")

    # 4c. The worker registered it via discovery (no metrics.json was written).
    metrics = final.get("metrics") or {}
    results = final.get("results") or {}
    registered = results.get("registered_model")
    print("metrics:", metrics)
    print("registered_model:", registered)
    assert str(metrics.get("weights_uri", "")).endswith(f"{job_id}.pt"), \
        f"weights_uri not discovered: {metrics.get('weights_uri')}"
    assert registered, "registered_model link missing on job results"

    print("\nI-11 E2E PASS: custom script ran, artifact discovered, model registered.")


if __name__ == "__main__":
    sys.exit(main())
