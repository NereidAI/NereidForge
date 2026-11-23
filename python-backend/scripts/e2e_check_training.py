# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Poll training observability for one job (real trainer pod logs + history).

Companion to e2e_vlm_training.py. Run INSIDE the deployed api pod:

    kubectl exec -n nereidforge <api-pod> -- env PYTHONPATH=/app python \
        /tmp/e2e_check_training.py <job-id> [interval_seconds]

Prints trainer pod phase, S3-backed history point count, job progress/metrics
and the tail of real trainer-pod stdout on every tick until the job reaches a
terminal state.
"""
import sys
import time

from app.db.session import SessionLocal
from app.services import Services

TERMINAL = {"completed", "failed", "cancelled", "stopped"}


def main() -> None:
    """Tick until the job leaves the running/pending lifecycle."""
    job_id = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    svc = Services(SessionLocal())
    while True:
        hist = svc.job.training_history("default", job_id)
        pods = svc.job.trainer_pods("default", job_id)
        job = svc.job.get_job_dict("training", "default", job_id)
        print(
            f"status={job['status']} progress={job['progress']} "
            f"history={len(hist)} pods={[(p.get('name'), p.get('phase')) for p in pods]}"
        )
        if hist:
            print("  last:", hist[-1])
        if job["status"] in TERMINAL:
            print("metrics:", job["metrics"])
            print("--- trainer log tail ---")
            print(svc.job.trainer_logs("default", job_id, tail=30)[-2000:])
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
