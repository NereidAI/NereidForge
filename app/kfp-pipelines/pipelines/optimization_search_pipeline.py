# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os

from kfp import dsl


def _load_component(filename: str):
    path = os.path.join(
        os.path.dirname(__file__), "..", "components", filename
    )
    spec = importlib.util.spec_from_file_location(
        f"nf_component_{filename.rsplit('.', 1)[0]}", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_katib_experiment = _load_component("run_katib_experiment_step.py").run_katib_experiment


@dsl.pipeline(
    name="optimization-search",
    description="Run a Katib grid-search Experiment in-cluster.",
)
def optimization_search_pipeline(
    experiment_json: str,
    s3_access_key: str,
    s3_secret_key: str,
    job_id: str = "",
    best_trial_s3_uri: str = "s3://nereidforge/optimized/best_trial.json",
    s3_endpoint: str = "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333",
    timeout_minutes: int = 360,
) -> str:
    step = run_katib_experiment(
        experiment_json=experiment_json,
        job_id=job_id,
        best_trial_s3_uri=best_trial_s3_uri,
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        timeout_minutes=timeout_minutes,
    )
    step.set_display_name("Run Katib experiment")
    # Creates an external Katib Experiment resource: a cached replay would
    # never submit a new search while reporting the old trial as current.
    step.set_caching_options(False)
    return step.output
