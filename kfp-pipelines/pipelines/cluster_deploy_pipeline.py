# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import os

import importlib.util

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


apply_manifests = _load_component("apply_manifests_step.py").apply_manifests


@dsl.pipeline(
    name="cluster-deploy",
    description="Apply Kubernetes manifests in-cluster (ISVC / ingress / deploy).",
)
def cluster_deploy_pipeline(
    manifests_json: str,
    wait_isvc_name: str = "",
    wait_isvc_namespace: str = "nereidforge",
    timeout_minutes: int = 30,
) -> str:
    step = apply_manifests(
        manifests_json=manifests_json,
        wait_isvc_name=wait_isvc_name,
        wait_isvc_namespace=wait_isvc_namespace,
        timeout_minutes=timeout_minutes,
    )
    step.set_display_name("Apply manifests")
    # Cluster mutation: identical manifests after an undeploy MUST re-create
    # the resources. KFP execution caching would otherwise skip the step and
    # replay the old "ready" output while the ISVC no longer exists.
    step.set_caching_options(False)
    return step.output
