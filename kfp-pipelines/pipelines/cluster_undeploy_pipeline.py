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


delete_manifests = _load_component("delete_manifests_step.py").delete_manifests


@dsl.pipeline(
    name="cluster-undeploy",
    description="Delete Kubernetes manifests in-cluster (ISVC / ingress / deploy).",
)
def cluster_undeploy_pipeline(manifests_json: str) -> str:
    step = delete_manifests(manifests_json=manifests_json)
    step.set_display_name("Delete manifests")
    # Mutation: never let an identical undeploy be served from cache (a cached
    # delete would report success without removing a freshly re-deployed ISVC).
    step.set_caching_options(False)
    return step.output
