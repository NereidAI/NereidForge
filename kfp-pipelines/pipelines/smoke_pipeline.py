# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""KFP smoke-test pipeline.

Submits a single echo step through the normal cluster-op channel
(submit -> wait -> result) to verify KFP compile/submit/scheduling/execution
and the NereidForge run-state mapping without touching cluster RBAC.
"""

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


smoke_echo = _load_component("smoke_step.py").smoke_echo


@dsl.pipeline(
    name="nf-smoke",
    description="Echo smoke test verifying KFP submit -> run -> wait.",
)
def smoke_pipeline(message: str = "pong") -> str:
    step = smoke_echo(message=message)
    step.set_display_name("Smoke echo")
    # The smoke test proves live execution end-to-end; a cached skip would
    # report connectivity that was never tested.
    step.set_caching_options(False)
    return step.output
