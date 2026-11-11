# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test echo step (requires no cluster RBAC).

Runs a trivial echo inside a KFP pod to verify the whole compile -> submit ->
schedule -> execute -> wait chain. Kept dependency-free on purpose so it can
run in any tenant namespace with only the default SA.
"""

from kfp import dsl

from images import nf_image


@dsl.component(base_image=nf_image("runtime"))
def smoke_echo(message: str) -> str:
    return message
