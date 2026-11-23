# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
NereidForge KFP Components Package

Only live components are exported here. The Q07 simplification removed the
checkpoint/export/register/event steps (training artifacts are written
straight to SeaweedFS; registration runs in the worker). The old optimizer
(opt_*) / evaluator (eval_*) placeholders referenced deleted images
(legacy runtime tags) and were pruned — optimizer work happens via the generic
optimization pipeline (optimize.py) and per-family ONNX converters.
"""

from .train_step import train_step
from .dataset_step import dataset_step

__all__ = [
    "train_step",
    "dataset_step",
]
