# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Lazy CUDA initialization shared by TensorRT hardware operations."""

import importlib


def initialize_cuda() -> None:
    """initialize_cuda creates the PyCUDA context only for hardware execution.

    Raises:
        RuntimeError: PyCUDA is installed but no usable CUDA driver/device can
            initialize a context. Metric-only imports never call this function.
    """
    try:
        importlib.import_module("pycuda.autoinit")
    except Exception as exc:
        raise RuntimeError("A working CUDA driver and device are required for TensorRT execution") from exc
