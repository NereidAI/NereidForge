# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Multi-vendor GPU accelerator profiles (single source of truth, ADR-012).

Each GPU vendor ships an *independent* trainer image/dependency stack — we never
merge vendors into one image. This table is the only place that maps a vendor to
its Kubernetes extended resource, its vision (`cv_image` = torch, plus the
optional `paddle_image` family) and LLM (`llm_image`) images, the device string
the trainer entrypoint expects, extra env, node selectors and tolerations.

Vendors:
  cpu    - no GPU; the CPU torch image.
  nvidia - nvidia.com/gpu, CUDA images (torch-cu / llm-cu), device "0".
  amd    - amd.com/gpu, ROCm/HIP images (torch-rocm / llm-rocm), device "0"
           (torch.cuda is HIP); HSA_OVERRIDE_GFX_VERSION may be set in env for
           RDNA parts without prebuilt gfx kernels.
  intel  - gpu.intel.com/xe (discrete Arc / Battlemage "xe" KMD), native
           torch.xpu images (torch-xpu / llm-xpu), device "xpu:0". v1 is single
           card only (stock torch XPU has no usable multi-card CCL yet).

The trainer image tag follows the repo-wide unified release tag.
"""

from __future__ import annotations

from typing import Any, Optional

from app.core.images import nf_image

# Vendor id -> profile. `resource` is None for CPU (no extended resource key).
# Image refs resolve through the registry profile (public/GHCR flat or
# harbor nested) via app.core.images.nf_image.
ACCELERATOR_PROFILES: dict[str, dict[str, Any]] = {
    "cpu": {
        "label": "CPU",
        "resource": None,
        "device": "cpu",
        "cv_image": nf_image("trainer/torch-cpu"),
        "paddle_image": nf_image("trainer/paddle-cpu"),
        "llm_image": None,
        "env": [],
        "node_selector": {},
        "tolerations": [],
        "distributed": True,
    },
    "nvidia": {
        "label": "NVIDIA CUDA",
        "resource": "nvidia.com/gpu",
        "device": "0",
        "cv_image": nf_image("trainer/torch-cu"),
        "paddle_image": nf_image("trainer/paddle-cu"),
        "llm_image": nf_image("trainer/llm-cu"),
        "env": [],
        "node_selector": {},
        "tolerations": [],
        "distributed": True,
    },
    "amd": {
        "label": "AMD ROCm",
        "resource": "amd.com/gpu",
        "device": "0",
        "cv_image": nf_image("trainer/torch-rocm"),
        # No official Paddle ROCm wheels: the AMD family ships torch only.
        "paddle_image": None,
        "llm_image": nf_image("trainer/llm-rocm"),
        # Empty by default: only set HSA_OVERRIDE_GFX_VERSION for RDNA parts that
        # lack prebuilt gfx kernels. Operators can extend this per deployment.
        "env": [],
        "node_selector": {},
        "tolerations": [],
        "distributed": True,
    },
    "intel": {
        "label": "Intel XPU (Arc)",
        "resource": "gpu.intel.com/xe",
        "device": "xpu:0",
        "cv_image": nf_image("trainer/torch-xpu"),
        # No Paddle XPU wheel exists on PyPI (zero xpu builds of paddlepaddle):
        # the Intel family ships torch only (ADR-014 D-01, corrected 2026-10).
        "paddle_image": None,
        "llm_image": nf_image("trainer/llm-xpu"),
        "env": [],
        "node_selector": {},
        "tolerations": [],
        # stock torch 2.14 XPU only registers gloo (CPU); no oneCCL binding yet.
        "distributed": False,
    },
}

# Accepted selector spellings -> canonical vendor id.
_VENDOR_ALIASES = {
    "cpu": "cpu",
    "none": "cpu",
    "nvidia": "nvidia",
    "nv": "nvidia",
    "cu": "nvidia",
    "cuda": "nvidia",
    "nvidia.com/gpu": "nvidia",
    "amd": "amd",
    "rocm": "amd",
    "hip": "amd",
    "amd.com/gpu": "amd",
    "intel": "intel",
    "xpu": "intel",
    "arc": "intel",
    "gpu.intel.com/xe": "intel",
    "gpu.intel.com/i915": "intel",
}


def normalize_vendor(value: Optional[str]) -> Optional[str]:
    """Map any accepted vendor/resource spelling to a canonical vendor id."""
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    return _VENDOR_ALIASES.get(key)


def get_profile(vendor: str) -> dict[str, Any]:
    """Return the profile dict for a canonical vendor id (defaults to cpu)."""
    return ACCELERATOR_PROFILES.get(vendor, ACCELERATOR_PROFILES["cpu"])


def resolve_vendor(
    job_config: Optional[dict] = None,
    params: Optional[dict] = None,
    system_accelerator: str = "",
    gpu_count: Any = 0,
) -> str:
    """Decide the canonical vendor for a job.

    Precedence: explicit per-job selector (config then params) -> system-wide
    trainer_accelerator_type -> nvidia when GPUs are requested (legacy default)
    -> cpu.
    """
    for src in (job_config or {}, params or {}):
        for key in ("accelerator", "vendor", "gpu_vendor", "accelerator_type"):
            vendor = normalize_vendor(src.get(key))
            if vendor:
                return vendor
    sys_vendor = normalize_vendor(system_accelerator)
    if sys_vendor:
        return sys_vendor
    try:
        if int(str(gpu_count or "0")) > 0:
            return "nvidia"
    except (TypeError, ValueError):
        pass
    return "cpu"
