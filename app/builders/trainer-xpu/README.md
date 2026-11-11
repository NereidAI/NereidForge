<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# trainer-xpu — Intel GPU (XPU) PyTorch runtime base

Base image for NereidForge Intel GPU training (multi-vendor GPU support,
see ADR-012). Targets Intel discrete GPUs on the new **xe** KMD (Battlemage /
Xe2, e.g. Arc B580) and recent i915 dGPUs. The host provides the kernel driver
(`/dev/dri/renderD*`); this image provides the userspace compute stack plus
prebuilt PyTorch XPU — no from-source torch build.

Contents:
- Ubuntu 24.04 userspace: NEO OpenCL `intel-opencl-icd`, Level Zero
  `libze1` / `libze-intel-gpu1`, `intel-ocloc`, `xpu-smi`, `clinfo` from the
  Intel graphics `unified` apt channel (Battlemage needs NEO >= 24.13, newer
  than noble universe).
- Python 3.12 venv at `/opt/xpu` with `torch==2.14.0+xpu` and oneAPI runtime
  (oneCCL / oneMKL / triton-xpu) from the PyTorch XPU wheel index injected
  through the `TORCH_XPU_INDEX` build arg.

Mirrors (root treaty §17):
- No mirror is hardcoded; regional builds inject `APT_MIRROR` and generic pip
  endpoints from the environment, whose empty values select pristine sources.
- `TORCH_XPU_INDEX` is mandatory and must identify the official or mirrored
  XPU wheel index. Generic PyPI is not a valid source for `torch+xpu`.
- Only the Intel graphics apt repo is pulled upstream (no regional mirror).

Build:

```bash
export TORCH_XPU_INDEX=<official-or-mirrored-xpu-wheel-index>
docker build --build-arg TORCH_XPU_INDEX="$TORCH_XPU_INDEX" \
  -t ghcr.io/nereidai/trainer-xpu-base:0.0.0-dev .
```

Publication uses the release workflow or the documented registry-profile
build path; a local build must never be pushed directly to GHCR.

Smoke test on a node advertising `gpu.intel.com/xe` (deployed by
`infra/scripts/manifests/intel-gpu-plugin.yaml`):

```bash
sudo docker run --rm --device /dev/dri:/dev/dri \
  ghcr.io/nereidai/trainer-xpu-base:0.0.0-dev \
  python -c "import torch;print(torch.__version__,torch.xpu.is_available(),torch.xpu.device_count());[print(i,torch.xpu.get_device_name(i)) for i in range(torch.xpu.device_count())]"
```

Downstream trainers (`trainer-llm-xpu`, `trainer-cv-xpu`) `FROM` this image and
add LLaMA-Factory / ultralytics respectively. The Intel plugin mounts only the
render node(s) assigned to the container, so an iGPU on the same host is not
exposed when the workload requests `gpu.intel.com/xe`.

## Verified on 2x Intel Arc B580 (Battlemage / xe KMD), 2026-09

Host KMD is `xe`; userspace must be the kobuk/client stack (NEO 26.x). Using
the noble `unified` repo (NEO 25.18) aborts inside the Level-Zero command
encoder (`command_encoder_xehp_and_later.inl`) on first XPU compute — do not
use it for Battlemage.

- `torch==2.14.0+xpu`, `torch.xpu.is_available()=True`, device_count=2, both
  reported as `Intel(R) Arc(TM) B580 Graphics`. FP32 and BF16 matmul pass.
- **Ultralytics 8.4.147 trains on XPU**: `yolo detect train device=xpu:0 ...`
  completes, reports `XPU:0 (Intel(R) Arc(TM) B580 Graphics)`, AMP check
  passes natively (`AMP: checks passed`), loss decreases, weights export.
  **Required:** install the matching **XPU** torchvision from the injected
  XPU wheel index, e.g.
  `pip install --no-deps --index-url "$TORCH_XPU_INDEX" torchvision==0.29.0+xpu`
  (torch 2.14 <-> torchvision 0.29). A default CUDA/CPU torchvision raises
  `RuntimeError: operator torchvision::nms does not exist` during warmup. Also
  pin `numpy<2.3` for that ultralytics version.
- Open questions / known limits:
  - Multi-GPU collectives: stock torch XPU registers only `gloo` (CPU tensors);
    there is no `ccl`/`nccl` backend without `oneccl_bindings_for_pytorch`,
    which has no ready wheel for torch 2.14 yet. v1 multi-GPU should plan for
    single-GPU QLoRA (a 12GB B580 fits 7B 4-bit) or pin a torch+ccl pair.
  - 4-bit QLoRA on XPU (bitsandbytes XPU vs auto-round) is validated separately
    in the LLM bring-up (gate 2).

## Gate 2 — LLM LoRA / QLoRA (verified 2026-09)

- BF16 LoRA with `transformers>=4.46` + `peft` on `device=xpu:0` trains (Qwen2.5
  0.5B, loss decreases end-to-end). Standard PyTorch optimizer, no special stack.
- **4-bit QLoRA works** with `bitsandbytes==0.50.x` XPU backend: a base layer is
  a real `Linear4bit`, NF4 forward+backward and `bnb.optim.AdamW8bit` step all
  run on XPU (Qwen2.5 0.5B verified). bnb XPU kernels are JIT-compiled through
  **triton-XPU**, which is why the image ships `gcc g++ make python3-dev
  libze-dev libze-intel-gpu-dev` (JIT needs a C/C++ compiler, `Python.h` and
  `level_zero/ze_api.h` at *runtime*, with `CC=gcc CXX=g++`). First call has a
  one-time kernel compile cost. A full 7B QLoRA fit on a 12GB B580 is the next
  scaling check (same code path); regional HF downloads inject `HF_ENDPOINT`
  from the environment (root treaty §17), never as a committed default.

