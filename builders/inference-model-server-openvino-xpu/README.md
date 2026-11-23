<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# inference/model-server-openvino-xpu

NF-built OVMS variant for Intel Arc dGPU serving (KServe `--target_device GPU`).

Layered on the upstream Ubuntu-24.04 based `openvino/model_server` image; adds
the kobuk Intel compute userspace so the bundled OpenVINO GPU plugin finds
Arc hardware:

- NEO (`intel-opencl-icd`, `libze-intel-gpu1`) from PPA `kobuk-team/intel-graphics`
  — Battlemage (Xe2) needs NEO >= 26; no domestic mirror exists (treaty-sanctioned
  exception, same stack as `trainer-xpu`).
- `libdrm-intel1` from noble-updates (>= 2.4.122), matching the validated
  trainer-xpu userspace on the new `xe` KMD (base image ships stock 2.4.120).

## Verified on 2x Intel Arc B580, 2026-10-26

- Pod scheduled with `gpu.intel.com/xe: 1`; device plugin chowns `/dev/dri/*`
  to the container uid.
- OVMS logs `Available devices for Open VINO: CPU, GPU`, model loads on GPU.
- FP32 ONNX yolov8n (mask): KServe v2 **binary tensor** inference on GPU
  p50 5.5ms / p99 11.2ms / 172 QPS (single stream, batch 1), outputs match CPU.
- **The container must run as root** (`securityContext.runAsUser=0`):
  with the image's stock uid 5000, NEO enumerates zero platforms even though
  the device node is chowned to that uid and ioctl works. Reproduces with any
  userspace (trainer image forced to uid 5000 fails the same way); uid 1000
  works. The KServe manifest adds the securityContext only for Intel GPU
  deploys; CPU deploys stay on the stock image as uid 5000.

## Known GPU caveat: int8-dynamic IR

An int8-*dynamic* QDQ IR loads fine on the GPU plugin but returns all-NaN
outputs (empty detections). GPU packages must use numerically GPU-verified
artifacts (FP16 static shape, or calibrated INT8); see the edge-model package
verification gate. FP32/FP16 are verified good.

## Build

```bash
# Local build only (no make target pushes). CI/GHCR uses the Dockerfile
# default base (openvino/model_server public tag).
make build-ovms-xpu
# Local/offline: pass a node-mirror-resolvable OVMS base when required
make build-ovms-xpu OVMS_BASE_IMG=<registry>/inference/model-server-openvino:0.0.1
```

The tag follows the tag-sourced version model like every NF-built image
(root treaty §19.5): an exact `v*` git-tag checkout produces the bare SemVer
tag (leading `v` stripped, e.g. `0.0.1`); any non-tag checkout builds the
rolling `0.0.0-dev` dev channel. GHCR publication runs in the
`publish-images` GitHub Actions workflow; publishing the local tag to a
private/air-gapped registry is an operator-only manual `docker push` with a
user-supplied hostname (or NereidOverview `harbor-sync.sh`).
