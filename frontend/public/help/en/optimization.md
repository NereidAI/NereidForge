<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Model Optimization

## Flow
1. Create an optimization job: select a source model (version) and one or more target devices.
2. Multiple targets: several devices can be selected (CPU/OpenVINO/NVIDIA, and others); each target produces an independent job.
3. The platform runs ONNX graph optimization and int8 quantization automatically; artifacts are stored in SeaweedFS and registered as new model versions (with a target-specific suffix).
4. Optimized versions carry real metrics such as speedup and size reduction, enabling side-by-side selection in the model registry.

## Live Progress
- Running jobs show four milestones: baseline evaluation, model optimization, quality-gate validation, and finalization.
- The percentage counts completed workflow milestones, not estimated time remaining; it reaches 100% only after the whole KFP run succeeds.
- The detail page shows live executor logs. `[opt] SUCCESS` means the optimizer finished; quality-gate validation may still be running.

## Prerequisites
- The source model must have an ONNX artifact (produced by training or manually uploaded as .onnx).
- Targets such as NVIDIA/TensorRT require the corresponding hardware/toolchain (jobs are marked pending when the cluster has no matching hardware).

## FAQ
- **Where does the speedup number come from?** A real benchmark run by optimize.py (ONNX Runtime), not an estimate.
- **How do I use an optimized version?** It is registered under the source model and can be deployed or evaluated.

## Next Steps
After optimization, open the Evaluation page to verify the optimized version's accuracy (accuracy_drop), or open the Models page to deploy it.
