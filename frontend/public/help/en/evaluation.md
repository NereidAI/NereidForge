<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Model Evaluation

## Flow
1. Create an evaluation job: select a model (version) and a dataset.
2. Adjustable parameters: confidence threshold (0.25 recommended), maximum number of images (0 means all), and task type.
3. Evaluation defaults to the validation subset produced during training; any dataset version can be selected instead (it is converted to YOLO automatically).
4. On completion, real metrics are shown: mAP50 / Precision / Recall / F1 / latency.

## Prerequisites
- The model must have an ONNX artifact.
- The dataset must be parseable into YOLO format (handled automatically by the platform).

## FAQ
- **Why is precision 0?** The model has not converged, or the threshold is so high that too few predictions are produced. Lower the confidence threshold and retry.
- **Which data is evaluated?** The training validation subset by default, to avoid data leakage.

## Next Steps
When the metrics meet the target, open the Models page to deploy, or return to the Optimization page for further tuning.
