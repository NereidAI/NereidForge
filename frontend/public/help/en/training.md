<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Model Training

## Target, Architecture, and Weights
- **Target model (output)**: the registry model that receives the new trained version. It selects neither the network nor its initialization. When left empty, the name is derived from the architecture or checkpoint.
- **Architecture**: the network to build, for example ResNet-50 or Faster R-CNN. Architecture and weights are two independent dimensions.
- **Starting weights**: the native training checkpoint used in Existing-checkpoint mode, sourced from the base model pool or a registered version. ONNX-only versions cannot initialize training.

## Initialization
- **Pretrained weights**: use the trainer's default pretrained initialization for the selected architecture, preserving the established default behavior.
- **From scratch**: build the selected architecture without pretrained weights or a checkpoint, using random initialization.
- **Existing checkpoint**: continue training from a base-pool entry or a registered version. Architecture can be left empty and is resolved from version metadata, a recognized checkpoint name, or the task default; select it explicitly only to override that resolution.

## LLM/VLM Base Models
- An LLM/VLM base model (from the model pool) fixes both structure and parameters, so the separate vision architecture and initialization controls do not apply.

## Flow
1. Select the task type, dataset, and annotated version.
2. Optionally select a target model, then confirm the architecture, initialization mode, and hyperparameters.
3. On start, the KFP training pipeline runs and progress is recorded.
4. On completion, the trained version is registered automatically and training curves are generated.

## FAQ
- **Where are the curves?** Open the Training curves tab on the job detail page.
- **How do I continue training?** Choose Existing checkpoint as the initialization mode, then pick a registered version or pool entry containing native training weights.
- **What happens after an interruption?** A failed job can be restarted. To reuse the previous parameters, retrain from the original job and verify the starting weights before submitting.
