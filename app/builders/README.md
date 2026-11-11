<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Builders — image build contexts and the Trainer SDK

Each subdirectory is a Docker build context or shared package for one
NereidForge image. Suite-built images use the public coordinate
`ghcr.io/nereidai/<flat-name>:<tag>`; resolve references through the repository
image helpers rather than constructing registry URLs in individual builds.

| Directory | Image or package | Category | Entrypoint |
|---|---|---|---|
| `sdk/` | `nereidforge-trainer` Python package | — | `nf-trainer` CLI |
| `trainer-torch/` | `trainer/torch-{cpu,cu,rocm,xpu}` vision tasks | train | `SCRIPT=train\|evaluate\|export\|catalog\|profile` |
| `trainer-paddle/` | `trainer/paddle-{cpu,cu}` detection | train | `SCRIPT=train\|evaluate\|export\|profile` |
| `trainer-llm-cu/`, `trainer-llm-rocm/`, `trainer-llm-xpu/` | `trainer/llm-{cu,rocm,xpu}` fine-tuning | train | `python /app/llm_train.py` |
| `ml-backend/`, `ml-backend-custom/` | `prediction/ml-backend*` Label Studio integration | prediction | gunicorn `_wsgi:app` on port 9090 |
| `evaluator/` | `train/evaluator` evaluation sidecar | train | `python /app/evaluate.py` |
| `optimizer/` | `train/optimizer` model conversion | train | `python /app/optimize.py` |
| `notebook-base/` | `prediction/jupyter-base` workbench | prediction | `jupyter lab` on port 8888 |

## Trainer contract

The [Trainer SDK README](sdk/README.md#trainer-contract) is the normative
public integration contract for built-in and user-provided training images. It
defines capability profiles, environment inputs, published artifacts, and final
metrics. Trainers whose build context needs the SDK use `app/builders` as the
Docker context so they can `COPY sdk` into the image.

Custom trainer code and its dependencies belong in the customer's repository
and image. The platform invokes that image through the same contract but does
not import or redistribute its implementation or weights.

## Canonical dataset layouts

Upload normalization happens once before training. Trainers and evaluators
consume the resulting layout rather than repeating archive-format detection.
Explicit `train/` and `val/` splits take precedence; flat layouts use the
relevant trainer's deterministic split behavior.

| Format | Canonical layout | Primary use |
|---|---|---|
| `yolo` | `data.yaml` plus `train|val|test/{images,labels}/`, or flat `images/` and `labels/` | Detection, boxes, and polygon-based tasks |
| `imagenet` | `class_names.json` plus `<split>/<class>/<file>`, or flat `images/<class>/<file>` | Classification and multilabel tasks |
| `llm-sft` | Root-level `*.jsonl` plus optional `images/` | LLM and VLM corpora |
| Raw media | Flat `images/` or `videos/` without a training format | Browsing and annotation only |

Detection text labels use normalized YOLO bbox lines (`cls cx cy w h`) or
polygon lines (`cls x1 y1 x2 y2 ...`). Images without label files remain
negative samples. Classification class order follows `class_names.json`.
Semantic segmentation additionally uses single-channel PNG masks, and pose uses
COCO keypoint annotations. The task-specific readers in `trainer-torch/` and
`trainer-paddle/` document their exact accepted variants.
