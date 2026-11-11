<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# 模型训练 / Training

## 目标、架构与权重 / Target, architecture, and weights
- **目标模型（输出）/ Target model (output)**：训练产物归属的模型；完成后会在该模型下注册新版本。它不决定网络结构或初始权重。留空时根据架构或 checkpoint 推导名称。 / The registry model that receives the new trained version. It selects neither the network nor its initialization.
- **网络架构 / Architecture**：要构建的网络结构，例如 ResNet-50 或 Faster R-CNN。架构与权重是两个独立维度。 / The network to build, independently of the weights loaded into it.
- **起始权重 / Starting weights**：选择“已有 checkpoint”后使用的原生训练检查点，可来自基础模型池或已注册版本。仅含 ONNX 的版本不能用于训练初始化。 / The native checkpoint used in checkpoint mode; ONNX-only versions cannot initialize training.

## 初始化方式 / Initialization
- **预训练权重 / Pretrained weights**：使用训练器为所选架构提供的默认预训练初始化，保持既有默认行为。 / Use the trainer's default pretrained initialization for the selected architecture.
- **从零训练 / From scratch**：不加载预训练权重或 checkpoint，随机初始化模型。 / Build the selected architecture without pretrained weights or a checkpoint.
- **已有 checkpoint / Existing checkpoint**：从基础模型池或已注册版本继续训练。架构可留空，由注册版本元数据、可识别的 checkpoint 名称或任务默认架构解析；需要覆盖时再显式选择。 / Continue from a native checkpoint. Leave architecture empty to resolve it from version metadata, a recognized checkpoint name, or the task default; select it explicitly only to override that resolution.

## LLM/VLM 基座模型 / LLM/VLM base models
- LLM/VLM 的“基础模型（模型池）”同时确定模型结构和参数，因此不显示独立的架构与初始化选择。 / An LLM/VLM base model includes both structure and parameters, so the separate vision architecture and initialization controls do not apply.

## 流程 / Flow
1. 选择任务类型、数据集及标注版本。 / Select the task, dataset, and annotated version.
2. 可选选择目标模型，再确认架构、初始化方式和超参数。 / Optionally select a target model, then confirm architecture, initialization, and hyperparameters.
3. 启动后运行 KFP 训练流水线并记录进度。 / Start the KFP training pipeline and track its progress.
4. 完成后自动注册模型版本并生成训练曲线。 / On completion, register the trained version and its training curves.

## 常见问题 / FAQ
- **曲线在哪看？/ Where are the curves?** 任务详情页的“训练曲线”标签。 / Open the Training curves tab on the job detail page.
- **如何续训？/ How do I continue training?** 初始化方式选择“已有 checkpoint”，再选择含原生训练权重的已注册版本或模型池条目。 / Select Existing checkpoint, then choose a registered version or pool entry containing native training weights.
- **中断了怎么办？/ What happens after failure?** 失败任务可重新启动；需要沿用之前参数时，从原任务选择重新训练并检查起始权重。 / Restart a failed job, or clone it with Retrain and verify the starting weights before submitting again.
