<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# 模型优化 / Optimization

## 流程 / Flow
1. 创建优化任务：选源模型（版本）+ 一个或多个目标设备。
2. 多目标：可选择多个设备（CPU/OpenVINO/NVIDIA 等），每个目标生成独立任务。
3. 平台自动做 ONNX 图优化 + int8 量化，产物落 seaweedfs 并注册为新模型版本（带目标后缀）。
4. 优化版本带加速比/体积缩减等真实指标，可在模型库对比选型。

## 实时进度 / Live progress
- 运行中的任务显示基线评测、模型优化、质量门禁验证和收尾四个阶段。 / Running jobs show four milestones: baseline evaluation, model optimization, quality-gate validation, and finalization.
- 百分比表示已完成的流水线里程碑，不是剩余时间估算；只有整个 KFP 运行成功后才显示 100%。 / The percentage counts completed workflow milestones, not estimated time remaining; it reaches 100% only after the whole KFP run succeeds.
- 详情页实时显示执行器日志。`[opt] SUCCESS` 表示优化器已完成，后续质量门禁仍可能正在运行。 / The detail page shows polling-based executor logs. `[opt] SUCCESS` means the optimizer finished; quality-gate validation may still be running.

## 前提条件 / Prerequisites
- 源模型需有 ONNX 产物（训练或手动上传的 .onnx）。
- NVIDIA/TensorRT 等目标需要对应硬件/工具链（集群无硬件时任务标记 pending）。

## 常见问题 / FAQ
- **加速比从哪来？** optimize.py 真实 benchmark（ONNX Runtime），非估算。
- **优化后怎么用？** 优化版本注册在源模型下，可部署/评估。

## 下一步 / Next Steps
优化后到**评估**页验证优化版精度（accuracy_drop），或到**模型**页部署。
