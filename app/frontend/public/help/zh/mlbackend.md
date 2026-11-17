<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# ML Backend（预标注服务）

## 用途
ML Backend 是 Label Studio 的模型服务，用于**半自动标注**：对未标注任务自动生成预标注框，标注员在此基础上人工修正。

## 流程
1. 注册 ML Backend：记录名称、URL、关联模型（可选）。
2. 在标注工作台“半自动标注”按钮旁的下拉中选择 ML Backend（或默认服务）。
3. 预标注作为 predictions 写入 Label Studio，在标注工作台可见。

## 前提条件
- ML Backend 服务需可达（集群内或已对外暴露）。
- 关联模型时，预标注结果必须与该模型的类别空间一致。

## 常见问题
- **预标注标签对不上？** 模型类别与标注配置需保持一致。
- **服务不健康？** 健康检查是真实的 HTTP 探测 `/health`。

## 下一步
注册模型 → 标注工作台选择服务 → 半自动标注 → 人工审核 → 导出用于训练。
