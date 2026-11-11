<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# ML Backend（预标注服务）/ ML Backend

## 用途 / Purpose
ML Backend 是 Label Studio 的模型服务，用于**半自动标注**：对未标注任务自动生成预标注框，标注员在此基础上人工修正。

## 流程 / Flow
1. 注册 ML Backend：记录名称、URL、关联模型（可选）。
2. 标注工作台"半自动标注"按钮旁下拉选择 ML Backend（或默认）。
3. 预标注写入 Label Studio 作为 predictions，标注工作台可见。

## 前提条件 / Prerequisites
- ML Backend 服务需可达（集群内或已暴露）。
- 关联模型时，预标注结果与该模型的类别空间一致。

## 常见问题 / FAQ
- **预标注标签对不上？** 模型类别与标注配置需一致。
- **服务不健康？** 健康检查为真实 HTTP 探测 `/health`。

## 下一步 / Next Steps
注册模型 → 标注工作台选择 → 半自动标注 → 人工审核 → 导出训练。
