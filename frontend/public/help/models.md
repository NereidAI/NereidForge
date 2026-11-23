<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# 模型库 / Models

## 流程 / Flow
1. 模型注册 / Register：训练完成自动注册，也可手动上传单文件、多文件制品或 .zip/.tar.gz/.tgz 压缩包。 / Training jobs register automatically; manual upload accepts single files, related file groups, and safe archives.
2. 版本管理：每个模型多版本，优化版本带目标设备与加速比。
3. 对比选型：版本表显示目标设备/加速比，选择最优版本。
4. 部署：选择版本 → 配置运行时（OpenVINO/Triton）+ GPU + 域名 → 自动创建 KServe 服务 + Ingress。

## 前提条件 / Prerequisites
- 手动上传支持格式目录中列出的模型文件；OpenVINO IR 必须包含同名 .xml/.bin，歧义压缩包需填写主制品路径。 / Use a format from the artifact catalog; IR needs its companion and ambiguous archives need an explicit primary path.
- 部署域名需在 DNS 解析（默认生成 `<模型>-<版本>.inference.<平台域名>`）。

## 制品包与能力 / Artifact bundles and capabilities
- 多文件上传会保留全部文件；压缩包通过路径、链接、重复项、文件数和解压大小校验后才登记，不会直接执行模型代码。 / Multi-file sets are preserved; archives are validated and extracted before registration, without executing model code.
- 版本展开列表支持嵌套路径，可下载具体文件；试跑也使用列表中的精确相对路径。 / Expanded versions expose nested files for exact-path download and trial selection.
- 能力目录明确区分登记、下载、试跑、优化和部署。TensorRT engine 与 Ascend OM 当前仅支持登记和下载；存储成功不代表可部署。 / Capabilities distinguish registration, download, trial, optimization, and deployment. TensorRT engines and Ascend OM are registration/download only; storage success does not imply deployability.
- 编译制品可填写运行时约束 JSON（工具链、驱动、计算能力等），这些字段只是约束记录，不是兼容性验证。 / Optional runtime-requirement JSON records toolchain/device constraints; it is not a compatibility proof.

## 图片试跑 / Image trials
- 在具体版本的制品列表中选择 **用图片试跑 / Test with image**，可选取数据集图片或上传本地图片。
- **自动匹配 / Auto** 只使用已验证的内置 Decoder；也可手动选择内置 Decoder。无法识别时选择 **自定义映射 / Custom mapping**，根据模型文档配置输入预处理、输出 tensor 和字段语义。
- 检测画框，实例分割绘制实例轮廓，语义分割叠加类别掩膜，分类显示 Top-K。未识别且未配置时不会猜测坐标或返回空白检测。
- 成功试跑后的 Decoder 选择按具体制品保存，下次打开自动复用；同一版本的其他制品互不影响。
- 上传时的 **Model Source** 只是模型家族提示，**Input Shape** 只是预处理/导出尺寸提示，两者都不能证明输出格式。

## 删除注册记录 / Delete registry entries
- 只有 `default` 租户中启用且具备 `admin` 角色的本地内置 `admin` 可以删除整个模型或单个版本；租户管理员、operator 和 SSO 同名账号均无此权限。 / Only the enabled local built-in `admin` in the `default` tenant with the current `admin` role can delete a model or one version; tenant admins, operators, and federated namesake accounts cannot.
- 删除整个模型会同时删除其所有注册版本。删除会归档 Hub 记录并移除 NF 记录，但保留 S3 制品字节。 / Deleting a model removes all of its registry versions. Hub records are archived and NF records are removed, while S3 artifact bytes are retained.
- 已部署、部署中、卸载中或卸载失败且服务仍存在的版本必须先成功卸载；后端返回 409 时不会发生部分删除。 / Deployed, deploying, terminating, or failed-undeploy versions must be undeployed successfully first; a 409 response means no partial deletion occurred.

## 常见问题 / FAQ
- **部署后怎么访问？** 部署响应返回域名；Ingress 自动创建，DNS 指向集群 IP。
- **运行时选哪个？** OpenVINO（CPU，已验证）或 Triton（GPU 场景预留）。
- **为什么提示 decoder mapping required？** 当前图没有可靠匹配的内置 Decoder。请手动选择已验证 Decoder，或根据实际 output tensors 配置映射。

## 下一步 / Next Steps
部署后可用推理端点；或继续优化/评估其他版本。
