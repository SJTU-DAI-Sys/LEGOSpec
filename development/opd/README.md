# OPD：暂不支持

2026-11-23：将当前策略 sampled-anchor OPD 从正式实现移到此开发区。
此目录不属于 legospec 安装包，没有注册入口，也不提供旧路径转发。

- target/anchor_opd.py：HF Qwen3 scorer，文档前缀及分支隐藏态计算。
- target/sglang_anchor_opd.py：SGLang scorer，复用 paged KV 前缀执行分支。
- sampled.py：当前策略采样、p/q 验证、SampledOPDBatch 回放及损失计算。
- integration_snapshots/：移出前的 CLI、TrainingStep、target setup、窗口、
  checkpoint、logger 和 backend 接口，供恢复开发时参考。
- configs/anchor_comparison_sft_rnn.yaml：OPD 对比实验的 SFT 对照 launcher。
- 测试资料位于仓库 tests/development/opd/，不进入默认回归。

这些是开发资料，不是可独立运行的插件。快照中的旧导入、接口和配置按原样保留；
恢复开发时按当时的组件边界整合，不能整文件覆盖主代码。

此前已暂缓的 opd_survival 公式、配方和 launcher 仍在
development/objectives/，与其他待开发 objective 一起保存。

正式路径移除了两个 scorer 工厂、OPD CLI 参数、采样/回放分支、target 容量特例、
窗口跳过逻辑、checkpoint OPD 元数据和 rollout 指标要求。
普通 e2e_tv 损失、固定数据监督、anchor 位置抽样、因果 block attention 和
推理的 p/q 接受/拒绝继续保留；这些不是 OPD 专属实现。
