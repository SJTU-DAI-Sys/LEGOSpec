# Draft

模型执行 Context → Backbone → Proposal；Objective 独立计算损失。
四轴直接位于本目录，公开入口为 LegoSpecModel、LegoSpecConfig。

Attention 配方统一使用 family/variant/access：dense 包含 MHA/GQA/MQA，
latent 为 MLA，linear 为 GDN/KDA。Dense/latent 支持 full/SWA/DSA 的已校验组合；
GDN/KDA 固定包含共享投影的逐头 gated block attention。
共享配置不共享权重，组合边界由配置入口检查。

- [组件边界](../DESIGN.md)
- [项目代码组织](../../../docs/COMPONENT_ORGANIZATION.md)
- [配方](../../../configs/README.md)

配置描述组件；backend、数据、batch、优化器和调度归运行 YAML/CLI。
不提供旧字符串配方、旧导入路径或旧 checkpoint 参数映射。
只有确认 context 提交到缓存，全部层执行成功后才发布新状态。
递归历史回退需要完整状态快照或重算。

export_composable 生成独立推理包，冷加载无需训练包。
当前回归及数值对照见 tmp_exp/draft_boundaries_20261123。
