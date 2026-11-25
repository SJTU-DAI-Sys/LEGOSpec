# 开发迭代与恢复约定

2026-11-22 用户最新要求：只维护当前实现，除非明确要求，不考虑旧版兼容。

已删除训练恢复的 PLAN_VERSION、ROUND_ORDER_VERSION、ANCHOR_SAMPLING_VERSION、LOSS_CONTRACT_VERSION，以及 checkpoint/optimizer 格式版本和执行契约标签。
已删除配置身份、数据指纹、rank topology、参数签名、loss contract、scheduler 类型和 pending plan 签名的恢复比较。不新增旧格式适配或自动迁移。

恢复直接加载当前 rank 的状态文件、optimizer/scheduler、FP32 master、RNG、cursor 和 pending plan。checkpoint 仍先写各 rank 状态，再发布公共元数据；模型权重和训练 YAML 继续保存。
shuffle 与 anchor 的种子只依赖实际 seed、epoch、phase、window 和样本信息，不再包含版本前缀。不承诺跨代码更新复现以前的采样序列。

training_mode=legacy|optimized 仍是已明确保留的执行模式。模型结构选择、外部依赖版本要求、缓存存储编码不属于这次训练恢复清理范围。

本次只修改当前源码、测试和说明；不迁移已有 checkpoint、cache、发布权重或历史运行快照。
验证记录：tmp_exp/resume_simplification_20261122/。
