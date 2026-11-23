# 已确认版本选择

- B2：只保留 `tar-raw-v1`，删除其他 cache reader/writer；不删除物理历史 cache。
- B3：固定异步采集。
- B6：固定有界 pinned buffer 预分配与复用。
- B7：实际输出为 packed 时自动 bulk transfer；普通输出使用 tensor 布局适配。
- B8：随 safetensors cache 退役，删除全部中间 writer。
- B9：固定流式 SHA256 与异步提交，完成提交后发布 manifest。
- DLite 两阶段与 BV objective 删除。
- 整体 `training_mode=legacy|optimized` 保留。GDN 后端/卷积语义、block attention 共享投影、Context 融合、Proposal 数学、Objective 数学等未被选择删除的分支继续保留。

最终优化路径是执行收敛，不意味着所有模型/硬件上已重新测得最高速度。容量参数仍可配置。

2026-11-22 补充确认：anchor 仅 sample_epoch；恢复版本标记与 loss contract 比较已按最新要求删除；DSA 生产固定 gather，reference 仅在 tests/unit。详见 VERSION_CLEANUP.md。
