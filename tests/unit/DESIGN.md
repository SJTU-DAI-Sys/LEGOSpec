# 单元与集成回归

覆盖四轴前向/梯度、GDN状态提交、稀疏/packed执行、分布式归约、hidden-feedback与block推理、独立导出、raw-tar恢复、optimizer完整resume。CUDA测试由可用设备决定，具体跳过原因用 pytest -rs 查看。
