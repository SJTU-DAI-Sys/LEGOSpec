# Tests

本目录保存可重复运行的正确性与回归测试。

- unit/：模型数学、梯度、状态、缓存、数据链路、训练、分布式归约和导出测试。
- reference/：测试使用的固定数值参考、最小源码 oracle 及来源记录；不属于运行时依赖。
- conftest.py：共享测试设置，临时文件和默认 TorchInductor/Triton 编译缓存统一写入项目根目录 tmp_exp/artifacts/。

运行 CPU 测试：
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m pytest tests/unit -q -rs

运行 CUDA 测试：
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m pytest tests/unit -q -rs

选择空闲 GPU。缺少 CUDA 等依赖时，对应测试会显示明确的 skip。
后续独立的完整链路测试可放 integration/，小型合成输入可放 fixtures/，实际需要时再建立目录。

实验、排查脚本、profiler trace、性能结果和历史源码/数值快照放 tmp_exp/，
按任务或实验分目录；不纳入 Git 或发布包。公开可复用的 benchmark 入口可放 benchmarks/，
其输出仍写 tmp_exp/。待开发实现及关联测试快照放 development/。
模型权重、数据和大型共享缓存保持现有独立存储位置。
