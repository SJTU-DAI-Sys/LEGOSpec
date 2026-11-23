# Training

负责训练装配、数据调度、目标特征获取、单步计算、参数更新、记录及完整恢复。
调用链为 scripts/composable/train.py → config.py → runtime/runner.py → runtime/setup.py。

## 根目录

- __init__.py：训练公共入口。
- config.py：CLI、运行参数解析和入口校验。

模型配方与 loss 数学归 modeling/draft；损失输出使用 objective/outputs.py 的契约。

## runtime/：训练生命周期

- setup.py：构造 draft、target、共享组件、FSDP、优化器和 TrainingStep，组装 session。
- runner.py：遍历 epoch、逻辑 batch、microbatch 和累积窗口，调度 forward/backward、更新与保存。
- state.py：Phase、RunState、TrainingSession 等阶段、游标和运行状态。
- checkpoint.py：权重导出、完整训练状态保存/恢复、每 rank RNG 和恢复元数据。

完整恢复包含 optimizer、scheduler、FP32 master、每 rank RNG、游标和 pending plan。
weights-only 用于推理或 init_from；恢复直接读写实际状态，不比较代码或配置版本。

## data/：训练输入和工作量规划

- loader.py：构造原始对话或隐藏态缓存 loader，选择 Dataset、collator 和分布式 sampler。
- batch.py：anchor 采样、标签及 mask 对齐、逻辑 batch、打包/解包和 TP batch 选择。
- layout.py：样本计划、packed batch 布局和索引契约。
- planning.py：训练成本估计、均衡 microstep 分配、CPU token 窗口及 target 工作量规划。
- prefetch.py：保持顺序的 CPU batch 预取和释放。

本目录组织训练批次；通用对话预处理和磁盘缓存格式归 legospec/data。

## target/：冻结监督的训练侧装配

- setup.py：确定 capture 层和容量，选择在线/缓存特征，装配 tokenizer、mask token 和共享 embedding/head。
- features.py：按目标层选择、拼接、设备搬运及取得明确的 teacher 特征。
- streaming.py：分离部署中的 target producer、packet 组织、广播和训练包调度。

模型捕获、缩放、final norm 和 backend 差异归 modeling/target；进程间传输实现归 distributed。

## execution/：单步计算和窗口分母

- step.py：TrainingStep 和调用参数组织，在一次 FSDP forward 中执行模型、Proposal 与 Objective。
- supervision.py：对齐目标监督行与 block，按需取得词表投影所需特征。
- windows.py：统计完整 optimizer 窗口的 loss 分母，准备和释放临时 batch。

Execution 消费模型和 Objective 接口，处理 packed/chunked 执行及归一化；不拥有优化器调度。

## optim/：优化器和学习率

- optimizer.py：参数分类、BF16/FP32 master、AdamW/Muon 组合、梯度归一化及更新。
- muon.py：Muon 数学、矩阵交换和分片更新。
- fsdp_layout.py：读取 FSDP 参数布局，生成分片与矩阵路由信息。
- scheduler.py：warmup、cosine、延迟和分阶段学习率调度。

参数布局读取集中在 fsdp_layout；优化数学不散入模型 forward 或 runner。

## logger/：指标与外部记录

- metrics.py：token accuracy、硬标签接受长度、分布接受率及 expected acceptance 等加性指标。
- logging.py：指标累积/归约、日志 payload、显存记录及 W&B 等 tracker 接入。

Logger 只消费结果，不改变训练计算或样本游标。

低层功能目录不反向调用 runtime/setup 或 runner；OPD 训练已移至 development/opd。
