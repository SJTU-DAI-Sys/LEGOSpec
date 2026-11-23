from pathlib import Path
R=Path(__file__).resolve().parents[2]
T=R/'legospec/training'
(T/'DESIGN.md').write_text('''# training

根目录只保留 `__init__.py` 入口、`config.py` 参数边界和 `outputs.py` 加性输出契约。

- `runtime/`：setup 组装 session 与 draft/FSDP；runner 管迭代、backward/step 时机和释放；state 管阶段与游标；checkpoint 管导出、完整保存/恢复及身份校验。
- `data/`：loader 管 JSON/hidden-cache loader 与 collator；batch 管 anchor/标签/mask/打包；layout 管索引契约；planning 管样本工作量分配及 CPU token 窗口；prefetch 管有序 CPU 预取。
- `target/`：features 管冻结 target、共享 embedding/head 和监督特征；streaming 管分离部署的生产循环及训练包。底层 target 实现仍在 modeling/target，传输实现在 distributed。
- `execution/`：step 管 TrainingStep 与调用参数组织；sampled 管当前策略采样、验证和回放；supervision 管词表分块的监督行；windows 管完整 optimizer 窗口分母与 batch 生命周期。
- `optim/`：optimizer 管 FP32 master、AdamW/Muon 组合、梯度裁剪/归一化；scheduler 管学习率；muon 管独立数学；fsdp_layout 隔离 FSDP 私有布局接口。
- `logger/`：metrics 管 accuracy、overlap、接受长度；logging 管聚合、日志与 tracker。

入口：scripts/composable/train.py → training/config.py → runtime/runner.py → runtime/setup.py。
训练循环调用 execution；冻结监督来自 target；输入批次来自 data；更新使用 optim；记录使用 logger。
低层功能目录不得反向导入 runtime/setup.py 或 runtime/runner.py。checkpoint 身份构造归 checkpoint；不反调 setup。

迁移直接更新调用方，不保留旧路径转发文件。内部目录使用命名空间包，不创建空 __init__.py。
Objective 数学仍归 modeling/draft/composable/objective。已停用的 component-normalized OPD 分支与窗口准备保存在 development/objectives/training_snapshots_20261122，不进入活动训练路径；E2E-TV 共享采样保留。

不改变模型参数路径、损失归一化或优化器恢复格式。完整恢复保留 optimizer、scheduler、FP32 master、每 rank RNG、训练游标和规划窗口。
回归记录位于 tests/training_layout_20261122/。
''')
dirs={
'runtime':'训练生命周期：setup 只构建/组装；runner 只调度；state 定义阶段/游标/session；checkpoint 独立拥有保存、恢复和训练身份。低层模块不反向调用 setup/runner。',
'data':'训练输入：loader 构建 loader/collator；batch 采样 anchor 并构建标签、mask、packed tensor；layout 定义布局索引；planning 规划工作量与 CPU token 窗口；prefetch 有界预取。预取不提交训练游标，不执行模型前向。',
'target':'冻结监督：features 构建目标和共享 embedding/head、解析缓存或在线 hidden；streaming 组织 producer 和训练包。依赖 modeling/target 与 distributed 的现有实现，不复制后端或传输协议。',
'execution':'训练计算：step 包含单次 FSDP 前向及调用数据组织；sampled 保存采样 token/uniforms 并重放；supervision 对齐监督行；windows 统计全累积分母并管理临时 batch。不得反调 runtime/setup 或 runner，objective 公式仍归模型 objective。',
'optim':'参数更新：optimizer 管 FP32 master 与 AdamW/Muon 路由、归一化和裁剪；scheduler 管学习率；muon 管正交化与通信；fsdp_layout 管私有 FSDP 布局读取。保存格式与参数顺序不因目录调整改变。',
'logger':'观测：metrics 计算加性 accuracy/接受长度；logging 累计并归约分子分母，组织日志并调用 tracker。指标记录不能更改训练计算或样本游标。'}
for name,text in dirs.items():(T/name/'DESIGN.md').write_text('# '+name+'\n\n'+text+'\n')
p=R/'docs/COMPONENT_ORGANIZATION.md';s=p.read_text();a=s.index('训练入口：');b=s.index('没有独立',a)
s=s[:a]+'''训练入口：`scripts/composable/train.py → training/config.py → runtime/runner.py → runtime/setup.py`。
`training/` 按功能分为 `runtime/`、`data/`、`target/`、`execution/`、`optim/`、`logger/`；根目录仅保留入口、配置与输出契约。
setup 只组装 session；数据加载、target 特征/生产循环、单步执行与窗口准备分别归所属目录。checkpoint 自己构造恢复身份，不反调 setup；低层功能目录不导入 runtime/setup 或 runner。
具体文件职责见 `legospec/training/DESIGN.md`。
'''+s[b:];p.write_text(s)
p=R/'docs/SIMPLIFICATION_PLAN.md';s=p.read_text()+'\n2026-11-22 训练目录按 runtime/data/target/execution/optim/logger 重组；拆分 setup 和 scheduler，合并 block 张量准备及指标计算，清理暂存 OPD loss 的专属训练残留。不保留旧路径转发层。\n';p.write_text(s)
p=R/'development/objectives/README.md';p.write_text(p.read_text()+'\n2026-11-22 训练目录重组：其余 OPD component-normalization、metadata window/replay 集成在移出前完整保存在 `training_snapshots_20261122/`。该快照使用重组前的导入路径；恢复时按新功能归属合并。\n')
print('Updated training ownership and module design documents.')
