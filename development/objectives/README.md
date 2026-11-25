# 待开发的训练目标

2026-11-22：暂存 `dpal_plus_plus`、`tv_survival`、`opd_survival`。
这些目标未注册到当前 schema / loss registry，现有训练入口拒绝选择它们。
此目录在 `legospec` 包外，不随 Python 包安装。

- D-PAL++：在 D-PAL 动态梯度 credit 上除去当前位置自身的平滑 overlap；rho 控制平滑，local_loss 选择局部损失，reduction 选择归一化。
- TV-survival：逐位置 TV 加连续前缀存活辅助项，survival_weight 控制后者。
- OPD-survival：接受位置 forward KL、拒绝位置 reverse KL，加前缀存活项；beta_rej、rejection_decay 控制拒绝项，survival_weight 控制存活项。离线接受长度由固定标签上的贪心匹配给出；采样模式使用 p/q 验证。

## 保留内容

- `snapshots/legospec/modeling/draft/composable/objective/survival.py`：TV/OPD-survival 原始实现。
- `snapshots/legospec/modeling/draft/composable/objective/dpal.py`：含 D-PAL++ 分支的原始实现，同时含共享的普通 D-PAL。
- `snapshots/` 其他文件：移出前的 schema、注册、配置归一化及训练集成参考。
- `configs/`、`scripts/config/`：原始 JSON 配方和 launcher。路径、权重引用按原样保留，其中旧 TV launcher 引用的 JSON 在移出前已不存在。这些是开发资料，不是当前可运行入口。
- 对应测试快照位于仓库 `development/objectives/tests/`。

## 恢复开发

快照不是独立可运行插件。恢复时应将相关公式与注册、schema、训练分支合并到届时的代码，并恢复专属测试；不要整文件覆盖持续演进的共享模块。
普通 `dpal_renyi`、`dpace`、`e2e_tv` 仍在主代码中。普通固定数据监督的 E2E-TV 路径保留。
历史 checkpoint、模型发布目录、训练源码快照未迁移。旧目标的完整 resume 仍应使用原始训练环境、配置及 rank topology。

2026-11-22 训练目录重组：其余 OPD component-normalization、metadata window/replay 集成在移出前完整保存在 `training_snapshots_20261122/`。该快照使用重组前的导入路径；恢复时按新功能归属合并。

2026-11-23：sampled-anchor OPD 的剩余 scorer、采样/回放与训练集成移至 ../opd/；当前正式训练不支持 OPD。此前 objective 快照继续保留在这里。
