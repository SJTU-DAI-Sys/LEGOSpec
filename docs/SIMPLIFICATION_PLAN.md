# 简化计划实施状态

2026-11-22：原计划已落为四轴和共享设施，实际目录以 COMPONENT_ORGANIZATION.md 与 data/modeling/training 三份 DESIGN.md 为准。

已实施：组件按 owner 归位；删除完整算法和注册器；合并 Context、Objective 短文件、access、Layer/Backbone、batch/windows；移除 adapter/recipes 和 core 中转目录；公开包与类命名统一；固定 raw-tar 最终采集链路；删除 DLite 两阶段、BV；保留整体训练模式开关。

数值、梯度、状态、分布式、导出、缓存恢复和完整 optimizer resume 由 tests/unit 覆盖。测试结果以本轮最终执行记录为准。历史计划、历史实验和源码 oracle 归档在公共树之外。

不要重新引入每类一个文件的规则。只有独立数学、实际复用、可选依赖或资源生命周期才构成拆文件理由；单次处理就近写。共享组件不得反向依赖具体算法。

2026-11-22 训练目录按 runtime/data/target/execution/optim/logger 重组；拆分 setup 和 scheduler，合并 block 张量准备及指标计算，清理暂存 OPD loss 的专属训练残留。不保留旧路径转发层。

2026-11-22 按用户要求删除训练恢复版本标签与身份比较，只保留当前实现的直接状态保存/恢复；不提供旧格式兼容。

2026-11-22 target 边界重组：拆出 training/target/setup；HF/SGLang 复用模型 adapter，移除 Gemma 独立 training backend；统一层字典与显式 final-norm teacher，移除训练端模型内部访问和重复 cached logits 路径。

2026-11-23 capture 边界收尾：删除 Gemma 专属 capture 文件/数据层构造分支；native capture 和并行运行时归 backend，模型 final norm 重建归 adapter。其余尚未实施的边界建议详见 TARGET_CAPTURE_PLAN.md。

2026-11-23：OPD 的剩余 scorer、sampled-anchor 训练/回放、CLI 和元数据移入 development/opd；正式代码当前不支持 OPD，普通 E2E-TV 和正常 anchor 抽样保留。回归与扫描见 tests/defer_opd_20261123/。

2026-11-23：draft 组件边界重构已实施，四轴直属 draft；移除旧 attention 配方与独立 block 分支，统一 attention 组件，局部初始化归组件，状态操作归 backbone，Objective 不依赖 training。结构见 legospec/modeling/DESIGN.md。
