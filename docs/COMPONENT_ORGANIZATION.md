# LEGOSpec 代码组织

四个可组合轴共享训练、数据、目标模型、分布式与推理设施。公开类为 `LegoSpecModel`、`LegoSpecConfig`，Python 包为 `legospec`。

- Context：`modeling/draft/context/fusion.py`。`concat_projection` 控制投影及可选归一化；`gated_fusion` 控制逐层融合。只负责目标特征融合。
- Backbone：`backbone/model.py` 中唯一的 `Backbone` 和 `Layer`。Dense（MHA/GQA/MQA）、latent（MLA）、linear（GDN/KDA）数学归 `backbone/attention/`；卷积归 `backbone/convolution.py`。Dense/latent 的历史访问独立选择 full/swa/dsa，GDN/KDA 内置共享投影与逐头 gate block attention。实现分别归入 attention/dense.py、latent.py、linear/；access/ 管历史访问，backends/ 管执行后端。详见 ATTENTION_CONFIG.md 和 legospec/modeling/DESIGN.md。
- Proposal：`proposal/parallel.py`、`markov.py`、`selector.py`、`sequential.py`。共享主干输出，保留 additive RNN、serial replacement、候选集与全词表残差的数学区别。
- Objective：`objective/` 下的 token、block、composition、DPAL，以及词表投影缓存与损失内核。独立输出分子/分母，统一训练器负责归约。

训练入口：`scripts/composable/train.py → training/config.py → runtime/runner.py → runtime/setup.py`。
`training/` 按功能分为 `runtime/`、`data/`、`target/`、`execution/`、`optim/`、`logger/`；根目录仅保留入口、配置与输出契约。
setup 只组装 session；数据加载、target 特征/生产循环、单步执行与窗口准备分别归所属目录。checkpoint 直接读写恢复状态，不反调 setup；低层功能目录不导入 runtime/setup 或 runner。
具体文件职责见 `legospec/training/DESIGN.md`。
没有独立 DLite/DFlash/DFlash2/DSpark 模型、训练 adapter、recipes 继承链或算法注册器；算法名称只可作为实验配方名。

推理：`inference/generate.py` 共用生成循环，选择 block 或 hidden-feedback 解码状态；`inference/verify.py` 共用 p/q 接受与正 p-q 残差。采样位于 `proposal/sampling.py`。
隐藏态反馈由 `backbone.schedule=stepwise`、`input_fusion=concat_hidden` 选择；仍经过 Context、同一个 Layer/GQA 与 Proposal。当前只允许一层 dense GQA、无卷积、parallel 投影。组合限制由 schema 入口检查，不表示所有组合都已经支持。

缓存：只读写 `tar-raw-v1`。采集入口 `scripts/data/capture_hidden_states.py`；固定异步 pipeline、pinned buffer 预分配/复用、packed 输出 bulk D2H、流式 SHA256、异步 durable commit。非 packed 输出按 tensor 适配。保留内存、token、worker、队列、shard 容量参数。

整体 `training_mode: legacy | optimized` 保留；其余未经选择的数学/后端选项继续保留。DLite 两阶段与 BV loss 已退役。
新模型保存为 `model_type=legospec`、`architectures=[LegoSpecModel]`；旧完整算法 checkpoint 不通过隐式兼容层加载。
完整 resume 必须使用原训练配置与 rank topology，并恢复 optimizer、scheduler、FP32 master、每 rank RNG 和 cursor；weights-only 仅用于推理或 init_from。

修改只针对公共迭代目录，训练中的源码快照、cache、checkpoint、model_publish 权重与原生 config 不参与迁移。

Linear 的构建、位置编码、共享 block 融合和 O 投影集中于 attention/linear/recurrence.py；Layer 保留通用 norm/残差/MLP。当前只保留固定共享投影分支，参数注册在 Layer.attention 下。

暂缓使用的实验目标及其恢复快照保存在 development/objectives/，不属于 legospec 安装包。

当前开发不维护旧版兼容。恢复直接加载完整状态，不保存或比较版本号、配置身份、规划签名。

Target 的 backend 与模型 adapter 分离：HF/SGLang 管执行与资源；默认/Gemma/Qwen3.5 adapter 管模型语义与 capture 差异。训练 target/setup 只提出训练需求，features 只组装特征，streaming 只调度生产/传输。最后一层 residual 和 normalized teacher 输出分开，详见 legospec/modeling/DESIGN.md。

2026-11-23：native packed capture 与 TP runtime 收入 target/sglang_backend；在线和离线共用 target factory 与 LEGOSpec 分布式初始化。Gemma 的 final norm 重建也归 adapter。其余边界检查和后续结构计划见 TARGET_CAPTURE_PLAN.md。

2026-11-23：OPD 的剩余 scorer、sampled-anchor 训练/回放、CLI 和元数据移入 development/opd；正式代码当前不支持 OPD，普通 E2E-TV 和正常 anchor 抽样保留。回归与扫描见 tmp_exp/defer_opd_20261123/。

2026-11-23：移除 draft/composable 目录层级。attention 统一为 Layer.attention 子模块，状态归 backbone/state，stepwise 归 backbone，Objective 输出和权重数学不再依赖 training；组件管理特殊初值。详见 legospec/modeling/DESIGN.md。
