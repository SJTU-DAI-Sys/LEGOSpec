代码风格要求：
防过度防御（不要杞人忧天）

要求：对于几乎不可能发生的边界情况或并发问题，不能擅自写一堆防御性代码。必须先把发生概率讲清楚，经用户确认后才能加。
不合规定的输入错误允许他报错。
禁令：严禁为了追求“理论上更稳”而擅自增加膨胀代码。

入口单次校验（信任内部类型系统）

要求：外部传入的数据，只在系统边界（入口）做一次校验。到了系统内部，就完全依靠静态类型系统来保证流转。

禁令：禁止在内部的每一层函数里都重复写校验逻辑（层层重复判断）。

拒绝提前抽象（不要为了封装而封装）

要求：只被使用一次的代码逻辑，就直接就近写在那里（内联）。

禁令：不要随意提前把单次逻辑拆分出去，做成毫无意义的 helper 函数或工具类。

控制复杂度（保持简单直接）

要求：能用单个函数清晰明了搞定的逻辑，就写在一个函数里。

禁令：非必要不要滥用 try/catch（异常捕获），也不要硬把简单的逻辑拆碎。

Keep reusable correctness and regression tests in the repository's `tests/`
directory. Keep test fixtures small and reproducible; generated test output goes
under `tmp_exp/artifacts/`.

Store experiments, debugging scripts, performance benchmarks/profiler traces,
logs, evaluation results, and historical source/numerical snapshots in the
repository-root `tmp_exp/` directory, organized by task or experiment.
`tmp_exp/` is local development material and must stay excluded from Git and
release packages. Reusable public benchmark entrypoints may live in
`benchmarks/`; their generated results still belong in `tmp_exp/`.

Keep deferred implementation code and its associated test snapshots in
`development/`, outside the installed package. Keep model weights, datasets,
and large shared caches in their existing dedicated storage locations; record
their paths in experiments rather than copying them into Git.

For new training runs, save weights every 15,000 optimizer steps and save the
final weights. Set `save_interval: 15000` and, where supported,
`save_strategy: interval` explicitly when preparing a launcher YAML.
Every intermediate checkpoint must support full training resume. Set
`save_training_state: true` on launchers that expose the option. Preserve
optimizer and scheduler state, FP32 master parameters, per-rank RNG, and the
training cursor. Use `resume_from` with the original training configuration and
rank topology. A weights-only export is suitable for inference or
`init_from`, not for continuing the same training run.

Choose training and evaluation hosts from the current task's instructions and
the location of its model, data, and checkpoints. Prefer using existing large
caches in place. Host addresses from one experiment are not global defaults.
Use separate training and evaluation hosts when specified for that task.
An evaluation can start once its checkpoint is verified. Training may proceed
while evaluations are queued or running when resources permit.

For new W&B projects, give the project a descriptive name that identifies the
target model and experiment series. Use detailed W&B run display names in the
form `context__backbone__proposal__objective__dataT0__s42`, with the actual data
temperature and seed. Include relevant layer counts, attention patterns,
convolution and proposal sizes, and loss weights or other objective parameters.
For example,
`dflash__gdn4_gqa1_dense_conv2g16__rnn_r256__ce0p1_l10p9_g4__dataT0__s42`.
Set the display name with `wandb_name` and keep `wandb_run_id` stable for upload
and resume.

Before starting a task, explicitly inspect `.agents/skills/*/SKILL.md` in this
repository. The `.agents` directory is hidden; do not rely on default search
results that exclude hidden files. If a task matches a repository skill's
description, read its entire `SKILL.md` and any required references before
proceeding. Follow its confirmation steps and use autonomous fallbacks only
where explicitly allowed.

For downloading, inspecting, extracting prompts from, or converting a new
dataset, use `.agents/skills/prepare-spec-decoding-dataset/SKILL.md`. Before
creating an adapter or running the full conversion, inspect five records.
When interaction is available, confirm both of the following with the user:

1. Which field or path to use for prompt extraction.
2. Whether to use `multi` to retain all user turns or `first` to retain only the
   first user turn.

When interaction is available, wait for explicit answers to both questions
before creating an adapter or running the full conversion. If interaction is
unavailable, choose the highest-confidence non-empty prompt candidate from the
inspected records and default to `multi`, unless the user has already specified
these choices. Report the selected field or path and turn mode, then proceed
with adapter creation, sample validation, and full conversion. Stop if no
credible prompt candidate exists.


## 模块边界与文件粒度

修改组件前阅读 `docs/COMPONENT_ORGANIZATION.md`、`docs/SIMPLIFICATION_PLAN.md`
和 legospec/data、modeling、training 中对应模块的 `DESIGN.md`。文档中的待实施接口不是现有 API；文件粒度以简化计划为准。
保持 Context、Backbone、Proposal、Objective 的职责边界，算法名称只作为配方，不新增完整算法维护路径。
组件可以同文件不同类/函数；不要求每个类、head、状态、注册项各建文件。
单次使用的处理就近实现；只有真实复用、独立复杂数学、可选依赖或资源生命周期才拆共享模块。
新增文件或 adapter 时说明它承担的实际职责与消费者；不为对齐目录树建立空 base、转发 wrapper 或工厂层。
共享训练直接组织 setup/runner/step 与必要设施；相同配置只在入口校验，内部使用已解析结果。
不以文件数为指标，不通过合并改变概率分布、参数绑定、状态语义或完整 checkpoint 恢复契约。

## 命名要求
公开模型与配置类使用 LegoSpecModel、LegoSpecConfig，包名 legospec。
关键变量按所属轴和实际含义命名，如 target_hidden_states、context_hidden_states、backbone_state、proposal_output、loss_numerator。
跨模块不使用含糊的单字母或 data/features 名称；局部数学 q/k/v 可保留。
关键公共名称有多个合理方案时先向用户列出候选，普通局部变量自行使用清晰名称。

Default development host (user instruction, 2026-11-22): perform LEGOSpec code
changes and verification directly on SSH host H800_2. Follow later explicit host
instructions. Inspect the intended remote project before editing and preserve its
existing changes; do not overwrite them with an older local checkout.

The main development repository is `/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public` on H800_2 (confirmed 2026-11-22).

Use `/mnt/dai-sys/wanghanzhen/projects/DAC27/envs/legospec-public-py311-cu130-h8002/bin/python` for this main project; this entrypoint selects the pinned SGLang/Torch/CUDA environment documented in docs/SGLANG_RUNTIME.md.

## 开发迭代与训练恢复（用户明确要求，2026-11-22）

开发更新默认只维护当前实现。除非用户明确要求兼容旧版，不添加旧版本分支、适配层、版本标记或兼容性预检查，也不为保留历史行为而保留代码。
同一次训练恢复按原配置和 rank topology 直接加载实际状态；不比较版本号、配置身份、规划签名或源码指纹。
保留 optimizer、scheduler、FP32 master、每 rank RNG、训练游标和 pending plan 的完整保存/恢复。无效输入由实际读取或执行操作正常报错。
