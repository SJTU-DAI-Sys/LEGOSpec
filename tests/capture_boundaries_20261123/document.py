from pathlib import Path
root=Path.cwd()
p=root/'legospec/modeling/target/DESIGN.md';s=p.read_text().replace('`gemma.py` 保留原生 embedding 缩放、softcap、native capture 初始化要求','`gemma.py` 保留原生 embedding 缩放、softcap、capture 选择和 final norm 重建').replace('Gemma native packed capture 算法继续在 `sglang_gemma4_capture.py`，离线采集和 adapter 共用；没有独立 Gemma training target 类。','通用 native packed capture 在 `sglang_backend/capture.py`，SGLang backend 创建引擎；TP 初始化在 `sglang_backend/parallel.py`。Gemma adapter 不创建 engine、不初始化分布式。在线训练和离线采集都调用 `get_training_target_model()`，没有模型专属构造入口。')
s+='''
2026-11-23 收尾：删除独立 Gemma capture 文件和 data 中 Gemma 分布式初始化分支。离线 collector 使用公共 LEGOSpec TP/DP mesh；native engine 在该 mesh 上建立 SGLang TP-scoped runtime，与在线训练一致。packed 请求边界、offsets、bulk aux/last 与 padded 训练还原保持一致。
缓存 reader 只负责读存储与检查维度；Gemma 权重 key、RMSNorm 和精度转换归 adapter。缓存当前 `recompute_gemma4_final_norm` 字段和 CLI 名称未迁移，不增加格式版本或兼容分支；后续若统一存储命名，应作为显式格式改动处理。
结构计划与其余边界检查见 `docs/TARGET_CAPTURE_PLAN.md`；本轮测试记录在 `tests/capture_boundaries_20261123/`。
''';p.write_text(s)
p=root/'legospec/modeling/target/sglang_backend/DESIGN.md';p.write_text('''# SGLang backend

- `capture.py`：NativeCapture 初始化 runner、内存池和 attention；执行独立请求的 packed prefill，输出层字典、normalized teacher、token offsets 和 bulk tensor views。结束或 forward 失败时清理请求与 KV 池。
- `parallel.py`：把 LEGOSpec TP/DP mesh 接入 native SGLang runtime，保留训练/采集已有的 WORLD；在线和离线共用。
- `model_runner.py` / `patch.py`：现有 communicator capture 执行路径的 runner 与进程组实现。
- `utils.py`：该路径的 logits processor/capture 输出适配。

两条执行路径由上层 `sglang.py` 根据 adapter 的真实 capture 接口选择；不是每种模型复制一套 backend。Gemma adapter 只描述模型语义，不持有 engine 工厂。
此目录为命名空间包。没有 `__init__.py` 提前导入另一条 runner 的可选依赖；消费者直接导入所需实现。
原生捕获依赖 SGLang native aux capture/runtime_context API。主项目环境是否具备它由实际导入/运行验证；单元协议测试不能代表真实大模型实载成功。
''')
p=root/'legospec/data/DESIGN.md';p.write_text(p.read_text()+'''\n在线/离线 target 都由 modeling/target 的同一构造入口创建。collector 不按 Gemma/Qwen 分支加载 runner 或初始化 WORLD；其 batch/context 容量需求作为 backend 参数传入。cache reader 调用模型 adapter 的 final norm 重建组件，不读取模型专属权重 key 或实现模型数学。\n''')
p=root/'docs/COMPONENT_ORGANIZATION.md';p.write_text(p.read_text()+'''\n2026-11-23：native packed capture 与 TP runtime 收入 target/sglang_backend；在线和离线共用 target factory 与 LEGOSpec 分布式初始化。Gemma 的 final norm 重建也归 adapter。其余边界检查和后续结构计划见 TARGET_CAPTURE_PLAN.md。\n''')
p=root/'docs/SIMPLIFICATION_PLAN.md';p.write_text(p.read_text()+'''\n2026-11-23 capture 边界收尾：删除 Gemma 专属 capture 文件/数据层构造分支；native capture 和并行运行时归 backend，模型 final norm 重建归 adapter。其余尚未实施的边界建议详见 TARGET_CAPTURE_PLAN.md。\n''')
(root/'docs/TARGET_CAPTURE_PLAN.md').write_text('''# Target capture 结构与边界检查（2026-11-23）

本轮目标已落实到代码；以下分开记录已实现结构与检查后建议，避免把计划当成已有接口。

## 已实现结构

```text
legospec/modeling/target/
├── training_target_model.py       # 统一构造入口与训练输出
├── hf.py                         # HF forward / shared components
├── sglang.py                     # capture 路径选择、padded/packed 输出与 scorer
├── components.py                 # 共享 embedding/head、TP 与权重加载
├── adapters/
│   ├── base.py                   # 默认模型结构与捕获语义
│   ├── gemma.py                  # 缩放、softcap、配置、final norm 重建
│   └── qwen35.py                 # Qwen3.5/3.6 特有捕获接口
├── sglang_backend/
│   ├── capture.py                # 通用 native packed capture 执行与资源生命周期
│   ├── parallel.py               # native TP runtime 接入已有 WORLD
│   ├── model_runner.py           # communicator 路径 runner
│   ├── patch.py                  # communicator 路径分布式运行时
│   └── utils.py                  # logits / capture 输出适配
├── anchor_opd.py                  # 当前 Qwen3 HF anchor 数学
└── sglang_anchor_opd.py           # 当前 Qwen3 SGLang anchor 执行

legospec/training/target/
├── setup.py                      # 训练需求、预算、tokenizer/mask、装配
├── features.py                   # 按层选择/拼接/设备搬运
└── streaming.py                  # producer、packet 与流水线调度

legospec/data/
├── capture_engine.py             # 采集调度、D2H、写缓存；统一 target factory
└── hidden_cache.py               # 缓存读取；final norm 数学调用 adapter
```

在线/离线：调用 `get_training_target_model()` → 选择 adapter + HF/SGLang → SGLang 按 capture 接口选择执行实现。两端先建 LEGOSpec TP/DP mesh；native capture 在该 mesh 中初始化 SGLang 进程组。
离线使用 `generate_capture_data(records)` 获取紧凑 token 布局；在线使用 `generate_training_data(...)` 返回 batch-first 张量。模型层决定语义，调用方决定数据形状与训练/采集预算。
没有 `sglang_gemma4_capture.py` 或旧路径转发；adapter 不再构建 runner。原始 decoder residual 和 normalized teacher 保持显式分开。

## 本轮额外修正

1. 离线 Gemma 专属 WORLD 初始化改为公共分布式入口，消除数据层对模型类型的判断。
2. Gemma final norm 权重 key、RMSNorm 数学与 FP32 转换移到 adapter。reader 只拿可调用 norm，继续读取/校验张量。仅需配置元数据的缓存读取不再依赖 Transformers 注册完整模型类。
3. 缓存空间估算没有模型数学，保留数据层并改名为 `check_capture_space`。
4. 移除 sglang_backend 包的 eager runner 导入，避免加载 native capture 时先加载另一执行路径的可选依赖。

## 其他边界检查：后续建议，尚未实施

优先级一：Objective 反向依赖 training。
`modeling/draft/composable/objective/base.py` 和 `dpal.py` 导入 `training/outputs.py`；base 还导入 `training/data/batch.py::position_weights`。
建议将损失输出契约归 objective/contracts 或现有 objective/base，训练/logger 只消费契约；position_weights 只有 Objective 调用，可直接并入 Objective.loss_weights，删除训练侧中转函数。验证所有损失数值、梯度、窗口分母与 logger 聚合。

优先级一：推理中的模型内部访问。
`inference/generate.py::TargetFeatures` 直接使用 `target.model.layers`；BlockDecoder / HiddenFeedbackDecoder 直接取 target embedding/head，没有统一承接模型 softcap 等共享组件语义。
建议将 HF target 的层定位与共享组件交给已有模型 adapter/components；生成循环继续拥有 KV cache、接受/拒绝、EOS 与预算。验证 block/hidden-feedback 两种模式的概率、缓存裁剪与 teacher 特征；不能仅更换类名后假定行为一致。

优先级二：config_utils 混合模型元数据与 draft 配方转换。
`load_text_model_config` 对 Qwen3.5 会转换成 Qwen3 draft 配置，而 Gemma 的 draft 转换已在 adapter.draft_config_values。建议分离 native text metadata 读取和 draft 配置派生，把 Qwen 派生规则归 adapter；HF/SGLang 构造及缓存读取只消费 native metadata。需保留 Qwen MoE/shared FFN 宽度和 RoPE 数值。

优先级二：custom_backend 的完整模型副本。
静态检查在活动 legospec/scripts/tests/unit 中未找到目录外消费者；目录内部仍维护多套完整模型。建议先核查动态加载/发布入口，再移到 development 或删除，避免形成第二条目标模型维护路径。没有在本轮删除这些未涉及文件。

保留的合理边界：anchor scorer 的 Qwen3 专属数学有明确模型限制，属于尚未泛化的真实实现，不应伪装成默认任意模型 scorer；chat template 属于数据格式而非 target 计算；cache 编码、传输、磁盘预算继续归 data。现有缓存字段/CLI `recompute_gemma4_final_norm` 是存储命名，未在本轮改变格式。

## 验证及限制

本轮增加 native capture 初始化、请求边界、token offsets、池释放、padded/packed 同源特征和 collector 公共入口测试；运行完整单元/GPU回归。
native runtime 接口测试采用受控 SGLang fixture，模型特征的 HF 小模型测试与 CUDA D2H/pinned-buffer 测试执行实际计算。
主项目固定环境的真实 native capture 导入结果单独记录在 tests/capture_boundaries_20261123/environment.json。不能把单元协议通过解释成 Gemma 大模型实载通过。
''')
