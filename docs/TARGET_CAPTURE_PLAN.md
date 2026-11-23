# Target capture 结构与边界检查（2026-11-23）

本轮目标已落实到代码；以下分开记录已实现结构与检查后建议，避免把计划当成已有接口。

## 已实现结构

```text
legospec/modeling/target/
├── training_target_model.py       # 统一构造入口与训练输出
├── hf.py                         # HF forward / shared components
├── sglang.py                     # 统一 native 执行、padded/packed 输出
├── components.py                 # 共享 embedding/head、TP 与权重加载
├── adapters/
│   ├── base.py                   # 默认模型结构与捕获语义
│   ├── gemma.py                  # 缩放、softcap、配置、final norm 重建
│   └── qwen35.py                 # Qwen3.5/3.6 特有捕获接口
└── sglang_backend/
    ├── capture.py                # 通用 native packed capture 执行与资源生命周期
    └── parallel.py               # collocated/disaggregated TP runtime 接入已有 WORLD

legospec/training/target/
├── setup.py                      # 训练需求、预算、tokenizer/mask、装配
├── features.py                   # 按层选择/拼接/设备搬运
└── streaming.py                  # producer、packet 与流水线调度

legospec/data/
├── capture_engine.py             # 采集调度、D2H、写缓存；统一 target factory
└── hidden_cache.py               # 缓存读取；final norm 数学调用 adapter
```

在线/离线：调用 `get_training_target_model()` → 选择 adapter + HF/SGLang → SGLang 统一使用 native runner。两端先建 LEGOSpec TP/DP mesh；collocated 创建 TP 子组，disaggregated 复用分工前创建的 target 子组。
离线使用 `generate_capture_data(records)` 获取紧凑 token 布局；在线使用 `generate_training_data(...)` 返回 batch-first 张量。模型层决定语义，调用方决定数据形状与训练/采集预算。
没有 `sglang_gemma4_capture.py` 或旧路径转发；adapter 不再构建 runner。原始 decoder residual 和 normalized teacher 保持显式分开。

## 本轮额外修正

1. 离线 Gemma 专属 WORLD 初始化改为公共分布式入口，消除数据层对模型类型的判断。
2. Gemma final norm 权重 key、RMSNorm 数学与 FP32 转换移到 adapter。reader 只拿可调用 norm，继续读取/校验张量。仅需配置元数据的缓存读取不再依赖 Transformers 注册完整模型类。
3. 缓存空间估算没有模型数学，保留数据层并改名为 `check_capture_space`。
4. 移除 sglang_backend 包的 eager runner 导入，避免加载 native capture 时先加载另一执行路径的可选依赖。

## 其他边界检查：后续建议，尚未实施

已实施：Objective 输出与位置权重归 draft/objective，training 消费其契约；原反向依赖已消除。见 legospec/modeling/DESIGN.md。

优先级一：推理中的模型内部访问。
`inference/generate.py::TargetFeatures` 直接使用 `target.model.layers`；BlockDecoder / HiddenFeedbackDecoder 直接取 target embedding/head，没有统一承接模型 softcap 等共享组件语义。
建议将 HF target 的层定位与共享组件交给已有模型 adapter/components；生成循环继续拥有 KV cache、接受/拒绝、EOS 与预算。验证 block/hidden-feedback 两种模式的概率、缓存裁剪与 teacher 特征；不能仅更换类名后假定行为一致。

优先级二：config_utils 混合模型元数据与 draft 配方转换。
`load_text_model_config` 对 Qwen3.5 会转换成 Qwen3 draft 配置，而 Gemma 的 draft 转换已在 adapter.draft_config_values。建议分离 native text metadata 读取和 draft 配置派生，把 Qwen 派生规则归 adapter；HF/SGLang 构造及缓存读取只消费 native metadata。需保留 Qwen MoE/shared FFN 宽度和 RoPE 数值。

优先级二：custom_backend 的完整模型副本。
静态检查在活动 legospec/scripts/tests/unit 中未找到目录外消费者；目录内部仍维护多套完整模型。建议先核查动态加载/发布入口，再移到 development 或删除，避免形成第二条目标模型维护路径。没有在本轮删除这些未涉及文件。

2026-11-23：OPD scorer、采样/回放和训练集成已移到 development/opd，当前版本不支持。其余保留的边界：chat template 属于数据格式而非 target 计算；cache 编码、传输、磁盘预算继续归 data。现有缓存字段/CLI `recompute_gemma4_final_norm` 是存储命名，未在本轮改变格式。

## 验证记录

边界整理时的 0.5.9 导入失败保留在 tests/capture_boundaries_20261123/environment.json，属于升级前记录。
当前 runtime、真实模型测试及完整回归结果见 [SGLang runtime](SGLANG_RUNTIME.md)。
