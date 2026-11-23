# Modeling

负责 draft 模型计算、冻结 target 的执行与特征接口，以及模型配置和 KV cache。
公开类为 LegoSpecConfig、LegoSpecModel。训练循环、优化器和数据读写分别归 training、data。

## 根目录

- config_utils.py：读取目标模型配置、提取文本配置，处理 Qwen3.5/3.6 的 draft 配置派生。
- kv_cache.py：目标模型推理使用的预分配 KV cache。
- __init__.py：包入口。

## draft/：模型入口和配置

- configuration.py：LegoSpecConfig、默认目标层选择及配置序列化。
- schema.py：公开配方的字段、类型、默认值和组合校验。
- runtime_config.py：将公开配置转换为执行时使用的逐层参数。
- loss_config.py：目标类型、loss 权重、归一化方式和 proposal 相关配置解析。
- modeling.py：LegoSpecModel，组装 Context → Backbone → Proposal，提供 block 和 hidden-feedback 调用接口。
- contracts.py：DraftInputs、各轴输出和共享 embedding/head 等调用契约。
- export.py：收集推理依赖，生成可独立加载的 Transformers 模型包。
- __init__.py：公开模型、配置、输入输出和状态类型。

配置入口完成校验，内部消费已解析结果。各组件拥有自己的参数和特殊初值；
顶层模型在 HF 初始化后统一调用组件的 reset_draft_parameters。

## draft/context/：目标特征融合

- fusion.py：ConcatProjectionContext 将选中层拼接投影；GatedFusionContext 按 draft 层对目标特征做标量或通道 gate 融合。
- __init__.py：按配置构建 Context。

Context 只处理目标特征融合，输出供 Backbone 使用的逐层特征。

## draft/backbone/：主干层与执行状态

- model.py：Backbone 组织层循环、indexer 和缓存发布；Layer 组织 norm、attention、残差、外部卷积和 MLP。
- convolution.py：外部 grouped convolution 及其初始化。
- state.py：dense/latent/indexer 状态、整体 DraftState 和 rollback。
- stepwise.py：训练与解码共用的 hidden-feedback 调度、mask 和状态扩展。
- __init__.py：构建 Backbone。

每层持有独立 attention 子模块。Linear 的位置加法仍在 Layer 的 input norm 和残差之前执行；
各层成功后才发布确认历史的新状态。

## draft/backbone/attention/：注意力计算

- __init__.py：按家族构建 attention。
- base.py：dense/latent 共用的 softmax 执行、KV 提交和 backend 调用。
- dense.py：MHA/GQA/MQA 投影、RoPE 和 attention。
- latent.py：MLA 投影、展开计算、吸收式推理和 latent cache。

### linear/

- recurrence.py：LinearAttention，以及 GDN/KDA 的历史 mixer、内部卷积、decay、scan 和共享输出投影。
- block.py：block 内双向 SDPA 与逐头 gate。
- events.py：confirmed context 写事件和 draft 读事件的布局。
- cache.py：递归状态、卷积尾部状态及确认前缀提交。

GDN/KDA 固定包含共享 Q/K/V/O 的 history read 与 gated block read，block read 不写历史状态。

### access/

- visibility.py：full、SWA、DSA 的位置、文档和有效性 mask。
- indexer.py：anchor query 对历史打分、top-k 选择及辅助监督。
- dsa.py：按选中索引 gather K/V 后计算 attention 和 teacher 统计。

### backends/

- flex.py：FlexAttention 和 block mask 的编译、调用及后端参数。
- torch_compat.py：当前执行环境需要的 PyTorch 编译适配。

家族决定参数化，access 决定历史访问，backend 决定算子执行。

## draft/proposal/：从主干特征提出 draft token

- parallel.py：独立位置的词表投影，以及共享输出处理。
- markov.py：低秩前驱 token bias、Markov/RNN head 和对应 Proposal。
- sequential.py：按 token 顺序递推的 hidden replacement Proposal。
- selector.py：候选选择、候选集评分及全词表残差 Proposal。
- sampling.py：概率归一化、采样和请求级随机种子。
- __init__.py：按配置构建 Proposal。

Proposal 负责预测分布和 token；loss 数学交给 Objective，接受/拒绝交给 inference。

## draft/objective/：监督和损失

- base.py：Supervision、公共 Objective 接口和位置权重。
- outputs.py：RatioTerms、ObjectiveOutput，保存 loss 和指标的分子/分母。
- token.py：逐 token CE、KL、TV、L1、NLA 和候选集 CE。
- block.py：DPACE、E2E-TV 等跨位置目标。
- dpal.py：D-PAL Rényi 目标及其独立梯度数学。
- composition.py：具名损失分支、Selector loss 组合和加权目标。
- vocab.py：teacher 词表分布、索引投影和局部复用缓存。
- kernels.py：共享词表损失计算与编译入口。
- __init__.py：构建 Objective。

Objective 不导入 training。训练侧消费输出，负责累积窗口和分布式归约。

## target/：冻结目标模型

- training_target_model.py：统一 backend 构造入口、特征输出契约和抽象接口。
- hf.py：HF 加载、forward、层捕获和共享组件。
- sglang.py：SGLang 统一入口、packed capture 与 padded 训练输出转换。
- components.py：共享 embedding/head、缩放/softcap、词表 TP 通信和独立组件权重加载。
- __init__.py：导出 target 接口。

输出中，hidden_states 是按 decoder 层号组织的原始输出；
最后一层 residual 与 final norm 后的 final_hidden_state 分开。

### adapters/

- base.py：默认模型结构、层定位、capture 和组件语义。
- gemma.py：Gemma embedding 缩放、softcap、配置及 final norm 重建。
- qwen35.py：Qwen3.5/3.6 的 native capture 差异。

Adapter 处理模型语义；backend 管执行、批次、通信和资源生命周期。

### sglang_backend/

- capture.py：native packed prefill、请求边界、辅助隐藏态和 teacher 捕获、KV 池释放。
- parallel.py：将 SGLang runtime 接入 collocated TP 或已有的 disaggregated target 进程组。

在线训练和离线采集共用 target 构造与 capture 入口。

### custom_backend/

gpt_oss.py、llama.py、llama4.py、phi3.py、qwen2.py、qwen3.py、qwen3_moe.py
保存各架构的定制完整模型实现；当前统一 HF/SGLang 构造入口未使用此目录。

OPD 及暂缓目标位于仓库 development/，当前不属于正式模型接口。
