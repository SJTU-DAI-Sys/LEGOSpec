# Qwen3.8-27B draft：qz 训练准备

主代码在 H800_2 的 `/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public` 维护。本配方部署到独立的 qz 目录 `/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/LEGOSpec_qwen38_27b_train_20261124`，使用该目录 `.venv`；不修改正在生成数据的旧 LEGOSpec/litespec 环境。

## 配方与 target 层

- DFlash context：concat_projection；target decoder 的原始输出层号为 **[1, 11, 21, 31, 41, 51, 61]**（0-based）。
- 对应第 **2、12、22、32、42、52、62** 个 decoder block；若使用 Hugging Face 含 embedding 的 hidden_states tuple，索引也是 **[2,12,22,32,42,52,62]**。
- 选择规则沿用现有均匀选层：`round(1 + i*(64-4)/(7-1))`，i=0..6。7 层各 5120 维，拼接 35840 维并投影到 5120 维。
- target block 1/21/41/61 为 linear attention，11/31/51 为 full attention。这些是目标层类型，与 draft 的 SWA/full 排布独立。
- teacher 监督另外使用最后一个 block（0-based 63）的 final RMSNorm 后输出；它不加入上述 7 层 context 拼接。
- draft：5 层，前 4 层 GQA SWA-2048，最后 1 层 GQA full；每层外部 grouped conv，kernel=2、group_size=16（attention 和 MLP 分支）。
- draft 每层：32 Q heads、8 KV heads、head_dim=128；target 原生为 24/4/256，保持原生 target。
- draft hidden_size=5120、FFN intermediate_size=17408，继承 27B target；可训练参数 **1,976,833,280**，不含冻结的 target embedding/head。
- proposal：full-vocabulary residual_selector，rank=256、top_k=16、stop_gradient=false。
- loss：`0.2 * KL(p_target || q_base) + 1.0 * D-PAL_Renyi(q_final,p_target)`；rho=0.3、complete_block_mean、无位置衰减。
- block_size=8，预测 7 个后续 token。

## qz 训练配置

`train_qz_node0.yaml` / `train_qz_node1.yaml`：dai-sys-1 两节点，各 8 张 H100，使用**在线共置模式**（disaggregate=false）。
每节点 4 个 TP2 target 组；所有 16 个 rank 都运行 draft，FSDP 组覆盖全部 16 ranks。target 每组 batch=2，shard_draft_by_tp=true，每个 TP rank 训练其中一条不同样本。
每个 microstep 全局 16 条样本，梯度累积 8 次，每次 optimizer 更新有效全局 batch=128。

`training_mode=optimized`，`schedule_window_updates=2`：在 2 个 optimizer update（16 个 microstep、全局 256 个样本）的 CPU 元数据窗口中估计代价并均衡工作，改变窗口内的样本分组和顺序。`async_schedule_planning=true`：每个 rank 使用一个 CPU 后台线程，在当前窗口训练时规划下一个窗口。规划读取已预处理的 token/mask，不提前做 target GPU 前向，不改变梯度累积、样本/anchors 或 loss 分母。

当前在线输入参数为 `dataloader_num_workers=1`、`dataloader_prefetch_factor=1`：每 rank 一个 DataLoader worker、最多一批队列（两机合计 16 个 worker）。`cpu_batch_prefetch=0`、`h2d_prefetch=false`；launcher 的 OMP/MKL 默认各 1 线程。CPU batch/H2D 预取仍仅用于离线 hidden cache。

当前训练参数：4 epochs、max_length=8192、512 anchors、Muon lr=0.01 / AdamW lr=6e-4、warmup=4%、seed=42；数据生成温度 T1 写入 W&B 名称。使用 FlexAttention/TRITON、optimized packed execution；backbone 和 objective gradient checkpointing 均关闭，objective 每次处理 16 blocks。
Target TP=2，max_running_requests=2、context_length=8193、max_total_tokens=16384；共置时 SGLang mem_fraction_static=0.45，为 draft 留出显存。

数据直接读取 qz 当前生成目录的 `default/part[01].jsonl`，两个分片在读取时组成同一个数据集。`chat_template=qwen3.8`，`train_data_columns=null`；原生 medium 模板将 reasoning_content 和 content 合并为完整 assistant 回答，所有 assistant 轮的思考和答案均作为监督区域。
当前仍按训练 max_length=8192 直接截断超长对话，完整回答切窗尚未实现；不能保证所有长样本最后一轮完整。2026-11-25 的思考/答案 loss mask 审计保存在 thinking_audit_20261125/。

每 15000 optimizer steps 保存完整可恢复 checkpoint，并保存最终权重；包含 optimizer/scheduler、FP32 master、RNG、游标和 pending scheduler plan。正式恢复用原配置与相同 rank topology 传 `--resume-from PATH`。
本次实际输出：`/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/runs/qwen38_27b_dflash_swa2048_rselector_dpal_s42_20261124/checkpoints_2`。
W&B 默认 offline，独立项目 `legospec-qwen38-27b-dflash-draft`，固定 run id。

## 启动命令（当前两机已运行，请勿重复启动）

生成两个 part 都产生 complete.json 后，在对应 GPU 网页终端执行：

```bash
cd /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/LEGOSpec_qwen38_27b_train_20261124
# worker 0 / qb-prod-gpu091
bash configs/qwen38_27b_dflash_swa2048_rselector_dpal/run_qz.sh 0
# worker 1 / qb-prod-gpu119（在另一个节点执行）
bash configs/qwen38_27b_dflash_swa2048_rselector_dpal/run_qz.sh 1
```

launcher 会检查生成完成文件；未完成时直接退出，不会停止生成或 keeper。实际启动时复用已验证的 keeper_lifecycle，在每个节点停止 keeper，训练完成或退出后恢复。`--dry-run` 不更改 keeper 或启动 GPU 工作。
当前 master 使用 qb-prod-gpu091，CPU 节点已验证 DNS 解析；若作业重建、更换 worker 0，两个节点设置相同的 `PET_MASTER_ADDR=新的worker0地址`。

安装脚本 `bootstrap_qz.sh` 使用 uv.lock，CUDA 13 工具链位于已有共享 tools/cuda-13.0；安装到独立 .venv，不改生成环境。
`run_qz.sh` 负责上述环境、rank、完成检查和已有 keeper 生命周期的组装，消费者为两个 qz GPU 节点。

## 验证边界

2026-11-25 按用户更新：max_length=8192、accumulation_steps=8、muon_lr=0.01；backbone_gradient_checkpointing=false、objective_gradient_checkpointing=false。target context=8193，target max_total_tokens=16384。两节点配置均经入口解析。以下短序列 smoke 是早期记录；当前 16 卡正式训练已通过首个 optimizer update，见末尾运行验证。


H800_2：真实 27B TP2 捕获全部 7 层及归一化 teacher；在线/离线结果一致、teacher logits 一致。完整 1.98B draft 使用真实特征通过 production FlexAttention/packed/checkpointing 的 KL+D-PAL 前向与反向。短序列验证不代表已完成 qz 16 卡 8K 长序列训练或吞吐测试。
具体环境、配置、数据读取、测试和 smoke 证据保存在 `tmp_exp/qwen38_27b_training_20261124/`，并导出到本地交付目录。

qz 环境验收通过：Python 3.11.15、Torch 2.13.0 / CUDA 13.0、SGLang 0.5.21、Transformers 5.12.1。causal-conv1d CUDA 扩展可加载；两节点配置解析及启动 dry-run 通过；8 条真实样本（两分片的头尾各取样）原生 medium/thinking token 对齐通过。报告为 tmp_exp/qwen38_27b_training_20261124/qz_environment_validation.json。该报告覆盖环境和数据接口，实际训练验证见下文。

最新共置验证（H800_2，TP2）：真实 27B、8192 tokens、每 TP rank 一条不同样本、W=2、累积=8、主干/objective checkpointing=false，完整前向和反向通过。峰值 allocated=60.36 GiB、reserved=72.17 GiB。另有 25 项调度、17 项后端、2 项长上下文 mask 输出/梯度测试通过，qz 实际环境新增 API 回归和两节点 dry-run 通过。运行时修复移除了 SGLang 0.5.21 不再接收的 piecewise graph 参数；packed BlockMask 改为 eager 构建以避开 Torch 2.13 编译错误，FlexAttention 内核仍编译执行。

在线异步规划验收（2026-11-25）：H800_2 相关 CPU 回归 43 项通过；qz 实际环境 12 项通过，两个节点 dry-run 通过。H800_2 的 8 rank 并发输入流水线对照使用 64 条真实生产样本、8K token 上限、64 microsteps/rank，并以每步 50ms sleep 模拟 GPU 时间。同步/2 workers/OMP4 的累计稳态输入等待约 2.31s；异步/0 workers/OMP1 约 0.32s，CPU 总耗时从 88.9 降至 24.3 core-seconds。异步/0 workers/OMP4 仅快约 0.8%，却消耗约 3.5 倍 CPU。更深队列没有稳定收益，该性能基准保留作对照；当前按用户要求采用每 rank 1 worker、prefetch_factor=1、OMP/MKL1。该测试使用重复的小型 warm-cache Arrow 数据集，不能等同于 qz 16 GPU 或全数据集冷缓存训练吞吐。实验报告为 tmp_exp/qwen38_27b_training_20261124/online_planning_recommendation.json。

启动超时修复（2026-11-25）：NativeCapture 强制关闭 FlashInfer AllReduce Fusion，使用普通 TP NCCL 通信，避免 Qwen 模型自动启用的融合工作区 JIT/锁等待。`dist_timeout=60` 的单位为分钟；训练路径将其转换为 3600 秒传给 SGLang，target TP 子组继承同一时限。所有 target replica 构造完成后通过 WORLD barrier 再进入数据加载。全量数据预处理与 anchor 资格过滤均使用 128 个进程，独立在 CPU 上先完成共享缓存；随后两节点再启动 GPU 训练。31 项 native capture、超时传递和在线调度回归在 H800_2 与 qz 实际环境均通过。

此前 1 轮配置的两机实跑完成首个 optimizer update：`optimizer_step=1 micro_step=8 loss=10.0187`，随后微步持续推进。原始成功样本 1,410,082 条，128 进程分词/有效 anchor 过滤耗时约 683.5 秒，可训练样本 1,410,081 条。运行验证报告为 tmp_exp/qwen38_27b_training_20261124/startup_repair_running_validation.json。

两机配置及实际启动命令均已修正为 `num_epochs=4`。原 1 轮任务按用户指令停止，最后记录为 optimizer_step=750；其日志保存在 `logs/train.node{0,1}.epoch1_stopped_before_e4_20261125.log`。新任务从头训练，warmup/cosine 调度按 4 轮重新构建；W&B 使用独立 run id `qwen38_27b_dflash_swa2048_rsel_kl02_dpal_e4_s42_20261125`，实际输出为 `checkpoints_2`。CUDA Graph 保持关闭，其余训练参数不变。
