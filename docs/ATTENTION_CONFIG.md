# Attention 配方

公开 API 仍为 `legospec.LegoSpecConfig` / `LegoSpecModel`，四轴边界不变。

- `dense`：`mha`、`gqa`、`mqa`，区别是 Q/KV 头数。
- `latent`：`mla`，训练使用展开式 K/V，eval 且 no_grad 时使用吸收式 MLA 和 latent cache。
- `linear`：`gdn`、`kda`。GDN 使用每头 decay，KDA 使用每头每个 key 通道的 decay。

Dense / latent 的历史访问方式独立选择 `full`、`swa`、`dsa`。例如：

```json
{"attention": {"family": "dense", "variant": "gqa", "access": "swa", "window_size": 512}}
```

```json
{"attention": {"family": "latent", "variant": "mla", "access": "dsa"}}
```

```json
{"attention": {"family": "linear", "variant": "kda"}}
```

公共层配置只包含 `attention` 和可选 `local_operator`。共享投影参数位于
`attention_configs.<variant>`；访问参数位于 `access_configs.full/swa/dsa`。
Linear 不配置 token access，也不提供 `block_attention` 开关。
每一层 GDN/KDA 固定包含同 block 双向 attention，包括第一层；两路共享 Q/K/V/O，
在共享输出投影前以逐 token、逐 value-head sigmoid gate 融合，gate 初值由
`attention_configs.gdn/kda.gate_init` 设置（默认 0.1）。KDA 复用相同事件布局与缓存，
历史只写 confirmed context，block read 不写状态。FLA KDA 的 key head dim 不超过 256。

现有数学顺序保持不变：规范化后的 GDN/KDA read 输出，加 gate × block SDPA 输出，
再经过共享 O 投影与残差，然后 MLP。内部 Q/KV depthwise convolution 不因分类变更而移除；
`dflash_gdn5_shared_block_attention_rnn.json` 与 `dflash_kda5_rnn.json` 无外部 grouped convolution。

DSA 当前仍要求所有层使用 DSA，SDPA 后端且无 selector proposal；它并不表示任意混合组合都受支持。
MLA 的吸收式 DSA 推理保留 latent cache，当前通过可见性 mask 执行；没有宣称它已具备稀疏计算加速。

## 历史配方

顶层 `configs/*.json` 是新训练配方，已迁移上述分类。旧文件名中的 battn3/battn4
仅保留命名来源；其中每层 linear 现在都内置 shared block attention，不能用新文件名推断旧权重结构。
当前代码只接受 family/variant/access 配方；旧 string 配方与独立 block 分支已删除。
新配方不允许把 block attention 单独配置。

## H800 环境

当前环境使用 Torch 2.13.0+cu130、Triton 3.7.1、FLA 0.5.2 和 TileLang 0.1.12；
TileLang 版本与 SGLang 0.5.21 的依赖一致。GDN/KDA 由 FLA 官方 backend dispatch 选择执行后端。

H800_2 主项目 Python 入口仍为
`/mnt/dai-sys/wanghanzhen/projects/DAC27/envs/legospec-public-py311-cu130-h8002/bin/python`。
升级环境独立安装 Torch/CUDA/SGLang，不再通过 .pth 依赖 LiteSpec 的旧环境。
安装方式及测试记录见 [SGLang runtime](SGLANG_RUNTIME.md)。

## 实现位置

Dense 位于 `attention/dense.py`，latent 位于 `attention/latent.py`，GDN/KDA 与内部 block attention 位于 `attention/linear/`；full/SWA/DSA 位于 `attention/access/`，Flex 等执行依赖位于 `attention/backends/`。详见 `legospec/modeling/DESIGN.md`。
