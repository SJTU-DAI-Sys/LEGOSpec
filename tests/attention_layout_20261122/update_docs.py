from pathlib import Path
import shutil,json
root=Path('/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public');work=root/'tests/attention_layout_20261122'
manifest=json.loads((work/'files.json').read_text())
def save(rel,text):
 p=root/rel;b=work/'backup'/rel
 if p.exists() and not b.exists():b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 if rel not in manifest['updated']:manifest['updated'].append(rel)
a='legospec/modeling/draft/composable/backbone/attention/'
save(a+'DESIGN.md','''# Attention

按参数化家族、历史访问、执行后端分组。配置继续使用 family/variant/access；目录没有新增配置轴。

```text
attention/
├── __init__.py       现有 dense/latent 构建入口
├── dense.py          MHA/GQA/MQA 投影与 attention
├── latent.py         MLA 投影、展开训练、吸收推理与 latent cache
├── softmax.py        dense/latent 共用的 KV 提交与 softmax 执行
├── linear/
│   ├── recurrence.py GDN/KDA 递归计算、read/write 与内部卷积
│   ├── block.py      linear 内部共享投影、逐头 gate block attention
│   ├── events.py     confirmed context 写事件与 draft 读事件布局
│   └── cache.py      递归状态定义及确认前缀提交
├── access/
│   ├── visibility.py full/SWA/DSA 的可见性
│   ├── indexer.py    DSA 的 anchor query 与历史打分、top-k
│   └── dsa.py        选中 KV 的 gather attention 与 teacher 统计
└── backends/
    ├── flex.py       FlexAttention 编译与 block mask
    └── torch_compat.py PyTorch 编译兼容补丁
```

MHA/GQA/MQA 共用 GQAAttention；MLA 的全部投影与 RoPE 逻辑留在 latent.py。
softmax.py 只保留这两类真实共用的 scaffold、缓存与执行，不存放某一家族的投影。
linear/ 内 GDN/KDA 共用 GDNContextMemoryMixer，KDA 使用逐 key 通道 decay。
block.py 是 linear 内部数学，公共配方不提供独立 block attention 开关。
access/ 同时服务 dense 和 latent；backends/ 是可选执行依赖，不负责模型分类。
这些子目录仅承载实现，不引入转发 wrapper、独立算法模型或新的类注册层。

MLA 的训练及启用梯度调用保持展开式 K/V；eval 且 no_grad 使用吸收式计算。
MLALayerState 保存 normalized KV latent、旋转后的独立 RoPE key 和位置/有效性/document 元数据。
吸收式 DSA 当前通过 visibility mask 计算，不宣称已获得稀疏计算加速；其余后端选择不变。

模块迁移保留类名、参数绑定、state_dict 路径及计算顺序。
内部 Python import 路径统一迁移，没有保留旧文件的转发壳。旧 checkpoint 权重与原始配方按原语义读取。
回归见 tests/unit/test_attention_taxonomy.py、test_mla_absorb.py、test_gdn_shared_block_attention.py；
迁移前后逐位一致性证据位于 tests/attention_layout_20261122/。
''')
save(a+'linear/DESIGN.md','''# Linear attention

recurrence.py 拥有 GDN/KDA 的投影、内部卷积、gate decay 与 scan；block.py 拥有其内部共享投影 block attention。
events.py 是两者共同使用的读写事件布局，cache.py 将状态类型和前缀提交放在一起。
消费方为统一 Backbone/Layer；状态类型还供 contracts 的类型标注使用。不得在此引入 MLP、残差或 Proposal。
''')
save(a+'access/DESIGN.md','''# Attention access

visibility.py 定义 full、SWA、DSA 可见性；indexer.py 对 confirmed history 打分并选择 top-k；dsa.py 执行稀疏 gather attention。
可见性服务 dense 与 latent，indexer 由 Backbone 持有且跨层共享，状态和参数路径仍为 backbone.indexer。
这里不拥有 Q/K/V/O 投影，也不新增 family 层级的模型实现。
''')
save(a+'backends/DESIGN.md','''# Execution backends

flex.py 提供 dense/latent 和共享训练调用的 FlexAttention 编译与 block-mask 操作；torch_compat.py 保留它实际依赖的兼容补丁。
这些是执行依赖，不构成新的 attention 类型或配方。GDN/KDA 使用的 FLA scan 仍由 linear/recurrence.py 按需加载。
''')
p=root/'legospec/modeling/draft/composable/backbone/DESIGN.md';s=p.read_text().replace('indexer 负责稀疏选择。','attention/access/indexer.py 负责稀疏选择。').replace('`convolution.py`, `indexer.py`, `kernels.py`, `model.py`','`convolution.py`, `kernels.py`, `model.py`');save(str(p.relative_to(root)),s)
p=root/'docs/COMPONENT_ORGANIZATION.md';s=p.read_text();s=s.replace('详见 ATTENTION_CONFIG.md。','实现分别归入 attention/dense.py、latent.py、linear/；access/ 管历史访问，backends/ 管执行后端。详见 ATTENTION_CONFIG.md 和 attention/DESIGN.md。');save(str(p.relative_to(root)),s)
p=root/'docs/ATTENTION_CONFIG.md';s=p.read_text();s+='\n## 实现位置\n\nDense 位于 `attention/dense.py`，latent 位于 `attention/latent.py`，GDN/KDA 与内部 block attention 位于 `attention/linear/`；full/SWA/DSA 位于 `attention/access/`，Flex 等执行依赖位于 `attention/backends/`。详见 `legospec/modeling/draft/composable/backbone/attention/DESIGN.md`。\n';save(str(p.relative_to(root)),s)
p=root/'tests/unit/test_attention_taxonomy.py';s=p.read_text()+'''

def test_mla_optimizer_head_geometry_after_module_move():
    from legospec.training.optimizer import register_optimizer_parameters
    model = LegoSpecModel(config("mla"))
    parameters, specs = register_optimizer_parameters(model)
    assert len(parameters) == len(list(model.parameters()))
    by_name = {spec.name: spec for spec in specs}
    query = by_name["backbone.layers.0.self_attn.q_b_proj.weight"]
    key_value = by_name["backbone.layers.0.self_attn.kv_b_proj.weight"]
    assert (query.optimizer, query.muon_heads, query.muon_head_dims) == ("muon", 2, (8,))
    assert (key_value.optimizer, key_value.muon_heads, key_value.muon_head_dims) == ("muon", 2, (4, 8))
''';save(str(p.relative_to(root)),s)
(work/'files.json').write_text(json.dumps(manifest,indent=2));print('Updated ownership docs and MLA optimizer regression')
