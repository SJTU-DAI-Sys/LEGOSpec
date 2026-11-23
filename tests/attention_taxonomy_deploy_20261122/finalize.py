from pathlib import Path
import shutil,json
root=Path('/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public');work=root/'tests/attention_taxonomy_deploy_20261122'
changes=json.loads((work/'changed_files.json').read_text())
def save(rel,text):
 p=root/rel;b=work/'backup'/rel
 if p.exists() and not b.exists():b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 if rel not in changes:changes.append(rel)
def edit(rel,a,b):
 s=(root/rel).read_text();assert a in s,(rel,a);save(rel,s.replace(a,b))
t=(work/'test_taxonomy.py').read_text().replace('def test_token_access(variant,access):','def test_token_access(variant,access,monkeypatch):').replace("  c.composable_config['access_configs']['dsa']['execution']='reference'","  from dsa_reference import reference_attention\n  from legospec.modeling.draft.composable.backbone.attention import sparse\n  monkeypatch.setattr(sparse, 'gather_attention', reference_attention)")
save('tests/unit/test_attention_taxonomy.py',t)
# Remove duplicate collected test, keeping initial copy in backup for reproducibility.
shutil.move(str(work/'test_taxonomy.py'),str(work/'backup/test_taxonomy_initial.py'))
changes.remove('tests/attention_taxonomy_deploy_20261122/test_taxonomy.py')
edit('tests/unit/test_gdn_shared_block_attention.py','''    baseline_recipe = deepcopy(recipe)
    for layer in baseline_recipe["composable_config"]["backbone"]["layers"]:
        del layer["block_attention"]
    with torch.device("meta"):
        baseline = LegoSpecModel(LegoSpecConfig(**baseline_recipe, **common))''','''    from legospec.modeling.draft.composable.schema import _execution_recipe
    baseline_config = _execution_recipe(LegoSpecConfig(**recipe, **common))
    del baseline_config._intrinsic_linear_attention
    for layer in baseline_config.composable_config["backbone"]["layers"]:
        del layer["block_attention"]
    with torch.device("meta"):
        baseline = LegoSpecModel(baseline_config)''')
edit('tests/unit/test_gdn_block_attention.py','''        assert layer["block_attention"] == dict(
            num_heads=32, head_dim=128, gate_init=0.1, share_gdn_projections=True
        )''','''        assert "block_attention" not in layer
        assert layer["gdn_config"]["intrinsic_block_attention"] is True
        assert layer["gdn_config"]["gate_init"] == 0.1''')
edit('tests/unit/test_gdn_block_attention.py','''    assert ["block_attention" in layer for layer in layers] == [
        count == 4,
        True,
        True,
        True,
        False,
    ]
    assert layers[-1]["attention"]["representation"] == "gqa"
    assert layers[1]["block_attention"] == dict(
        num_heads=32, head_dim=32, gate_init=0.1
    )''','''    assert all("block_attention" not in layer for layer in layers)
    assert all(layer["gdn_config"]["intrinsic_block_attention"] for layer in layers[:4])
    assert layers[-1]["attention"]["representation"] == "gqa"
    assert layers[1]["gdn_config"]["gate_init"] == 0.1''')
# Cover actual KDA model wiring in FLA, including dense event layout and cached reads.
p=root/'tests/unit/test_attention_taxonomy.py'
t=p.read_text()+'''

@pytest.mark.parametrize("variant", ["gdn", "kda"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_linear_model_fla_forward_backward_and_cache(variant):
 torch.manual_seed(42);torch.set_num_threads(1)
 c=config(variant)
 p=c.composable_config['attention_configs'][variant]
 p.update(linear_key_head_dim=32,linear_value_head_dim=32)
 c.execution_config['gdn_backend']='fla'
 model=LegoSpecModel(c).cuda().to(torch.bfloat16)
 x=data();x.target_hidden_states=x.target_hidden_states.cuda().bfloat16();x.draft_embeddings=x.draft_embeddings.cuda().bfloat16()
 x.anchor_positions=x.anchor_positions.cuda();x.block_keep_mask=x.block_keep_mask.cuda()
 y=model(x).backbone_hidden_states
 y.float().square().mean().backward()
 assert torch.isfinite(y).all()
 assert torch.isfinite(model.backbone.layers[0].mixer.in_proj_a.weight.grad).all()
 assert torch.isfinite(model.backbone.layers[0].block_attention.gate_proj.weight.grad).all()
 model.eval();state=DraftState();context=x.target_hidden_states[:,:6];emb=x.draft_embeddings[:,:4]
 def run(start,end,state):
  return model(DraftInputs(context[:,start:end],emb,torch.tensor([[end]],device='cuda'),torch.ones(1,1,device='cuda',dtype=torch.bool),context_position_ids=torch.arange(start,end,device='cuda')[None],draft_state=state,mode='features')).backbone_hidden_states
 with torch.no_grad():
  run(0,3,state);cached=run(3,6,state);full=run(0,6,None)
 torch.testing.assert_close(cached,full,atol=.015,rtol=.05)


def test_mla_dsa_absorbed_cache_matches_expanded():
 torch.manual_seed(42);torch.set_num_threads(1)
 model=LegoSpecModel(config('mla','dsa')).eval();x=data()
 expanded=model(x).backbone_hidden_states
 with torch.no_grad():absorbed=model(x).backbone_hidden_states
 torch.testing.assert_close(absorbed,expanded,atol=2e-6,rtol=2e-5)
 state=DraftState();context=torch.randn(1,6,32);emb=torch.randn(1,4,16)
 def run(start,end,state):
  return model(DraftInputs(context[:,start:end],emb,torch.tensor([[end]]),torch.ones(1,1,dtype=torch.bool),context_position_ids=torch.arange(start,end)[None],draft_state=state,mode='features')).backbone_hidden_states
 with torch.no_grad():
  run(0,3,state);cached=run(3,6,state);full=run(0,6,None)
 torch.testing.assert_close(cached,full,atol=2e-6,rtol=2e-5)
'''
save('tests/unit/test_attention_taxonomy.py',t)
doc='''# Attention 配方

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
历史 string attention 配方继续按原结构读取，包括原先可选的 block 分支，避免悄悄增加权重。
旧 checkpoint、model_publish、训练快照和缓存未迁移；继续训练必须使用原始配置与完整训练状态。
新配方不允许把 block attention 单独配置。
'''
save('docs/ATTENTION_CONFIG.md',doc)
save('configs/README.md','# Model recipes\n\nSee [attention families and access](../docs/ATTENTION_CONFIG.md). Maintained JSON recipes use the family/variant/access interface. Historical checkpoint recipes must be loaded from their original snapshot.\n')
edit('README.md','- [Execution options](docs/training-execution.md)','- [Attention families and access](docs/ATTENTION_CONFIG.md)\n- [Execution options](docs/training-execution.md)')
edit('docs/COMPONENT_ORGANIZATION.md','GQA、MLA、GDN 数学归 `backbone/attention/`；卷积归 `backbone/convolution.py`。MHA/MQA 是 GQA 的头数配置。','Dense（MHA/GQA/MQA）、latent（MLA）、linear（GDN/KDA）数学归 `backbone/attention/`；卷积归 `backbone/convolution.py`。Dense/latent 的历史访问独立选择 full/swa/dsa，GDN/KDA 内置共享投影与逐头 gate block attention。详见 ATTENTION_CONFIG.md。')
edit('legospec/modeling/draft/composable/backbone/attention/DESIGN.md','gdn.py、gdn_events.py','gdn.py 同时实现 GDN 每头 decay 与 KDA 每 key 通道 decay，linear 固定共享投影与逐头 gate block attention；gdn.py、gdn_events.py')
for rel in ['AGENTS.md','.cursor/rules/h800-workspace.mdc']:
 p=root/rel;s=p.read_text();s+='\nThe main development repository is `/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public` on H800_2 (confirmed 2026-11-22).\n';save(rel,s)
(work/'changed_files.json').write_text(json.dumps(changes,indent=2))
print('Finalized tests, docs and main-project preference')
