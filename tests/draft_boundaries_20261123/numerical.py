from pathlib import Path
import sys,importlib,torch,json
base=Path(__file__).resolve().parent
old=sys.argv[1]=="before"
if old:sys.path.insert(0,str(base/"before"))
else:sys.path.insert(0,str(base.parents[1]))
mod=importlib.import_module("legospec.modeling.draft"+(".composable" if old else ""))
contracts=importlib.import_module(mod.__name__+".contracts")
torch.set_num_threads(1)
results={}
for variant in ["gqa","mla","gdn","kda"]:
 torch.manual_seed(89)
 family="dense" if variant=="gqa" else "latent" if variant=="mla" else "linear"
 att={"family":family,"variant":variant}
 if family!="linear":att["access"]="full"
 profile={}
 if variant=="mla":profile=dict(num_attention_heads=2,num_key_value_heads=2,q_lora_rank=8,kv_lora_rank=8,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4)
 if family=="linear":profile=dict(linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=4,linear_value_head_dim=4,query_conv_kernel_size=2,context_conv_kernel_size=2)
 cfg=mod.LegoSpecConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_target_layers=4,
 vocab_size=31,num_attention_heads=2,num_key_value_heads=1,head_dim=8,block_size=4,
 execution_config={"gdn_backend":"reference"},composable_config={
 "context":{"type":"concat_projection","target_layer_ids":[0,2]},
 "backbone":{"type":"layer_stack","layers":[{"attention":att} for _ in range(2)]},
 "proposal":{"type":"parallel"},"objective":{"type":"hard_ce"},"attention_configs":{variant:profile}})
 model=mod.LegoSpecModel(cfg)
 path=base/(variant+"_baseline.pt")
 def key(name):
  if family=="linear":
   for part in ["position_embedding","mixer","block_attention"]:
    name=name.replace("."+part+".",".attention."+part+".")
  else:name=name.replace(".self_attn.",".attention.")
  return name
 if not old:
  expected=torch.load(path,weights_only=False)
  model.load_state_dict(expected["weights"])
  values=expected["inputs"]
 else:
  values=dict(target_hidden_states=torch.randn(1,7,32),draft_embeddings=torch.randn(1,8,16),
   anchor_positions=torch.tensor([[3,6]]),block_keep_mask=torch.ones(1,2,dtype=torch.bool),mode="features")
 output=model(contracts.DraftInputs(**values)).draft_hidden_states
 output.square().mean().backward()
 weights={key(k) if old else k:v.detach().clone() for k,v in model.state_dict().items()}
 gradients={key(k) if old else k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
 if old:torch.save(dict(weights=weights,inputs=values,output=output.detach(),gradients=gradients),path)
 else:
  torch.testing.assert_close(output,expected["output"],rtol=0,atol=0)
  assert gradients.keys()==expected["gradients"].keys()
  for k,v in gradients.items():torch.testing.assert_close(v,expected["gradients"][k],rtol=0,atol=0,msg=k)
  results[variant]={"forward":"bitwise_equal","gradients":"bitwise_equal"}
if not old:(base/"numerical.json").write_text(json.dumps(results,indent=2))
print(results or "baseline_saved")
