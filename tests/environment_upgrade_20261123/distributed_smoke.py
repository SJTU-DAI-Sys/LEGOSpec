from pathlib import Path
import os,sys,json
import torch
import torch.distributed as dist
from legospec.distributed import init_distributed,init_disaggregated,destroy_distributed
from legospec.modeling.target import get_training_target_model
from legospec.modeling.config_utils import load_text_model_config
mode=sys.argv[1]
paths={
 "gemma_tp2":"/share/dai-sys/wanghanzhen/models/google/gemma-4-12B-it",
 "qwen35_tp2":"/mnt/dai-sys/wanghanzhen/projects/DAC27/models/hf/Qwen/Qwen3.6-35B-A3B/345d467b117d1fd5eda9b879a5988de9ae5f47a7",
 "qwen_disagg":"/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B",
}
topology=None
if mode=="qwen_disagg":
 topology=init_disaggregated(timeout=10,target_ranks_per_node=1,draft_ranks_per_node=1,target_tp_size=1)
else:
 init_distributed(tp_size=2)
try:
 if topology is None or topology.is_target:
  path=paths[mode];cfg=load_text_model_config(path);layers=[1,cfg.num_hidden_layers-1]
  target=get_training_target_model(path,backend="sglang",capture_layer_ids=layers,torch_dtype=torch.bfloat16,
    context_length=128,max_total_tokens=512,max_running_requests=8,mem_fraction_static=0.7,attention_backend="triton")
  ids=torch.tensor([[2,16,27,38],[2,49,50,0]],device="cuda")
  mask=torch.tensor([[1,1,1,1],[1,1,1,0]],device="cuda")
  out=target.generate_training_data(ids,mask,mask,return_logits=False)
  packed=target.generate_capture_data([dict(source_index=i,input_ids=ids[i,:int(mask[i].sum())]) for i in range(2)])
  for layer in layers:
   torch.testing.assert_close(out.hidden_states[layer][mask.bool()],packed.hidden_states[layer])
   assert torch.isfinite(out.hidden_states[layer]).all()
  head=target.build_shared_components(0).lm_head
  with torch.no_grad():
   logits=head(out.final_hidden_state[torch.arange(2,device="cuda"),mask.sum(-1)-1])
  torch.testing.assert_close(logits.float(),packed.last_token_logits.float(),rtol=0.02,atol=0.2)
  proj=torch.nn.Linear(cfg.hidden_size,8,device="cuda",dtype=torch.bfloat16)
  proj(out.hidden_states[layers[0]]).float().square().mean().backward()
  assert torch.isfinite(proj.weight.grad).all()
  if dist.get_rank()==0:
   result=dict(mode=mode,layers=layers,online_offline_equal=True,teacher_head_equal=True,backward=True)
   (Path(__file__).parent/(mode+"_result.json")).write_text(json.dumps(result,indent=2))
   print("SMOKE_OK",result,flush=True)
 dist.barrier()
finally:
 destroy_distributed()
