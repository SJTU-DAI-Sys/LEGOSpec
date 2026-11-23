from pathlib import Path
import os, fcntl, json, sys
root=Path(__file__).resolve().parents[2]
locks=[]
for name in ["LEGOSpec_20261111","LEGOSpec_swa_20261112","LEGOSpec_public"]:
 f=(root.parent/name/"tests/h8002_eval_dispatch.lock").open("a")
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
os.environ.update(MASTER_ADDR="127.0.0.1",MASTER_PORT="29781",RANK="0",WORLD_SIZE="1",LOCAL_RANK="0")
import torch
from legospec.distributed import init_distributed,destroy_distributed
from legospec.modeling.target.training_target_model import get_training_target_model
from legospec.modeling.config_utils import load_text_model_config
kind=sys.argv[1]
path="/share/dai-sys/wanghanzhen/models/"+("google/gemma-4-12B-it" if kind=="gemma" else "Qwen/Qwen3-4B")
cfg=load_text_model_config(path);last=cfg.num_hidden_layers-1
init_distributed()
try:
 target=get_training_target_model(path,backend="sglang",capture_layer_ids=[1,last],
  torch_dtype=torch.bfloat16,context_length=128,max_running_requests=8,
  max_total_tokens=512,mem_fraction_static=0.6,attention_backend="triton")
 ids=torch.tensor([[2,15,26,37],[2,48,59,0]],device="cuda")
 mask=torch.tensor([[1,1,1,1],[1,1,1,0]],device="cuda")
 online=target.generate_training_data(ids,mask,mask,return_logits=False)
 offline=target.generate_capture_data([dict(source_index=i,input_ids=ids[i,:int(mask[i].sum())]) for i in range(2)])
 for layer in [1,last]:
  torch.testing.assert_close(online.hidden_states[layer][mask.bool()],offline.hidden_states[layer])
  assert torch.isfinite(online.hidden_states[layer]).all()
 assert not torch.is_inference(online.final_hidden_state)
 assert not torch.equal(online.hidden_states[last],online.final_hidden_state)
 torch.testing.assert_close(online.final_hidden_state[mask.bool()],offline.final_hidden_state)
 head=target.build_shared_components(0).lm_head
 with torch.no_grad():
  logits=head(online.final_hidden_state[torch.arange(2,device="cuda"),mask.sum(-1)-1])
 torch.testing.assert_close(logits.float(),offline.last_token_logits.float(),rtol=0.02,atol=0.2)
 proj=torch.nn.Linear(cfg.hidden_size,8,device="cuda",dtype=torch.bfloat16)
 proj(online.hidden_states[1]).float().square().mean().backward()
 assert torch.isfinite(proj.weight.grad).all()
 target.set_capture_layers([last])
 final_only=target.generate_training_data(ids,mask,mask,return_logits=False)
 torch.testing.assert_close(final_only.hidden_states[last],online.hidden_states[last])
 if kind=="qwen":
  scorer=target.build_anchor_scorer()
  anchors=torch.tensor([[0,2],[0,1]],device="cuda")
  keep=torch.ones_like(anchors,dtype=torch.bool)
  prefix=scorer.prefill_anchors(ids,mask.bool(),anchors,keep,capture_layer_ids=[1,last])
  torch.testing.assert_close(prefix.full_hidden[mask.bool()],online.final_hidden_state[mask.bool()],rtol=0.02,atol=0.02)
  for layer in [1,last]:
   torch.testing.assert_close(prefix.context_hidden[layer][mask.bool()],online.hidden_states[layer][mask.bool()],rtol=0.02,atol=0.1)
  sampled=torch.tensor([[[60,61,62],[63,64,65]],[[66,67,68],[69,70,71]]],device="cuda")
  branches=scorer.score_sampled_blocks(prefix,sampled).clone()
  prefix.release()
  target.set_capture_layers([last])
  for row in range(2):
   for col in range(2):
    length=int(anchors[row,col])+1
    branch=torch.cat([ids[row,:length],sampled[row,col,:-1]])[None]
    reference=target.generate_training_data(branch,torch.ones_like(branch),torch.ones_like(branch),return_logits=False)
    actual=branches[row,col].float()
    expected=reference.final_hidden_state[0,length-1:].float()
    relative=(actual-expected).norm()/expected.norm()
    cosine=torch.nn.functional.cosine_similarity(actual,expected,dim=-1).min()
    print("branch_error",row,col,"relative_l2",float(relative),"min_cosine",float(cosine),flush=True)
    assert relative<0.01 and cosine>0.9999
 result=dict(model=kind,layers=[1,last],tokens=int(mask.sum()),online_offline_equal=True,final_only_equal=True,teacher_head_equal=True,backward=True)
 (Path(__file__).parent/(kind+"_result.json")).write_text(json.dumps(result,indent=2))
 print("SMOKE_OK",json.dumps(result),flush=True)
finally:
 destroy_distributed()
