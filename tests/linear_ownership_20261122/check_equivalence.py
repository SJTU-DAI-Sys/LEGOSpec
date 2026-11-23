"""Exact before/after outputs, gradients, weights, optimizer ownership and cache."""
from pathlib import Path
import sys,dataclasses,json
import torch
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root),str(root/'tests/unit')]
from test_attention_taxonomy import config,data
from test_gdn_shared_block_attention import config as legacy_shared
from test_gdn_block_attention import config as legacy_separate
from legospec import LegoSpecModel
from legospec.training.optimizer import register_optimizer_parameters
from legospec.modeling.draft.composable.contracts import DraftInputs,DraftState

def flatten(value,prefix=''):
 if isinstance(value,torch.Tensor):return {prefix:value.detach().cpu()}
 if dataclasses.is_dataclass(value):value={f.name:getattr(value,f.name) for f in dataclasses.fields(value)}
 if isinstance(value,dict):return {k:v for name,item in value.items() for k,v in flatten(item,prefix+'/'+str(name)).items()}
 return {}

def capture():
 torch.set_num_threads(1);records={};specifications={}
 cases=[(variant+'/'+access,config(variant,access)) for variant in ['mha','gqa','mqa','mla','gdn','kda'] for access in (['full'] if variant in ['gdn','kda'] else ['full','swa','dsa'])]
 for enabled in [False,True]:
  for conv in [False,True]:
   for mode in ['block','confirmed_history']:
    cases.append((f'legacy_shared/{enabled}/{conv}/{mode}',legacy_shared(enabled=enabled,conv=conv,q_mode=mode)))
   cases.append((f'legacy_separate/{enabled}/{conv}',legacy_separate(enabled=enabled,conv=conv)))
 for name,c in cases:
  torch.manual_seed(42);m=LegoSpecModel(c);x=data()
  # Use nontrivial external convolution weights so fusion ordering is observable.
  with torch.no_grad():
   for layer in m.backbone.layers:
    if layer.attention_conv is not None:layer.attention_conv.kernel_projection.weight.normal_(std=.03)
  y=m(x).backbone_hidden_states;y.square().mean().backward()
  record={'weights':m.state_dict(),'output':y,'gradients':{n:p.grad for n,p in m.named_parameters() if p.grad is not None}}
  _,specs=register_optimizer_parameters(m)
  specifications[name]=[dataclasses.asdict(s) for s in specs]
  m.eval();state=DraftState();context=torch.randn(1,6,32);emb=torch.randn(1,4,16)
  def run(start,end,state):
   return m(DraftInputs(context[:,start:end],emb,torch.tensor([[end]]),torch.ones(1,1,dtype=torch.bool),context_position_ids=torch.arange(start,end)[None],draft_state=state,mode='features')).backbone_hidden_states
  with torch.no_grad():
   record['eval']=m(x).backbone_hidden_states
   run(0,3,state);record['cached']=run(3,6,state);record['full']=run(0,6,None)
  record['cache']=state;records[name]=flatten(record)
 return records,specifications

work=Path(__file__).parent;actual,specifications=capture()
if sys.argv[1]=='before':
 torch.save(actual,work/'before.pt');(work/'optimizer_before.json').write_text(json.dumps(specifications))
 print('Saved',len(actual),'configuration signatures',flush=True)
else:
 expected=torch.load(work/'before.pt',weights_only=True);assert expected.keys()==actual.keys();count=0
 assert json.loads(json.dumps(specifications))==json.loads((work/'optimizer_before.json').read_text())
 for case,record in actual.items():
  assert record.keys()==expected[case].keys(),case
  for key,value in record.items():
   torch.testing.assert_close(value,expected[case][key],rtol=0,atol=0,msg=case+key);count+=1
 print('BITWISE EQUIVALENT:',len(actual),'configurations,',count,'tensors; optimizer names/order/geometry identical',flush=True)
