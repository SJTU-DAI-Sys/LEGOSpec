"""Compare same-seed weights, outputs, gradients and confirmed cache across moves."""
from pathlib import Path
import sys,dataclasses
import torch
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root),str(root/'tests/unit')]
from test_attention_taxonomy import config,data
from legospec import LegoSpecModel
from legospec.modeling.draft.composable.contracts import DraftInputs,DraftState

def flatten(value,prefix=''):
 if isinstance(value,torch.Tensor):return {prefix:value.detach().cpu()}
 if dataclasses.is_dataclass(value):value={f.name:getattr(value,f.name) for f in dataclasses.fields(value)}
 if isinstance(value,dict):return {k:v for name,item in value.items() for k,v in flatten(item,prefix+'/'+str(name)).items()}
 return {}

def capture():
 torch.set_num_threads(1);records={}
 for variant in ['mha','gqa','mqa','mla','gdn','kda']:
  for access in (['full'] if variant in ['gdn','kda'] else ['full','swa','dsa']):
   torch.manual_seed(42);m=LegoSpecModel(config(variant,access));x=data()
   y=m(x).backbone_hidden_states;y.square().mean().backward()
   record={'weights':m.state_dict(),'output':y,'gradients':{n:p.grad for n,p in m.named_parameters() if p.grad is not None}}
   m.eval();state=DraftState();context=torch.randn(1,6,32);emb=torch.randn(1,4,16)
   def run(start,end,state):
    return m(DraftInputs(context[:,start:end],emb,torch.tensor([[end]]),torch.ones(1,1,dtype=torch.bool),context_position_ids=torch.arange(start,end)[None],draft_state=state,mode='features')).backbone_hidden_states
   with torch.no_grad():
    record['eval']=m(x).backbone_hidden_states
    run(0,3,state);record['cached']=run(3,6,state);record['full']=run(0,6,None)
   record['cache']=state
   records[variant+'/'+access]=flatten(record)
 return records

path=Path(__file__).parent/'before.pt'
actual=capture()
if sys.argv[1]=='before':
 torch.save(actual,path);print('Saved',len(actual),'variant/access signatures',flush=True)
else:
 expected=torch.load(path,weights_only=True)
 assert expected.keys()==actual.keys()
 count=0
 for case,record in actual.items():
  assert record.keys()==expected[case].keys(),case
  for key,value in record.items():
   torch.testing.assert_close(value,expected[case][key],rtol=0,atol=0,msg=case+key);count+=1
 print('BITWISE EQUIVALENT:',len(actual),'variant/access combinations,',count,'tensors',flush=True)
