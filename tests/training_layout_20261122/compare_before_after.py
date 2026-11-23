"""Compare relocated execution and cross-layout checkpoint resume to the frozen pre-change sources."""
from pathlib import Path
import copy, json, sys, tarfile, types
import torch

R=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tests/unit'))
torch.set_num_threads(1)
from test_vocab_reuse import wrapper
from legospec.training.data.batch import prepare_layout
from legospec.training.optim.optimizer import BF16Optimizer
from legospec.training.runtime.checkpoint import load_distributed_training_state, restore_rng_state

with tarfile.open(W/'before.tgz') as tar:
    def source(name):return tar.extractfile('legospec/training/'+name+'.py').read().decode()
    step_source=source('step');optimizer_source=source('optimizer');checkpoint_source=source('checkpoint')
# Only imports are rebound to their new locations; the reference class/function bodies are unchanged.
paths={'anchor_opd':'execution.sampled','batch':'data.batch','metrics':'logger.metrics','outputs':'outputs','supervision':'execution.supervision','fsdp_layout':'optim.fsdp_layout','muon':'optim.muon'}
for old,new in paths.items():
    step_source=step_source.replace('from legospec.training.'+old+' import','from legospec.training.'+new+' import')
    optimizer_source=optimizer_source.replace('from legospec.training.'+old+' import','from legospec.training.'+new+' import')
step_source=step_source.replace('from legospec.training.component_reduction import combine_component_terms\n','')
def module(name,code):
    m=types.ModuleType(name);sys.modules[name]=m;exec(compile(code,name,'exec'),m.__dict__);return m
old_step=module('training_before_step',step_source).TrainingStep
old_optimizer=module('training_before_optimizer',optimizer_source).BF16Optimizer
old_checkpoint=module('training_before_checkpoint',checkpoint_source)

def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)

cases=[]
for proposal in ('parallel','rnn','residual_selector'):
 for kind in ('soft_ce','weighted_sum','dpal_renyi','e2e_tv'):
  for mode in ('legacy','optimized'):
    torch.manual_seed(42)
    current=wrapper(kind,proposal);current.optimized=mode=='optimized'
    before=old_step(copy.deepcopy(current.draft_model),copy.deepcopy(current.lm_head),copy.deepcopy(current.embed_tokens),4,objective_chunk_blocks=2,training_mode=mode)
    assert list(dict(current.named_parameters()))==list(dict(before.named_parameters()))
    ids=torch.randint(31,(2,24));mask=torch.ones_like(ids,dtype=torch.float32);mask[0,17:]=0
    layout=prepare_layout(ids,mask,8,8,complete_blocks=True)
    data=dict(input_ids=ids,loss_mask=mask,target_hidden_states=torch.randn(2,24,32),target_last_hidden_states=torch.randn(2,24,16),layout=layout,context_valid_mask=torch.ones_like(ids,dtype=torch.bool),collect_detailed_metrics=True)
    a,b=before(**data),current(**data);equal(a.loss,b.loss)
    equal(a.loss_terms.numerator,b.loss_terms.numerator);equal(a.loss_terms.denominator,b.loss_terms.denominator)
    assert a.ratio_metrics.keys()==b.ratio_metrics.keys()
    for k in a.ratio_metrics:
        equal(a.ratio_metrics[k].numerator,b.ratio_metrics[k].numerator);equal(a.ratio_metrics[k].denominator,b.ratio_metrics[k].denominator)
    a.loss.backward();b.loss.backward()
    for (name,p),(_,q) in zip(before.named_parameters(),current.named_parameters()):equal(p.grad,q.grad)
    cases.append(dict(proposal=proposal,objective=kind,mode=mode))

torch.manual_seed(42)
model=torch.nn.Linear(4,3).to(torch.bfloat16)
opt=old_optimizer(model,lr=0.01,total_steps=10,warmup_ratio=0.2)
def update(model,opt):
    model(torch.randn(5,4).bfloat16()).float().square().mean().backward();opt.step()
update(model,opt)
cursor=dict(epoch=2,next_batch=7,phase_step=7,global_step=7,optimizer_step=1)
folder=W/'cross_layout_checkpoint'
old_checkpoint.save_distributed_training_state(str(folder),{**opt.state_dict(),'cursor':cursor,'rng_state':old_checkpoint.capture_rng_state()})
update(model,opt);expected=copy.deepcopy(opt.state_dict())
resumed=torch.nn.Linear(4,3).to(torch.bfloat16);newopt=BF16Optimizer(resumed,lr=0.01,total_steps=10,warmup_ratio=0.2)
state=load_distributed_training_state(str(folder));equal(state['cursor'],cursor)
newopt.load_state_dict(state);restore_rng_state(state['rng_state']);update(resumed,newopt)
equal(expected,newopt.state_dict());equal(model.state_dict(),resumed.state_dict())
report=dict(exact_forward_gradient_metric_cases=cases,cross_layout_resume='exact',checked_resume_fields=['optimizer','scheduler','fp32_master_params','rng','cursor','model_parameters'])
(W/'equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'{len(cases)} forward/gradient/metric cases bitwise equal; pre-change checkpoint resumes to bitwise-equal next update.')
