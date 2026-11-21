"""Packed execution must preserve the existing mathematical objective."""
from copy import deepcopy
import pytest
import torch
from test_vocab_reuse import wrapper, OBJECTIVES
from litespec.core.composable.batch import prepare_layout

@pytest.mark.parametrize('proposal', ['parallel', 'markov', 'rnn', 'sequential_rnn', 'residual_selector', 'topk_selector'])
@pytest.mark.parametrize('kind', OBJECTIVES)
def test_packed_matches_padded(kind, proposal):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    if proposal == 'topk_selector' and kind != 'hard_ce':
        pytest.skip('Existing candidate-only CE contract')
    legacy = wrapper(kind, proposal)
    packed = deepcopy(legacy)
    packed.optimized = True
    ids = torch.randint(31, (2, 24))
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[0, 19:] = False
    mask = valid.float()
    mask[0, :4] = 0
    mask[1, :12] = 0
    data = dict(input_ids=ids, loss_mask=mask, context_valid_mask=valid,
                target_hidden_states=torch.randn(2, 24, 32),
                target_last_hidden_states=torch.randn(2, 24, 16),
                layout=prepare_layout(ids, mask, 8, 16, complete_blocks=True),
                collect_detailed_metrics=True)
    expected, actual = legacy(**data), packed(**data)
    torch.testing.assert_close(actual.loss_terms.numerator, expected.loss_terms.numerator, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual.loss_terms.denominator, expected.loss_terms.denominator)
    expected.loss_terms.numerator.backward()
    actual.loss_terms.numerator.backward()
    for (name, p), (other, q) in zip(legacy.named_parameters(), packed.named_parameters()):
        assert name == other
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=2e-5, msg=name)

@pytest.mark.parametrize('context', ['dflash', 'dflare'])
@pytest.mark.parametrize('attention,access', [('gqa','dense'), ('mha','sliding_window'), ('mla','dense'), ('mla','sliding_window'), ('gqa','dsa'), ('gdn','linear')])
@pytest.mark.parametrize('conv', [False, True])
def test_shared_backbones(context, attention, access, conv):
    from torch import nn
    from litespec.modeling.draft.composable import ComposableDraftConfig, ComposableDraftModel
    from litespec.core.composable.training_model import ComposableTrainingModel
    torch.manual_seed(17); torch.set_num_threads(1)
    profiles = {'gqa': {}, 'mha': {'num_key_value_heads': 2},
        'mla': dict(num_attention_heads=2,num_key_value_heads=2,q_lora_rank=4,kv_lora_rank=4,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=8),
        'gdn':dict(linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=4,linear_value_head_dim=4)}
    layer=dict(attention=attention, access=access)
    if conv: layer['local_operator']=dict(type='dflash2_conv',kernel_size=2,group_size=4)
    recipe=dict(context=dict(type=context,target_layer_ids=[0,1]),
        backbone=dict(type='layer_stack',layers=[layer]), attention_configs={attention:profiles[attention]},
        access_configs={access: dict(window_size=8) if access=='sliding_window' else
                        dict(num_query_heads=2,head_dim=8,rope_head_dim=4,index_topk=4,stage='sparse') if access=='dsa' else {}},
        proposal=dict(type='rnn',rank=4),objective=dict(type='soft_ce',loss_decay_gamma=4.))
    cfg=ComposableDraftConfig(composable_config=recipe,hidden_size=16,intermediate_size=32,
        num_target_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        vocab_size=31,block_size=4,execution_config=dict(attention_backend='sdpa',gdn_backend='reference'))
    legacy=ComposableTrainingModel(ComposableDraftModel(cfg),nn.Linear(16,31,bias=False).requires_grad_(False),
        nn.Embedding(31,16).requires_grad_(False),4,objective_chunk_blocks=2)
    # Exercise non-identity convolution and nonzero recurrent vocabulary outputs.
    with torch.no_grad():
        for n,p in legacy.named_parameters():
            if 'kernel_projection' in n or 'markov_w2' in n: p.normal_(std=.03)
    packed=deepcopy(legacy); packed.optimized=True
    ids=torch.randint(31,(2,16)); valid=torch.ones_like(ids,dtype=torch.bool);valid[0,12:]=False
    mask=valid.float();mask[0,:3]=0;mask[1,:7]=0
    data=dict(input_ids=ids,loss_mask=mask,context_valid_mask=valid,
        target_hidden_states=torch.randn(2,16,32),target_last_hidden_states=torch.randn(2,16,16),
        layout=prepare_layout(ids,mask,4,12,complete_blocks=True))
    a,b=legacy(**data),packed(**data)
    torch.testing.assert_close(a.loss,b.loss,rtol=2e-5,atol=2e-6)
    a.loss.backward();b.loss.backward()
    for (n,p),(_,q) in zip(legacy.named_parameters(),packed.named_parameters()):
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=2e-5,msg=n)

@pytest.mark.parametrize('mode', ['sampled','all'])
@pytest.mark.parametrize('chunk', [0,2])
def test_eagle3_packing(mode,chunk):
    from test_composable_eagle3 import config, components, batch, wrapper as eagle_wrapper
    from litespec.modeling.draft.composable import ComposableDraftModel
    from litespec.core.composable.anchor_sampling import prepare_sample_epoch_layout
    torch.manual_seed(11);torch.set_num_threads(1)
    cfg=config();shared=components(cfg)
    legacy=eagle_wrapper(ComposableDraftModel(cfg),shared,mode=mode,chunk=chunk,anchors=12)
    packed=deepcopy(legacy);packed.optimized=True
    data=batch(cfg,count=2,length=14)
    data['context_valid_mask'][0,10:]=False;data['loss_mask'][0,10:]=0
    data['layout']=prepare_sample_epoch_layout(data['input_ids'],data['loss_mask'],data['context_valid_mask'],
        block_size=4,num_anchors=12,seed=42,epoch=0,supervision_mode=mode,minimum_anchor=1)
    a,b=legacy(**data),packed(**data)
    torch.testing.assert_close(a.loss,b.loss,rtol=2e-5,atol=2e-6)
    a.loss.backward();b.loss.backward()
    for (n,p),(_,q) in zip(legacy.named_parameters(),packed.named_parameters()):
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=2e-5,msg=n)

@pytest.mark.parametrize('proposal',['parallel','rnn'])
def test_sampled_opd_replay_packs_backbone(proposal):
    from test_anchor_opd import fixture
    legacy,scorer,data=fixture(proposal)
    with torch.no_grad(): record=legacy(**data,opd_scorer=scorer,metadata_only=True)['sampled_opd_batch']
    packed=deepcopy(legacy);packed.optimized=True
    a,b=legacy(**data,sampled_opd_batch=record),packed(**data,sampled_opd_batch=record)
    torch.testing.assert_close(a.loss,b.loss,rtol=2e-5,atol=2e-6)
    a.loss.backward();b.loss.backward()
    for (name,p),(_,q) in zip(legacy.named_parameters(),packed.named_parameters()):
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=2e-5,msg=name)

@pytest.mark.parametrize('proposal',['parallel','rnn','sequential_rnn','residual_selector'])
@pytest.mark.parametrize('cpu_plan',[False,True])
def test_empty_rank_still_has_gradients(proposal,cpu_plan):
    from litespec.core.composable.packing import prepare_packed_plan
    model=wrapper('dpal_renyi',proposal);model.optimized=True
    ids=torch.randint(31,(2,24));mask=torch.zeros_like(ids,dtype=torch.float)
    layout=prepare_layout(ids,mask,8,8,complete_blocks=True)
    valid=torch.ones_like(ids,dtype=torch.bool)
    data=dict(input_ids=ids,loss_mask=mask,context_valid_mask=valid,
        target_hidden_states=torch.randn(2,24,32),target_last_hidden_states=torch.randn(2,24,16),layout=layout)
    if cpu_plan: data['execution_plan']=prepare_packed_plan(valid,layout,8)
    output=model(**data);output.loss.backward()
    assert output.loss==0
    for name,p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None,name
            assert torch.isfinite(p.grad).all() and not p.grad.count_nonzero(),name

@pytest.mark.parametrize('proposal',['parallel','rnn','residual_selector'])
@pytest.mark.parametrize('chunk',[0,2])
def test_sequence_balanced_dpal_keeps_sample_owners(proposal,chunk):
    from litespec.core.composable.objectives.selector import LossBranches
    from litespec.core.composable.packing import prepare_packed_plan
    legacy=wrapper('dpal_renyi',proposal);legacy.objective_chunk_blocks=chunk
    recipe=deepcopy(legacy.draft_model.config.composable_config)
    objective=recipe['objective']['final'] if 'selector' in proposal else recipe['objective']
    objective['reduction']='sequence_anchor_mean'
    legacy.loss_branches=LossBranches(recipe)
    packed=deepcopy(legacy);packed.optimized=True
    ids=torch.randint(31,(3,24));mask=torch.ones_like(ids,dtype=torch.float)
    mask[0,12:]=0;mask[1,10:14]=0;mask[2]=0
    layout=prepare_layout(ids,mask,8,12,complete_blocks=False)
    valid=torch.ones_like(ids,dtype=torch.bool)
    data=dict(input_ids=ids,loss_mask=mask,context_valid_mask=valid,
        target_hidden_states=torch.randn(3,24,32),target_last_hidden_states=torch.randn(3,24,16),layout=layout)
    a=legacy(**data);b=packed(**data,execution_plan=prepare_packed_plan(valid,layout,8))
    torch.testing.assert_close(a.loss,b.loss,rtol=2e-5,atol=2e-6)
    a.loss.backward();b.loss.backward()
    for (name,p),(_,q) in zip(legacy.named_parameters(),packed.named_parameters()):
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=2e-5,msg=name)
