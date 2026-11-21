"""Layer-specific SWA windows must survive export and retain DFlash visibility."""
from types import SimpleNamespace
import pytest
import torch
from litespec.modeling.draft.composable import ComposableDraftConfig, ComposableDraftModel
from litespec.modeling.draft.composable.backbone.attention.access.base import build_visibility


def config(window=1, access='sliding_window'):
    layers=[dict(attention='gqa',access='sliding_window') for _ in range(4)]
    layers[0].update(access=access,window_size=window)
    layers.append(dict(attention='gqa',access='dense'))
    return ComposableDraftConfig(hidden_size=16,intermediate_size=32,vocab_size=31,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,num_target_layers=4,
        composable_config=dict(context=dict(type='dflash',target_layer_ids=[0,1]),
            backbone=dict(type='layer_stack',layers=layers),attention_configs={'gqa':{}},
            access_configs={'sliding_window':{'window_size':512},'dense':{}},
            proposal=dict(type='rnn',rank=4),objective=dict(type='hard_ce')))


def test_layer_windows_export_and_runtime(tmp_path):
    cfg=config();cfg.save_pretrained(tmp_path)
    restored=ComposableDraftConfig.from_pretrained(tmp_path)
    model=ComposableDraftModel(restored)
    assert [m.window_size for m in model.modules() if hasattr(m,'window_size')] == [1,512,512,512,None]
    assert restored.composable_config['access_configs']['sliding_window']['window_size']==512
    assert restored.composable_config['backbone']['layers'][0]['window_size']==1


@pytest.mark.parametrize('window',[0,-1,True,1.5])
def test_invalid_window(window):
    with pytest.raises(ValueError,match='window_size'):config(window)


def test_dense_layer_cannot_silently_ignore_override():
    with pytest.raises(ValueError,match='requires sliding_window'):config(1,'dense')


def test_window_one_keeps_latest_history_and_whole_current_block():
    inputs=SimpleNamespace(anchor_positions=torch.tensor([[5]]),block_keep_mask=torch.tensor([[True]]),anchor_document_ids=None)
    positions=torch.arange(8)[None];valid=torch.ones_like(positions,dtype=torch.bool)
    mask=build_visibility(inputs,positions,valid,torch.zeros_like(positions),4,window_size=1)[0,0]
    assert mask[:,:8].nonzero().tolist()==[[0,4],[1,4],[2,4],[3,4]]
    assert mask[:,8:].all()
