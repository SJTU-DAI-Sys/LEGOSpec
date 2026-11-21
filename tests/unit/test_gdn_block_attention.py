"""GDN/block fusion: head gates, branch isolation, confirmed cache and export."""
from copy import deepcopy
from dataclasses import replace
import math
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModel

from litespec.modeling.draft.composable import ComposableDraftConfig, ComposableDraftModel
from litespec.modeling.draft.composable.backbone.attention.block import GatedBlockAttention
from litespec.modeling.draft.composable.contracts import DraftState, ModelInputs
from litespec.modeling.draft.composable.export import export_composable


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def config(enabled=True, conv=False, num_layers=3):
    layers = [dict(attention="gdn", access="linear") for _ in range(num_layers)]
    if enabled:
        for layer in layers:
            layer["block_attention"] = dict(num_heads=2, head_dim=4, gate_init=0.1)
    if conv:
        for layer in layers:
            layer["local_operator"] = dict(type="dflash2_conv", kernel_size=2, group_size=4)
    return ComposableDraftConfig(hidden_size=16, intermediate_size=32, vocab_size=31,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, num_target_layers=4,
        block_size=4, execution_config={"gdn_backend": "reference"},
        composable_config=dict(context=dict(type="dflash", target_layer_ids=[0, 2]),
            backbone=dict(type="layer_stack", layers=layers),
            attention_configs=dict(gdn=dict(linear_num_key_heads=1, linear_num_value_heads=2,
                linear_key_head_dim=4, linear_value_head_dim=4,
                query_conv_kernel_size=2, context_conv_kernel_size=2)),
            proposal=dict(type="parallel"), objective=dict(type="hard_ce")))


def inputs():
    return ModelInputs(target_hidden_states=torch.randn(1, 7, 32),
        draft_embeddings=torch.randn(1, 8, 16), anchor_positions=torch.tensor([[3, 6]]),
        block_keep_mask=torch.ones(1, 2, dtype=torch.bool), mode="features")


def test_all_gdn_layers_have_block_attention_and_legacy_recipe_has_no_block_parameters():
    model = ComposableDraftModel(config())
    assert [layer.block_attention is not None for layer in model.backbone.layers] == [True, True, True]
    assert any("block_attention" in name for name in model.backbone.layers[0].state_dict())
    legacy = ComposableDraftModel(config(enabled=False))
    assert not any("block_attention" in name for name in legacy.state_dict())
    for layer in model.backbone.layers:
        gate = layer.block_attention.gate_proj
        torch.testing.assert_close(gate.weight, torch.zeros_like(gate.weight))
        torch.testing.assert_close(gate.bias.sigmoid(), torch.full_like(gate.bias, 0.1))


@pytest.mark.parametrize("mutation", ["first_gqa", "gqa", "heads", "dim", "gate0", "gate1", "gatebool", "nan", "unknown", "null"])
def test_invalid_config(mutation):
    c = config()
    layers = c.composable_config["backbone"]["layers"]
    branch = layers[1]["block_attention"]
    if mutation == "first_gqa": layers[0].update(attention="gqa", access="dense")
    elif mutation == "gqa": layers[1].update(attention="gqa", access="dense")
    elif mutation == "heads": branch["num_heads"] = 0
    elif mutation == "dim": branch["head_dim"] = True
    elif mutation == "gate0": branch["gate_init"] = 0
    elif mutation == "gate1": branch["gate_init"] = 1
    elif mutation == "gatebool": branch["gate_init"] = True
    elif mutation == "nan": branch["gate_init"] = float("nan")
    elif mutation == "unknown": branch["typo"] = 2
    elif mutation == "null": layers[1]["block_attention"] = None
    with pytest.raises(ValueError, match="block_attention"):
        c.validate_recipe()


def test_branch_matches_explicit_attention_with_dynamic_per_token_head_gates():
    branch = GatedBlockAttention(8, 4, dict(num_heads=2, head_dim=3, gate_init=0.1))
    with torch.no_grad():
        branch.gate_proj.weight.normal_()
        branch.gate_proj.bias.copy_(torch.tensor([-1., 1.]))
    hidden = torch.randn(2, 8, 8, requires_grad=True)
    expected = []
    for sample in hidden:
        rows = []
        for block in sample.split(4):
            q, k, v = branch.qkv_proj(block).reshape(4, 3, 2, 3).unbind(1)
            heads = []
            for h in range(2):
                weights = (q[:, h] @ k[:, h].T / math.sqrt(3)).softmax(-1)
                gates = (block @ branch.gate_proj.weight[h] + branch.gate_proj.bias[h]).sigmoid()
                heads.append((weights @ v[:, h]) * gates[:, None])
            rows.append(branch.out_proj(torch.cat(heads, -1)))
        expected.append(torch.cat(rows))
    actual = branch(hidden)
    torch.testing.assert_close(actual, torch.stack(expected))
    actual.square().sum().backward()
    for parameter in branch.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_attention_is_bidirectional_and_cannot_cross_blocks():
    branch = GatedBlockAttention(8, 4, dict(num_heads=2, head_dim=4, gate_init=0.1))
    hidden = torch.randn(1, 8, 8, requires_grad=True)
    grad = torch.autograd.grad(branch(hidden)[0, 0].square().sum(), hidden)[0]
    assert grad[0, 3].abs().sum() > 0  # last slot affects the first slot
    assert torch.count_nonzero(grad[:, 4:]) == 0


@pytest.mark.parametrize("conv", [False, True])
@pytest.mark.parametrize("layer_index", [0, 1])
def test_parallel_branch_sees_pre_read_pre_convolution_input(conv, layer_index):
    model = ComposableDraftModel(config(conv=conv)).eval()
    layer = model.backbone.layers[layer_index]
    data, seen = inputs(), []
    norm_hook = layer.input_norm.register_forward_hook(lambda m, a, out: seen.append(out.detach().clone()))
    branch_hook = layer.block_attention.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach().clone()))
    model(data)
    norm_hook.remove(); branch_hook.remove()
    assert len(seen) == 2
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)


def test_zero_block_output_recovers_original_gdn_and_history_is_not_polluted():
    legacy = ComposableDraftModel(config(enabled=False)).eval()
    model = ComposableDraftModel(config()).eval()
    missing, unexpected = model.load_state_dict(legacy.state_dict(), strict=False)
    assert missing and all("block_attention" in key for key in missing) and not unexpected
    for layer in model.backbone.layers:
        torch.nn.init.zeros_(layer.block_attention.out_proj.weight)
    data = inputs()
    torch.testing.assert_close(model(data).backbone_hidden_states, legacy(data).backbone_hidden_states,
                               rtol=0, atol=0)


@pytest.mark.parametrize("conv,num_layers", [(False, 3), (True, 3), (False, 5)])
def test_cache_matches_full_prefix_and_stores_only_confirmed_history(conv, num_layers):
    model = ComposableDraftModel(config(conv=conv, num_layers=num_layers)).eval()
    if not conv:
        assert all(layer.attention_conv is None and layer.mlp_conv is None
                   for layer in model.backbone.layers)
    data, state = inputs(), DraftState()
    for anchor in (3, 6):
        start = 0 if state.backbone_state is None else int(state.backbone_state.next_positions[0])
        block = torch.randn(1, 4, 16)
        full = replace(data, draft_embeddings=block, anchor_positions=torch.tensor([[anchor]]),
                       block_keep_mask=torch.ones(1, 1, dtype=torch.bool))
        cached = replace(full, target_hidden_states=data.target_hidden_states[:, start:anchor],
                         context_position_ids=torch.arange(start, anchor)[None], draft_state=state)
        actual = model(cached).backbone_hidden_states
        torch.testing.assert_close(actual, model(full).backbone_hidden_states, rtol=2e-5, atol=2e-6)
        saved = deepcopy(state.backbone_state)
        assert len(saved.layers) == num_layers
        # A second proposal at the same anchor commits no context and cannot modify S.
        replay = replace(cached, target_hidden_states=data.target_hidden_states[:, :0],
                         context_position_ids=torch.empty(1, 0, dtype=torch.long),
                         draft_embeddings=torch.randn_like(block))
        model(replay)
        assert not state.layers  # block attention does not add a persistent KV cache
        for index, old in saved.layers.items():
            new = state.backbone_state.layers[index]
            torch.testing.assert_close(new.recurrent_state, old.recurrent_state, rtol=0, atol=0)
            torch.testing.assert_close(new.kv_conv_state, old.kv_conv_state, rtol=0, atol=0)


def test_future_context_other_anchors_and_invalid_blocks_are_isolated():
    model = ComposableDraftModel(config()).eval()
    data = inputs()
    expected = model(data).backbone_hidden_states
    context = data.target_hidden_states.clone(); context[:, 3:] += 100
    hidden = data.draft_embeddings.clone(); hidden[:, 4:] += 100
    actual = model(replace(data, target_hidden_states=context, draft_embeddings=hidden)).backbone_hidden_states
    torch.testing.assert_close(actual[:, :4], expected[:, :4], rtol=0, atol=0)
    actual = model(replace(data, block_keep_mask=torch.tensor([[True, False]]))).backbone_hidden_states
    torch.testing.assert_close(actual[:, :4], expected[:, :4], rtol=0, atol=0)
    assert torch.count_nonzero(actual[:, 4:]) == 0


@pytest.mark.parametrize("conv,num_layers", [(True, 3), (False, 5)])
def test_checkpointed_training_has_same_outputs_and_gradients(conv, num_layers):
    plain = ComposableDraftModel(config(conv=conv, num_layers=num_layers)).train()
    replay = deepcopy(plain); replay.backbone.gradient_checkpointing = True
    data = inputs()
    outputs = []
    for model in (plain, replay):
        output = model(data).backbone_hidden_states
        output.square().mean().backward(); outputs.append(output.detach())
    torch.testing.assert_close(*outputs)
    for (name, a), (_, b) in zip(plain.named_parameters(), replay.named_parameters()):
        assert a.grad is not None, name
        assert b.grad is not None, name
        assert torch.isfinite(a.grad).all(), name
        torch.testing.assert_close(a.grad, b.grad, msg=name)


@pytest.mark.parametrize("num_layers", [3, 5])
def test_export_roundtrip_preserves_learned_gates_and_outputs(tmp_path, num_layers):
    model = ComposableDraftModel(config(num_layers=num_layers)).eval()
    with torch.no_grad():
        model.backbone.layers[0].block_attention.gate_proj.weight.normal_(std=0.2)
        model.backbone.layers[0].block_attention.gate_proj.bias.fill_(0.7)
    data = inputs()
    expected = model(data).backbone_hidden_states
    export_composable(model, tmp_path)
    loaded = AutoModel.from_pretrained(tmp_path, trust_remote_code=True).eval()
    assert all(layer.block_attention is not None for layer in loaded.backbone.layers)
    torch.testing.assert_close(loaded.backbone.layers[0].block_attention.gate_proj.weight,
                               model.backbone.layers[0].block_attention.gate_proj.weight)
    # Dynamic modules have their own dataclasses; uncached structural inputs are shared.
    torch.testing.assert_close(loaded(data).backbone_hidden_states, expected, rtol=0, atol=0)


def test_production_five_gdn_recipe_removes_external_convolutions_and_dense_attention():
    root = Path(__file__).resolve().parents[2]
    path = root / "configs/dflash_gdn5_shared_block_attention_rnn.json"
    c = ComposableDraftConfig(**json.loads(path.read_text()), num_target_layers=36,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128)
    recipe = c.runtime_config().composable_config
    layers = recipe["backbone"]["layers"]
    assert len(layers) == 5
    for layer in layers:
        assert layer["attention"] == dict(representation="gdn", access="linear")
        assert layer["local_operator"] == dict(type="none")
        assert layer["block_attention"] == dict(num_heads=32, head_dim=128, gate_init=0.1,
                                              share_gdn_projections=True)
        assert layer["gdn_config"]["query_conv_kernel_size"] == 2
        assert layer["gdn_config"]["context_conv_kernel_size"] == 4
    assert recipe["proposal"] == dict(type="rnn", rank=256)


@pytest.mark.parametrize("count", [3, 4])
def test_production_recipe_block_attention_layers(count):
    root = Path(__file__).resolve().parents[2]
    path = root / f"configs/37__dflash__gdn4_battn{count}h32d32_gqa1_dense_conv2g16__rnn_r256__ce0p1_l10p9_g4__dataT0.json"
    c = ComposableDraftConfig(**json.loads(path.read_text()), num_target_layers=36,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128)
    layers = c.runtime_config().composable_config["backbone"]["layers"]
    assert ["block_attention" in layer for layer in layers] == [count == 4, True, True, True, False]
    assert layers[-1]["attention"]["representation"] == "gqa"
    assert layers[1]["block_attention"] == dict(num_heads=32, head_dim=32, gate_init=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_bfloat16_branch_forward_and_backward():
    branch = GatedBlockAttention(2560, 8, dict(num_heads=32, head_dim=32, gate_init=0.1))
    branch = branch.cuda().to(torch.bfloat16)
    hidden = torch.randn(2, 16, 2560, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = branch(hidden)
    assert output.shape == hidden.shape and output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    # Compare the fused SDPA path to its independent FP32 mathematical definition.
    blocks = hidden.detach().float().reshape(4, 8, 2560)
    qkv = torch.nn.functional.linear(blocks, branch.qkv_proj.weight.float()).reshape(4, 8, 3, 32, 32)
    q, k, v = (part.transpose(1, 2) for part in qkv.unbind(2))
    values = ((q @ k.transpose(-2, -1) / math.sqrt(32)).softmax(-1) @ v).transpose(1, 2)
    gates = torch.nn.functional.linear(blocks, branch.gate_proj.weight.float(), branch.gate_proj.bias.float()).sigmoid()
    reference = torch.nn.functional.linear((values * gates[..., None]).flatten(-2), branch.out_proj.weight.float()).reshape_as(output)
    torch.testing.assert_close(output.float(), reference, rtol=0.03, atol=0.0005)
    output.float().square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for parameter in branch.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
