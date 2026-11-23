"""Shared Q/K/V/O: grouped heads, one Q/O evaluation, gradients and cache safety."""

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from transformers import AutoModel

from legospec.modeling.draft import LegoSpecConfig, LegoSpecModel
from legospec.modeling.draft.backbone.attention.linear.block import (
    GatedBlockAttention,
)
from legospec.modeling.draft.backbone.attention.linear import (
    recurrence as gated_deltanet,
)
from legospec.modeling.draft.backbone.attention.linear.events import (
    build_event_layout,
)
from legospec.modeling.draft.backbone.state import DraftState
from legospec.modeling.draft.contracts import DraftInputs
from legospec.modeling.draft.export import export_composable


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def config(enabled=True, conv=False, num_layers=3, q_mode="block"):
    layers = [{'attention': {'family': 'linear', 'variant': 'gdn'}} for _ in range(num_layers)]
    for layer in layers:
        if conv:
            layer["local_operator"] = dict(
                type="grouped_conv", kernel_size=2, group_size=4
            )
    return LegoSpecConfig(
        hidden_size=16,
        intermediate_size=32,
        vocab_size=31,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_target_layers=4,
        block_size=4,
        execution_config={"gdn_backend": "reference"},
        composable_config=dict(
            context=dict(type="concat_projection", target_layer_ids=[0, 2]),
            backbone=dict(type="layer_stack", layers=layers),
            attention_configs=dict(
                gdn=dict(
                    linear_num_key_heads=1,
                    linear_num_value_heads=2,
                    linear_key_head_dim=4,
                    linear_value_head_dim=4,
                    query_conv_kernel_size=4 if q_mode == "confirmed_history" else 2,
                    context_conv_kernel_size=2,
                    query_conv_mode=q_mode,
                )
            ),
            proposal=dict(type="parallel"),
            objective=dict(type="hard_ce"),
        ),
    )


def inputs():
    return DraftInputs(
        target_hidden_states=torch.randn(1, 7, 32),
        draft_embeddings=torch.randn(1, 8, 16),
        anchor_positions=torch.tensor([[3, 6]]),
        block_keep_mask=torch.ones(1, 2, dtype=torch.bool),
        mode="features",
    )


@pytest.mark.parametrize("mutation", ["heads", "dim", "flag"])
def test_invalid_shared_recipe(mutation):
    c = config()
    c.composable_config["backbone"]["layers"][0]["block_attention"] = {"num_heads": 4}
    with pytest.raises(ValueError, match="block_attention"):
        c.validate_recipe()


def test_grouped_shared_attention_matches_explicit_softmax_with_different_value_dimension():
    recipe = dict(num_heads=4, head_dim=4, gate_init=0.1, share_gdn_projections=True)
    gdn = dict(
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=6,
    )
    branch = GatedBlockAttention(16, 4, recipe, gdn_config=gdn)
    assert set(dict(branch.named_parameters())) == {
        "gate_proj.weight",
        "gate_proj.bias",
    }
    with torch.no_grad():
        branch.gate_proj.weight.normal_(std=0.2)
    hidden = torch.randn(2, 8, 16, requires_grad=True)
    q = torch.randn(2, 8, 2, 4, requires_grad=True)
    kv = torch.randn(2, 8, 32, requires_grad=True)
    expected = []
    for sample in range(2):
        blocks = []
        for start in (0, 4):
            x = hidden[sample, start : start + 4]
            qb = q[sample, start : start + 4]
            kb = kv[sample, start : start + 4, :8].reshape(4, 2, 4)
            vb = kv[sample, start : start + 4, 8:].reshape(4, 4, 6)
            gates = (x @ branch.gate_proj.weight.T + branch.gate_proj.bias).sigmoid()
            heads = []
            for head in range(4):
                weights = (
                    qb[:, head // 2] @ kb[:, head // 2].T / math.sqrt(4)
                ).softmax(-1)
                heads.append((weights @ vb[:, head]) * gates[:, head, None])
            blocks.append(torch.cat(heads, -1))
        expected.append(torch.cat(blocks))
    actual = branch(hidden, query=q, key_value=kv)
    torch.testing.assert_close(actual, torch.stack(expected))
    actual.square().sum().backward()
    for tensor in (hidden, q, kv, *branch.parameters()):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_both_reads_share_exact_query_and_evaluate_q_and_output_projection_once(
    monkeypatch,
):
    model = LegoSpecModel(config(num_layers=1)).eval()
    layer = model.backbone.layers[0]
    data, seen, calls = inputs(), {}, {"q": 0, "kv": 0, "out": 0}
    handles = []
    for key, module in [
        ("q", layer.attention.mixer.in_proj_q),
        ("kv", layer.attention.mixer.in_proj_kv),
        ("out", layer.attention.mixer.out_proj),
    ]:

        def count(module, args, result, key=key):
            calls[key] += 1

        handles.append(module.register_forward_hook(count))

    def block_hook(module, args, kwargs):
        seen["block_q"] = kwargs["query"].detach().clone()

    handles.append(
        layer.attention.block_attention.register_forward_pre_hook(block_hook, with_kwargs=True)
    )
    original = gated_deltanet.reference_event_scan

    def scan(q, *args, **kwargs):
        seen["gdn_q"] = q.detach().clone()
        return original(q, *args, **kwargs)

    monkeypatch.setattr(gated_deltanet, "reference_event_scan", scan)
    model(data)
    for handle in handles:
        handle.remove()
    assert calls == {"q": 1, "kv": 2, "out": 1}
    layout = build_event_layout(data, model.block_size)
    torch.testing.assert_close(
        seen["gdn_q"][0, layout.read_events],
        seen["block_q"].reshape(-1, 1, 4)[layout.query_indices],
        rtol=0,
        atol=0,
    )


def test_block_loss_updates_shared_q_kv_and_output_weights():
    model = LegoSpecModel(config(num_layers=1)).train()
    layer = model.backbone.layers[0]
    # Remove the history-read contribution; shared projections must still train.
    hook = layer.attention.mixer.norm.register_forward_hook(lambda module, args, out: out * 0)
    model(inputs()).backbone_hidden_states.square().mean().backward()
    hook.remove()
    for module in (
        layer.attention.mixer.in_proj_q,
        layer.attention.mixer.in_proj_kv,
        layer.attention.mixer.out_proj,
        layer.attention.block_attention.gate_proj,
    ):
        assert (
            module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
        )
        assert module.weight.grad.abs().sum() > 0




@pytest.mark.parametrize(
    "conv,q_mode,num_layers",
    [(False, "block", 5), (True, "block", 3), (False, "confirmed_history", 3)],
)
def test_cached_and_full_prefix_match_without_draft_writes(conv, q_mode, num_layers):
    model = LegoSpecModel(
        config(conv=conv, q_mode=q_mode, num_layers=num_layers)
    ).eval()
    data, state = inputs(), DraftState()
    for anchor in (3, 6):
        start = (
            0
            if state.backbone_state is None
            else int(state.backbone_state.next_positions[0])
        )
        full = replace(
            data,
            draft_embeddings=torch.randn(1, 4, 16),
            anchor_positions=torch.tensor([[anchor]]),
            block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        cached = replace(
            full,
            target_hidden_states=data.target_hidden_states[:, start:anchor],
            context_position_ids=torch.arange(start, anchor)[None],
            draft_state=state,
        )
        actual = model(cached).backbone_hidden_states
        torch.testing.assert_close(
            actual, model(full).backbone_hidden_states, rtol=2e-5, atol=2e-6
        )
        saved = deepcopy(state.backbone_state)
        replay = replace(
            cached,
            target_hidden_states=data.target_hidden_states[:, :0],
            context_position_ids=torch.empty(1, 0, dtype=torch.long),
            draft_embeddings=torch.randn_like(full.draft_embeddings),
        )
        model(replay)
        assert not state.layers
        for index, old in saved.layers.items():
            new = state.backbone_state.layers[index]
            for key in ("recurrent_state", "kv_conv_state", "q_conv_state"):
                before, after = getattr(old, key), getattr(new, key)
                if before is None:
                    assert after is None
                else:
                    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_future_context_other_blocks_and_invalid_blocks_are_isolated():
    model = LegoSpecModel(config()).eval()
    data = inputs()
    expected = model(data).backbone_hidden_states
    context = data.target_hidden_states.clone()
    context[:, 3:] += 100
    hidden = data.draft_embeddings.clone()
    hidden[:, 4:] += 100
    actual = model(
        replace(data, target_hidden_states=context, draft_embeddings=hidden)
    ).backbone_hidden_states
    torch.testing.assert_close(actual[:, :4], expected[:, :4], rtol=0, atol=0)
    masked = model(
        replace(data, block_keep_mask=torch.tensor([[True, False]]))
    ).backbone_hidden_states
    torch.testing.assert_close(masked[:, :4], expected[:, :4], rtol=0, atol=0)
    assert torch.count_nonzero(masked[:, 4:]) == 0


@pytest.mark.parametrize("conv", [False, True])
def test_checkpointed_training_preserves_outputs_and_shared_gradients(conv):
    plain = LegoSpecModel(config(conv=conv)).train()
    checkpointed = deepcopy(plain)
    checkpointed.backbone.gradient_checkpointing = True
    data = inputs()
    outputs = []
    for model in (plain, checkpointed):
        output = model(data).backbone_hidden_states
        output.square().mean().backward()
        outputs.append(output.detach())
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for (name, a), (_, b) in zip(
        plain.named_parameters(), checkpointed.named_parameters()
    ):
        assert a.grad is not None and b.grad is not None, name
        assert torch.isfinite(a.grad).all(), name
        torch.testing.assert_close(a.grad, b.grad, msg=name)


def test_export_roundtrip_preserves_shared_recipe_and_gates(tmp_path):
    model = LegoSpecModel(config()).eval()
    data = inputs()
    with torch.no_grad():
        model.backbone.layers[0].attention.block_attention.gate_proj.weight.normal_(std=0.2)
    expected = model(data).backbone_hidden_states
    export_composable(model, tmp_path)
    loaded = AutoModel.from_pretrained(tmp_path, trust_remote_code=True).eval()
    torch.testing.assert_close(
        loaded(data).backbone_hidden_states, expected, rtol=0, atol=0
    )
    assert set(loaded.state_dict()) == set(model.state_dict())
    for layer in loaded.backbone.layers:
        assert set(layer.attention.block_attention.state_dict()) == {
            "gate_proj.weight",
            "gate_proj.bias",
        }


def test_production_recipe_only_adds_five_head_gates_without_external_convolution():
    root = Path(__file__).resolve().parents[2]
    common = dict(
        hidden_size=2560,
        intermediate_size=9728,
        vocab_size=151936,
        num_target_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )
    new = root / "configs/dflash_gdn5_shared_block_attention_rnn.json"
    recipe = json.loads(new.read_text())
    with torch.device("meta"):
        shared = LegoSpecModel(LegoSpecConfig(**recipe, **common))
    gates = sum(p.numel() for name, p in shared.named_parameters() if "block_attention.gate_proj." in name)
    assert gates == 5 * (2560 * 32 + 32) == 409760
    assert len(shared.backbone.layers) == 5 and not shared.backbone.attention_indices
    for layer in shared.backbone.layers:
        assert layer.attention_conv is None and layer.mlp_conv is None
        assert layer.attention.block_attention.head_dim == 128
        assert layer.attention.block_attention.num_heads == 32
        assert set(layer.attention.block_attention.state_dict()) == {
            "gate_proj.weight",
            "gate_proj.bias",
        }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_cuda_bfloat16_matches_fp32_and_backpropagates_to_all_inputs():
    recipe = dict(num_heads=32, head_dim=128, gate_init=0.1, share_gdn_projections=True)
    gdn = dict(
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    branch = (
        GatedBlockAttention(2560, 8, recipe, gdn_config=gdn).cuda().to(torch.bfloat16)
    )
    hidden = torch.randn(
        1, 16, 2560, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    q = torch.randn(
        1, 16, 16, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    kv = torch.randn(
        1, 16, 6144, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    output = branch(hidden, query=q, key_value=kv)
    qb = (
        q.detach()
        .float()
        .reshape(2, 8, 16, 128)
        .transpose(1, 2)
        .repeat_interleave(2, 1)
    )
    kb = (
        kv.detach()
        .float()[..., :2048]
        .reshape(2, 8, 16, 128)
        .transpose(1, 2)
        .repeat_interleave(2, 1)
    )
    vb = kv.detach().float()[..., 2048:].reshape(2, 8, 32, 128).transpose(1, 2)
    weights = (qb @ kb.transpose(-2, -1) / math.sqrt(128)).softmax(-1)
    gates = F.linear(
        hidden.detach().float().reshape(2, 8, 2560),
        branch.gate_proj.weight.float(),
        branch.gate_proj.bias.float(),
    ).sigmoid()
    expected = ((weights @ vb).transpose(1, 2) * gates[..., None]).reshape(1, 16, 4096)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output.float(), expected, rtol=0.03, atol=0.001)
    output.float().square().mean().backward()
    for tensor in (hidden, q, kv, *branch.parameters()):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert q.grad.abs().sum() > 0 and kv.grad.abs().sum() > 0
