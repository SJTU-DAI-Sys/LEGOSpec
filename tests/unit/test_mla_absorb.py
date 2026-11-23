from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from transformers import Qwen3Config
from legospec.modeling.draft.backbone.attention.latent import MLAAttention
from legospec.modeling.draft.backbone.attention.base import (
    AttentionForward,
)
from legospec.modeling.draft.backbone.state import DraftState, MLALayerState


def make(separate, window, interleaved=True, dtype=torch.float32, device="cpu"):
    cfg = Qwen3Config(
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        head_dim=8,
        layer_types=["full_attention"],
        attention_dropout=0.0,
        attention_bias=False,
    )
    cfg.q_lora_rank = 8
    cfg.kv_lora_rank = 6
    cfg.qk_nope_head_dim = 4
    cfg.qk_rope_head_dim = 4
    cfg.v_head_dim = 4
    cfg.rope_interleave = interleaved
    cfg.rope_parameters = {"rope_type": "default"}
    cfg._attn_implementation = "sdpa"
    recipe = (
        {"access": "dense"}
        if window is None
        else {"access": "sliding_window", "window_size": window}
    )
    model = (
        MLAAttention(cfg, 0, recipe, separate)
        .to(device=device, dtype=dtype)
        .eval()
    )
    if separate:
        with torch.no_grad():
            model.context_kv_b_proj.weight.mul_(1.7)
    return model


def inputs(context, anchor, state=None):
    batch = context.size(0)
    positions = torch.arange(context.size(1), device=context.device)[None].expand(
        batch, -1
    )
    return SimpleNamespace(
        context_position_ids=positions,
        context_valid_mask=torch.ones_like(positions, dtype=torch.bool),
        document_ids=torch.zeros_like(positions),
        anchor_document_ids=None,
        anchor_positions=torch.full((batch, 1), anchor, device=context.device),
        block_keep_mask=torch.ones(batch, 1, device=context.device, dtype=torch.bool),
        draft_state=state,
        packed_plan=None,
        selection=None,
    )


def rope(positions, dtype):
    angle = positions[..., None].float() * torch.tensor(
        [0.13, 0.27, 0.13, 0.27], device=positions.device
    )
    return angle.cos().to(dtype), angle.sin().to(dtype)


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("window", [None, 3])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_absorbed_cache_and_output(separate, window, interleaved, precision):
    torch.manual_seed(42)
    device = "cuda" if precision == "bf16" else "cpu"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    model = make(separate, window, interleaved, dtype, device)
    context = torch.randn(2, 7, 16, device=device, dtype=dtype)
    query = torch.randn(2, 4, 16, device=device, dtype=dtype)
    state = DraftState()
    with torch.no_grad():
        for start, end in [(0, 5), (5, 7), (7, 7)]:
            packet = inputs(context[:, start:end], end, state)
            packet.context_position_ids = packet.context_position_ids + start
            reference = inputs(context[:, :end], end)
            if end == 7:
                packet.block_keep_mask[1] = False
                reference.block_keep_mask[1] = False
            pos = torch.cat(
                (
                    packet.context_position_ids,
                    torch.arange(end, end + 4, device=device)[None].expand(2, -1),
                ),
                1,
            )
            fullpos = torch.cat(
                (
                    reference.context_position_ids,
                    torch.arange(end, end + 4, device=device)[None].expand(2, -1),
                ),
                1,
            )
            expected = AttentionForward.forward(
                model, query, context[:, :end], rope(fullpos, dtype), reference
            )
            actual = model(query, context[:, start:end], rope(pos, dtype), packet)
            torch.testing.assert_close(
                actual,
                expected,
                atol=0.015 if precision == "bf16" else 2e-6,
                rtol=0.04 if precision == "bf16" else 2e-5,
            )
            cache = state.layers[0]
            assert isinstance(cache, MLALayerState)
            assert cache.kv_latent.shape[-1] == 6 and cache.key_rope.shape[-1] == 4
            assert cache.kv_latent.ndim == 3
            assert cache.positions.size(1) <= (end if window is None else window)
        state.rollback(torch.tensor([6, 5], device=device))
        assert not (
            state.layers[0].valid
            & (state.layers[0].positions >= torch.tensor([[6], [5]], device=device))
        ).any()


def test_training_outputs_gradients_and_weight_format_unchanged():
    torch.manual_seed(43)
    model = make(True, None).train()
    reference = deepcopy(model)
    context = torch.randn(2, 5, 16, requires_grad=True)
    query = torch.randn(2, 4, 16, requires_grad=True)
    packet = inputs(context, 5)
    embeddings = rope(torch.arange(9)[None].expand(2, -1), torch.float32)
    actual = model(query, context, embeddings, packet)
    expected = AttentionForward.forward(reference, query, context, embeddings, packet)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert model.state_dict().keys() == reference.state_dict().keys()
    for p, r in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, r.grad, atol=0, rtol=0)


def test_full_model_generation_and_export(tmp_path):
    from types import MethodType
    from transformers import Qwen3ForCausalLM, AutoModel
    from legospec import LegoSpecConfig, LegoSpecModel
    from legospec.inference.generate import generate
    from legospec.modeling.draft.export import export_composable
    from test_component_boundaries import tiny_model

    cfg = tiny_model("parallel").config.to_dict()
    recipe = cfg["composable_config"]
    recipe["backbone"]["layers"] = [{'attention': {'family': 'latent', 'variant': 'mla', 'access': 'full'}}]
    recipe["attention_configs"] = {
        "mla": dict(
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=8,
            kv_lora_rank=6,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
        )
    }
    draft = LegoSpecModel(LegoSpecConfig(**cfg)).eval()
    draft.mask_token_id = 0
    target = Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=31,
            eos_token_id=None,
            pad_token_id=0,
        )
    ).eval()
    reference = deepcopy(draft)
    for module in reference.modules():
        if isinstance(module, MLAAttention):
            module.forward = MethodType(AttentionForward.forward, module)
    prompt = torch.tensor([[1, 2, 3, 4]])
    actual = generate(target, draft, prompt, temperature=0, max_new_tokens=12)
    expected = generate(target, reference, prompt, temperature=0, max_new_tokens=12)
    assert actual["generated_ids"] == expected["generated_ids"]
    export_composable(draft, tmp_path)
    loaded = AutoModel.from_pretrained(tmp_path, trust_remote_code=True).eval()
    loaded.mask_token_id = 0
    assert draft.state_dict().keys() == loaded.state_dict().keys()
    assert (
        generate(target, loaded, prompt, temperature=0, max_new_tokens=12)[
            "generated_ids"
        ]
        == actual["generated_ids"]
    )
