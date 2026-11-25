"""Numerical E2E-TV and causal query visibility."""

import tempfile
from dataclasses import replace

import pytest
import torch

from legospec.modeling.draft import LegoSpecConfig, LegoSpecModel
from legospec.modeling.draft.backbone.state import DraftState
from legospec.modeling.draft.contracts import DraftInputs, ProposalOutput
from legospec.modeling.draft.export import export_composable
from legospec.modeling.draft.objective import build_objective
from legospec.modeling.draft.objective.base import Supervision


def test_formula_gradient_and_complete_blocks():
    q = torch.tensor([0.75, 0.25]).log().repeat(1, 2, 2, 1).requires_grad_()
    p = torch.tensor([0.25, 0.75]).log().repeat(1, 2, 2, 1).requires_grad_()
    mask = torch.tensor([[[1, 1], [1, 0]]])
    with torch.no_grad():
        q[:, 1] = float("nan")
    result = build_objective({"type": "e2e_tv"})(
        ProposalOutput(q, draft_logits=q), Supervision(mask, mask, p)
    )
    assert result.loss_terms.denominator == 1
    torch.testing.assert_close(result.loss, torch.tensor(0.625))
    result.loss.backward()
    torch.testing.assert_close(
        q.grad[0, 0], torch.tensor([[0.140625, -0.140625], [0.046875, -0.046875]])
    )
    assert not q.grad[:, 1].count_nonzero() and p.grad is None
    with pytest.raises(ValueError, match="gamma"):
        build_objective({"type": "e2e_tv", "loss_decay_gamma": 4})


def model_with_mode(mode, device="cpu", backend="sdpa", conv=True):
    recipe = {
        "context": {"type": "concat_projection", "target_layer_ids": [0]},
        "backbone": {
            "type": "layer_stack",
            "layers": [{'attention': {'family': 'dense', 'variant': 'gqa', 'access': 'full'}}],
        },
        "attention_configs": {"gqa": {}},
        "proposal": {"type": "rnn", "rank": 4},
        "objective": {"type": "e2e_tv"},
    }
    if mode is not None:
        recipe["backbone"]["query_attention"] = mode
    if conv:
        recipe["backbone"]["layers"][0]["local_operator"] = {
            "type": "grouped_conv",
            "kernel_size": 2,
            "group_size": 4,
        }
    config = LegoSpecConfig(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_target_layers=2,
        vocab_size=23,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        block_size=4,
        attention_dropout=0.0,
        composable_config=recipe,
        execution_config={"attention_backend": backend},
    )
    return LegoSpecModel(config).to(device)


def inputs_for(device="cpu"):
    return DraftInputs(
        torch.randn(1, 9, 16, device=device),
        torch.randn(1, 8, 16, device=device),
        torch.tensor([[3, 7]], device=device),
        torch.tensor([[True, True]], device=device),
        mode="features",
    )


def test_causal_prefix_cross_block_and_legacy_full():
    causal = model_with_mode("causal_parallel").eval()
    full = model_with_mode(None).eval()
    full.load_state_dict(causal.state_dict(), strict=True)
    inputs = inputs_for()
    changed = inputs.draft_embeddings.clone()
    changed[:, 3] += torch.randn_like(changed[:, 3])
    perturbed = replace(inputs, draft_embeddings=changed)
    with torch.no_grad():
        base = causal(inputs).backbone_hidden_states
        other = causal(perturbed).backbone_hidden_states
        torch.testing.assert_close(base[:, :3], other[:, :3], atol=0, rtol=0)
        torch.testing.assert_close(base[:, 4:], other[:, 4:], atol=0, rtol=0)
        assert not torch.allclose(
            full(inputs).backbone_hidden_states[:, :3],
            full(perturbed).backbone_hidden_states[:, :3],
        )
        padding = replace(inputs, block_keep_mask=torch.tensor([[True, False]]))
        torch.testing.assert_close(
            causal(padding).backbone_hidden_states[:, :4], base[:, :4]
        )
    explicit = LegoSpecConfig.from_dict(full.config.to_dict())
    explicit.composable_config["backbone"]["query_attention"] = "full_parallel"
    ref = LegoSpecModel(explicit).eval()
    ref.load_state_dict(full.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(
            ref(inputs).backbone_hidden_states,
            full(inputs).backbone_hidden_states,
            atol=0,
            rtol=0,
        )


def test_causal_export_and_incremental_cache():
    from transformers import AutoModel

    model = model_with_mode("causal_parallel").eval()
    data = inputs_for()
    first = replace(
        data,
        anchor_positions=data.anchor_positions[:, :1],
        block_keep_mask=data.block_keep_mask[:, :1],
        draft_embeddings=data.draft_embeddings[:, :4],
        target_hidden_states=data.target_hidden_states[:, :3],
        draft_state=DraftState(),
    )
    second = replace(
        data,
        anchor_positions=data.anchor_positions[:, 1:],
        block_keep_mask=data.block_keep_mask[:, 1:],
        draft_embeddings=data.draft_embeddings[:, 4:],
    )
    cached = replace(
        second,
        target_hidden_states=data.target_hidden_states[:, 3:7],
        context_position_ids=torch.arange(3, 7)[None],
        draft_state=first.draft_state,
    )
    with torch.no_grad():
        model(first)
        expected = model(second).backbone_hidden_states
        torch.testing.assert_close(model(cached).backbone_hidden_states, expected)
    with tempfile.TemporaryDirectory() as folder:
        export_composable(model, folder)
        loaded = AutoModel.from_pretrained(
            folder, trust_remote_code=True, local_files_only=True
        ).eval()
        assert (
            loaded.config.composable_config["backbone"]["query_attention"]
            == "causal_parallel"
        )
        with torch.no_grad():
            torch.testing.assert_close(loaded(second).backbone_hidden_states, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
def test_causal_sdpa_flex_and_fused_loss():
    sdpa = model_with_mode("causal_parallel", "cuda").eval()
    flex = model_with_mode("causal_parallel", "cuda", "flex_attention").eval()
    flex.load_state_dict(sdpa.state_dict(), strict=True)
    data = inputs_for("cuda")
    with torch.no_grad():
        torch.testing.assert_close(
            flex(data).backbone_hidden_states,
            sdpa(data).backbone_hidden_states,
            rtol=2e-4,
            atol=2e-4,
        )
    q = torch.randn(2, 4, 7, 19, requires_grad=True)
    p = torch.randn_like(q)
    mask = torch.ones(q.shape[:-1], dtype=torch.bool)
    objective = build_objective({"type": "e2e_tv"})

    def loss(q, p, mask):
        return objective(
            ProposalOutput(q, draft_logits=q), Supervision(mask, mask, p)
        ).loss

    expected = loss(q, p, mask)
    cuda = q.detach().cuda().requires_grad_()
    actual = loss(cuda, p.cuda(), mask.cuda())
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, cuda)[0].cpu(), torch.autograd.grad(expected, q)[0]
    )
