"""Numerical E2E-TV, sampled replay, and causal query visibility."""

import legospec.training.execution.sampled as execution_sampled

from dataclasses import replace
from pathlib import Path
import tempfile
from unittest.mock import Mock

import pytest
import torch

from legospec.training.execution import sampled as anchor_opd
from legospec.modeling.draft.composable.objective import build_objective
from legospec.modeling.draft.composable.objective.base import Supervision
from legospec.training.execution.step import TrainingStep
from legospec.modeling.draft.composable import LegoSpecConfig, LegoSpecModel
from legospec.modeling.draft.composable.contracts import (
    ProposalOutput,
    DraftState,
    DraftInputs,
)
from legospec.modeling.draft.composable.export import export_composable
from test_anchor_opd import fixture


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
    old, scorer, values = fixture("rnn", device)
    cfg = old.draft_model.config
    cfg.execution_config["attention_backend"] = backend
    recipe = cfg.composable_config
    if mode is not None:
        recipe["backbone"]["query_attention"] = mode
    if conv:
        recipe["backbone"]["layers"][0]["local_operator"] = {
            "type": "grouped_conv",
            "kernel_size": 2,
            "group_size": 4,
        }
    recipe["objective"] = {"type": "e2e_tv"}
    model = LegoSpecModel(cfg).to(device)
    wrap = TrainingStep(
        model,
        old.lm_head,
        old.embed_tokens,
        3,
        num_anchors=4,
        objective_chunk_blocks=0,
        opd_data_mode="sampled_anchor",
    )
    return wrap, scorer, values


@pytest.mark.parametrize("mode", ["full_parallel", "causal_parallel"])
def test_sampled_opd_e2e_uses_one_target_rollout(mode, monkeypatch):
    wrapper, scorer, inputs = model_with_mode(mode)
    with torch.no_grad():
        teacher = scorer.model.model(
            inputs["input_ids"],
            attention_mask=inputs["context_valid_mask"],
            output_hidden_states=True,
        )
    inputs.update(
        target_hidden_states=teacher.hidden_states[1],
        target_last_hidden_states=teacher.last_hidden_state,
    )
    scored, projected = [], []
    score = scorer.score_sampled_blocks

    def record_score(*args, **kwargs):
        hidden = score(*args, **kwargs)
        scored.append(hidden)
        return hidden

    monkeypatch.setattr(scorer, "score_sampled_blocks", record_score)
    verifier = execution_sampled.verify_sampled_blocks
    verify = Mock(wraps=verifier)
    monkeypatch.setattr(anchor_opd, "verify_sampled_blocks", verify)
    hook = wrapper.lm_head.register_forward_hook(
        lambda _, args, logits: projected.append((args[0], logits))
    )
    results = []
    try:
        for log in (False, True):
            torch.manual_seed(123)
            wrapper.zero_grad(set_to_none=True)
            verify.reset_mock()
            projected.clear()
            out = wrapper(**inputs, opd_scorer=scorer, collect_detailed_metrics=log)
            assert 0 <= out.loss <= 1
            assert "final/e2e_tv/expected_prefix" in out.ratio_metrics
            (branch_logits,) = (
                logits for hidden, logits in projected if hidden is scored[-1]
            )
            if log:
                verify.assert_called_once()
                logits, *rollout = verify.call_args.args
                assert logits is branch_logits
                accepted = out.ratio_metrics["rollout/accepted_length"]
                reachable = out.ratio_metrics["rollout/reachable_positions"]
                with torch.no_grad():
                    reference = verifier(wrapper.lm_head(scored[-1]), *rollout)
                assert (
                    accepted.denominator
                    == reachable.denominator
                    == out.loss_terms.denominator
                )
                torch.testing.assert_close(accepted.numerator, reference.sum().float())
                keep = rollout[-1]
                torch.testing.assert_close(
                    reachable.numerator,
                    (
                        (reference + 1).clamp_max(wrapper.draft_model.block_size - 1)
                        * keep
                    )
                    .sum()
                    .float(),
                )
            else:
                assert not any(key.startswith("rollout/") for key in out.ratio_metrics)
            out.loss.backward()
            assert verify.call_count == int(log)
            grad = wrapper.draft_model.context.projection.weight.grad
            assert torch.isfinite(grad).all() and grad.abs().sum() > 0
            results.append((out.loss.detach(), grad.detach().clone()))
    finally:
        hook.remove()
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
    assert (scorer.prefill_calls, scorer.branch_calls) == (2, 2)
    assert all(p.grad is None for p in scorer.model.parameters())


def inputs_for(device="cpu"):
    return DraftInputs(
        torch.randn(1, 9, 16, device=device),
        torch.randn(1, 8, 16, device=device),
        torch.tensor([[3, 7]], device=device),
        torch.tensor([[True, True]], device=device),
        mode="features",
    )


def test_causal_prefix_cross_block_and_legacy_full():
    wrapper, _, _ = model_with_mode("causal_parallel")
    causal = wrapper.draft_model.eval()
    legacy, _, _ = model_with_mode(None)
    full = legacy.draft_model.eval()
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

    model = model_with_mode("causal_parallel")[0].draft_model.eval()
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
    base = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(dir=base) as folder:
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
    sdpa = model_with_mode("causal_parallel", "cuda")[0].draft_model.eval()
    flex = model_with_mode("causal_parallel", "cuda", "flex_attention")[
        0
    ].draft_model.eval()
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
