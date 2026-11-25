from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from legospec.training.batch import prepare_layout
from legospec.modeling.draft.composable.objective import build_objective
from legospec.modeling.draft.composable.objective.base import Supervision
from legospec.modeling.draft.composable.objective.vocab import (
    VocabCache,
    project_teacher_rows,
)
from legospec.training.step import TrainingStep
from legospec.modeling.draft.composable import LegoSpecConfig, LegoSpecModel
from legospec.modeling.draft.composable.contracts import ProposalOutput
from legospec.modeling.target.target_utils import FlattenedLMHead


TOKEN_TYPES = ("hard_ce", "soft_ce", "forward_kl", "tv", "l1", "nla", "lk_alpha")
OBJECTIVES = TOKEN_TYPES + (
    "dpace",
    "tv_survival",
    "opd_survival",
    "e2e_tv",
    "dpal_renyi",
    "dpal_plus_plus",
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_mixed_losses_preserve_native_values_and_gradients(dtype):
    q = torch.randn(2, 3, 7, 31, dtype=dtype, requires_grad=True)
    p = torch.randn_like(q, requires_grad=True)
    labels = torch.randint(31, q.shape[:-1])
    mask = torch.ones_like(labels, dtype=torch.float32)
    mask[0, 1] = 0
    supervision = Supervision(labels=labels, prediction_mask=mask, target_logits=p)
    components = [dict(type=name, weight=0.25) for name in TOKEN_TYPES]
    components += [dict(type="l1", weight=0.2), dict(type="soft_ce", weight=0)]
    mixed = build_objective(
        dict(type="weighted_sum", components=components, loss_decay_gamma=4.0)
    )
    prediction = ProposalOutput(q, draft_logits=q)
    actual = mixed(prediction, supervision).loss_terms.numerator
    expected = sum(
        item["weight"]
        * build_objective(dict(type=item["type"], loss_decay_gamma=4.0))(
            prediction, supervision
        ).loss_terms.numerator
        for item in components
        if item["weight"]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (actual_grad,) = torch.autograd.grad(actual, q, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, q)
    tolerance = (
        dict(rtol=2e-6, atol=2e-8) if dtype == torch.float32 else dict(rtol=0, atol=0)
    )
    torch.testing.assert_close(actual_grad, expected_grad, **tolerance)
    assert p.grad is None


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_native_softmax_is_shared_without_changing_equal_distribution_subgradient(
    dtype,
):
    q = torch.randn(2, 3, 7, 31, dtype=dtype, requires_grad=True)
    p = q.detach().clone().requires_grad_()
    labels = torch.zeros(q.shape[:-1], dtype=torch.long)
    supervision = Supervision(
        labels, torch.ones_like(labels), p, vocab_cache=VocabCache()
    )
    loss = build_objective(
        dict(
            type="weighted_sum",
            loss_decay_gamma=None,
            components=[dict(type="l1", weight=0.5), dict(type="tv", weight=0.5)],
        )
    )
    with patch("torch.softmax", wraps=torch.softmax) as softmax:
        value = loss(
            ProposalOutput(q, draft_logits=q), supervision
        ).loss_terms.numerator
        assert softmax.call_count == (2 if dtype == torch.float32 else 3)
    assert value.item() == 0
    value.backward()
    torch.testing.assert_close(q.grad, torch.zeros_like(q), rtol=0, atol=0)
    assert p.grad is None


def test_distribution_cache_does_not_cross_loss_calls_or_confuse_logits():
    q = torch.randn(3, 31, requires_grad=True)
    cache = VocabCache()
    first = cache.distribution(q)
    assert first is cache.distribution(q)
    assert first.float_logits() is first.float_logits()
    assert first.probabilities() is first.probabilities()
    assert first is not cache.distribution(q.clone())
    assert first is not VocabCache().distribution(q)
    detached = cache.detached_probabilities(q)
    assert not detached.requires_grad
    assert detached.data_ptr() == first.probabilities().data_ptr()


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_indexed_projection_preserves_rows_and_softcap(dtype):
    hidden = torch.randn(2, 13, 16, dtype=dtype)
    indices = torch.tensor([[[1, 2, 3], [2, 3, 4]], [[1, 1, 1], [1, 2, 3]]])
    head = (
        FlattenedLMHead(16, 31, bias=False, softcap=30.0)
        .to(dtype)
        .requires_grad_(False)
    )
    gathered = hidden.gather(1, indices.flatten(1)[..., None].expand(-1, -1, 16))
    with torch.no_grad():
        expected = head(gathered).reshape(2, 2, 3, 31)
    rows = []
    hook = head.register_forward_pre_hook(lambda _, args: rows.append(args[0].shape[0]))
    actual = project_teacher_rows(head, hidden, indices)
    hook.remove()
    assert rows == [7]
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    value = torch.randn(5, 16, dtype=dtype, requires_grad=True)
    actual = head(value)
    reference = torch.tanh(F.linear(value, head.weight) / 30.0) * 30.0
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), value, retain_graph=True)[0],
        torch.autograd.grad(reference.sum(), value)[0],
        rtol=0,
        atol=0,
    )


def wrapper(kind, proposal="parallel"):
    objective = dict(
        type=kind,
        loss_decay_gamma=None
        if kind in ("dpal_renyi", "dpal_plus_plus", "e2e_tv")
        else 4.0,
    )
    if kind == "weighted_sum":
        objective["components"] = [
            dict(type="hard_ce", weight=0.1),
            dict(type="l1", weight=0.9),
        ]
    prop = dict(type=proposal)
    if proposal != "parallel":
        prop["rank"] = 4
    if "selector" in proposal:
        prop["top_k"] = 3
        objective = dict(
            type="selector_losses",
            loss_decay_gamma=objective["loss_decay_gamma"],
            base=dict(type="hard_ce", weight=0.1),
            final={
                **{k: v for k, v in objective.items() if k != "loss_decay_gamma"},
                "weight": 1.0,
            },
        )
    recipe = dict(
        context=dict(type="concat_projection", target_layer_ids=[0, 1]),
        backbone=dict(
            type="layer_stack", layers=[dict(attention="gqa", access="dense")]
        ),
        proposal=prop,
        objective=objective,
        attention_configs=dict(gqa={}),
        access_configs={},
    )
    config = LegoSpecConfig(
        composable_config=recipe,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_target_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=31,
        block_size=8,
        execution_config=dict(attention_backend="sdpa"),
        attention_dropout=0.0,
    )
    head = FlattenedLMHead(16, 31, bias=False).requires_grad_(False)
    return TrainingStep(
        LegoSpecModel(config),
        head,
        nn.Embedding(31, 16).requires_grad_(False),
        4,
        objective_chunk_blocks=2,
    )


@pytest.mark.parametrize("kind", OBJECTIVES + ("weighted_sum",))
@pytest.mark.parametrize("proposal", ("parallel", "markov", "residual_selector"))
def test_all_loss_wrappers_keep_projection_alignment_and_checkpoint_gradients(
    kind, proposal
):
    train = wrapper(kind, proposal)
    reference = deepcopy(train)
    ids = torch.randint(31, (2, 24))
    mask = torch.ones_like(ids, dtype=torch.float32)
    mask[0, 17:] = 0
    layout = prepare_layout(ids, mask, 8, 8, complete_blocks=True)
    data = dict(
        input_ids=ids,
        loss_mask=mask,
        target_hidden_states=torch.randn(2, 24, 32),
        target_last_hidden_states=torch.randn(2, 24, 16),
        layout=layout,
        collect_detailed_metrics=True,
    )
    actual = train(**data)
    actual.loss_terms.numerator.backward()

    def direct(head, hidden, indices):
        selected = hidden.gather(
            1, indices.flatten(1)[..., None].expand(-1, -1, hidden.size(-1))
        )
        with torch.no_grad():
            return head(selected).reshape(*indices.shape, -1)

    with patch("legospec.training.step.project_teacher_rows", direct):
        expected = reference(**data)
        expected.loss_terms.numerator.backward()
    torch.testing.assert_close(
        actual.loss_terms.numerator, expected.loss_terms.numerator
    )
    torch.testing.assert_close(
        actual.loss_terms.denominator, expected.loss_terms.denominator
    )
    for (name, parameter), (other_name, other) in zip(
        train.named_parameters(), reference.named_parameters()
    ):
        assert name == other_name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad)


def test_topk_selector_keeps_separate_candidate_and_full_vocab_normalizers():
    train = wrapper("hard_ce", "topk_selector")
    ids = torch.randint(31, (2, 24))
    mask = torch.ones_like(ids, dtype=torch.float32)
    data = dict(
        input_ids=ids,
        loss_mask=mask,
        target_hidden_states=torch.randn(2, 24, 32),
        target_last_hidden_states=torch.randn(2, 24, 16),
        layout=prepare_layout(ids, mask, 8, 8, complete_blocks=True),
        collect_detailed_metrics=True,
    )
    output = train(**data)
    output.loss_terms.numerator.backward()
    assert torch.isfinite(output.loss)
    assert (
        train.draft_model.proposal.candidate_selector.successor_codebook.grad
        is not None
    )
