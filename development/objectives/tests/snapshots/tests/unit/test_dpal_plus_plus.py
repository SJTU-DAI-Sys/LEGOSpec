import pytest
import torch

from legospec.modeling.draft.composable.objective import build_objective
from legospec.modeling.draft.composable.objective.base import Supervision
from legospec.modeling.draft.composable.objective.dpal import (
    dpal_credit,
    renyi_half_and_overlap,
)
from legospec.modeling.draft.composable.contracts import ProposalOutput


def test_dpal_plus_plus_credit_is_detached_marginal():
    rho = 0.3
    overlap = torch.tensor([[0.9, 0.1, 0.9, 0.7]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    smooth = torch.where(
        mask.bool(), rho + (1 - rho) * overlap.detach(), 1
    ).requires_grad_()
    (marginal,) = torch.autograd.grad((smooth.cumprod(-1) * mask).sum(), smooth)
    credit = dpal_credit(overlap, mask, rho, divide_own_overlap=True)
    torch.testing.assert_close(credit, marginal)
    assert not credit.requires_grad


def test_dpal_plus_plus_loss_gradient_and_complete_block_denominator():
    q = torch.randn(1, 3, 7, 11, requires_grad=True)
    p = torch.randn_like(q, requires_grad=True)
    mask = torch.ones(q.shape[:-1])
    mask[:, 2, -1] = 0
    objective = build_objective({"type": "dpal_plus_plus"})
    result = objective(
        ProposalOutput(q, draft_logits=q),
        Supervision(
            labels=torch.zeros_like(mask, dtype=torch.long),
            prediction_mask=mask,
            target_logits=p,
        ),
    )
    actor, overlap = renyi_half_and_overlap(q[:, :2], p[:, :2])
    rho = objective.rho
    smooth = rho + (1 - rho) * overlap
    credit = dpal_credit(overlap, torch.ones_like(overlap), rho) / smooth
    expected = (actor * credit).sum() / 2
    assert result.loss_terms.denominator == 2
    torch.testing.assert_close(result.loss, expected)
    (expected_grad,) = torch.autograd.grad(expected, q)
    result.loss.backward()
    torch.testing.assert_close(q.grad, expected_grad)
    assert p.grad is None
    with pytest.raises(ValueError, match="rho > 0"):
        build_objective({"type": "dpal_plus_plus", "rho": 0})
