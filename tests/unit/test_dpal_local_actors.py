"""Actor ablations retain D-PAL's detached credit and unchanged reductions."""

import pytest
import torch
from legospec.modeling.draft.objective import build_objective
from legospec.modeling.draft.objective.base import Supervision
from legospec.modeling.draft.objective.dpal import dpal_credit
from legospec.modeling.draft.contracts import ProposalOutput
from legospec.modeling.draft.loss_config import normalize_objective
from test_dpal_renyi import fixture


@pytest.mark.parametrize("kind", ["forward_kl", "tv"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actor_credit_and_gradient(kind, dtype):
    torch.manual_seed(42)
    q = torch.randn(2, 3, 7, 19).to(dtype).requires_grad_()
    p = torch.randn_like(q).requires_grad_()
    mask = torch.ones(q.shape[:-1])
    mask[0, 1] = 0
    mask[1, 2, -1] = 0
    active = mask.bool() & mask.bool().all(-1, keepdim=True)
    with torch.no_grad():
        q[~active] = float("nan")
        p[~active] = float("nan")
    objective = build_objective(dict(type="dpal_renyi", local_loss=kind))
    out = objective(
        ProposalOutput(q, draft_logits=q),
        Supervision(torch.zeros_like(mask).long(), mask, target_logits=p),
    )
    qr = q.detach()[active].requires_grad_()
    pr = p.detach()[active].float()
    logp, logq = pr.log_softmax(-1), qr.float().log_softmax(-1)
    local = (
        (logp.exp() * (logp - logq)).sum(-1)
        if kind == "forward_kl"
        else 0.5 * (pr.softmax(-1) - qr.float().softmax(-1)).abs().sum(-1)
    )
    overlap = torch.zeros_like(mask).masked_scatter(
        active, torch.minimum(logp, logq).exp().sum(-1)
    )
    credit = dpal_credit(overlap, active.float(), 0.3)
    expected = (local * credit[active]).sum() / active.all(-1).sum()
    torch.testing.assert_close(out.loss, expected)
    grad = torch.autograd.grad(out.loss, q)[0]
    ref = torch.autograd.grad(expected, qr)[0]
    torch.testing.assert_close(
        grad[active],
        ref,
        rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
        atol=0.001 if dtype == torch.bfloat16 else 1e-7,
    )
    assert (
        torch.isfinite(grad).all() and grad[~active].abs().sum() == 0 and p.grad is None
    )


@pytest.mark.parametrize("kind", ["renyi", "forward_kl", "tv"])
def test_disable_checkpoint_matches_original_loss_and_gradients(kind):
    torch.manual_seed(99)
    torch.set_num_threads(1)
    model, train, data, layout = fixture("rnn")
    train.loss_branches.objectives["final"] = build_objective(
        dict(type="dpal_renyi", local_loss=kind)
    )
    params = tuple(model.parameters())
    a = train(**data, layout=layout)
    ga = torch.autograd.grad(a.loss, params)
    train.objective_chunk_blocks = 0
    train.objective_gradient_checkpointing = False
    b = train(**data, layout=layout)
    gb = torch.autograd.grad(b.loss, params)
    torch.testing.assert_close(a.loss, b.loss)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, rtol=8e-4, atol=3e-6)


@pytest.mark.parametrize("kind", ["bad", None, 42])
def test_reject_invalid_local_loss(kind):
    with pytest.raises(ValueError, match="local_loss"):
        normalize_objective(dict(type="dpal_renyi", local_loss=kind))
