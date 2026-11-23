"""Probabilistic p/q acceptance and positive p-q residual sampling."""

import torch
from legospec.modeling.draft.proposal.sampling import draw


def verify(proposal_ids, q_rows, p_rows, temperature, generator):
    for j, token in enumerate(proposal_ids):
        p, q = p_rows[j], q_rows[j]
        if temperature == 0:
            accepted = token == int(p.argmax())
        else:
            if q[token] <= 0:
                raise ValueError("Sampled proposal has zero probability")
            accepted = bool(
                torch.rand((), device=p.device, generator=generator)
                < torch.minimum(torch.ones_like(p[token]), p[token] / q[token])
            )
        if not accepted:
            residual = (p - q).clamp_min(0)
            if not torch.isfinite(residual).all() or residual.sum() <= 0:
                raise ArithmeticError("Invalid rejection residual")
            replacement = (
                int(p.argmax())
                if temperature == 0
                else draw(residual / residual.sum(), generator)
            )
            return j, replacement
    bonus = (
        int(p_rows[-1].argmax()) if temperature == 0 else draw(p_rows[-1], generator)
    )
    return len(proposal_ids), bonus
