"""Frozen component verifier. Original source SHA256: 126f644e5f0effcba43e41173cef93f4747a0bc801e8ee05e891381c47387889"""
import torch

def draw(probs, generator):
    return int(torch.multinomial(probs, 1, generator=generator).item())

def verify(proposal_ids, q_rows, p_rows, temperature, generator):
    """Returns accepted proposal prefix length and next target token."""
    for j, token in enumerate(proposal_ids):
        p, q = p_rows[j], q_rows[j]
        if temperature == 0:
            accepted = token == int(p.argmax())
        else:
            assert q[token] > 0, 'A sampled proposal has zero probability'
            accepted = bool(torch.rand((), device=p.device, generator=generator)
                            < torch.minimum(torch.ones_like(p[token]), p[token] / q[token]))
        if not accepted:
            residual = (p-q).clamp_min(0)
            # Positive rejection probability implies positive residual mass.
            if not torch.isfinite(residual).all() or residual.sum() <= 0:
                raise ArithmeticError('Invalid rejection residual')
            replacement = int(p.argmax()) if temperature == 0 else draw(residual/residual.sum(), generator)
            return j, replacement
    bonus = int(p_rows[-1].argmax()) if temperature == 0 else draw(p_rows[-1], generator)
    return len(proposal_ids), bonus
