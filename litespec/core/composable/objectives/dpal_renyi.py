"""D-PAL with a Renyi-1/2 actor and detached full-vocabulary overlap credit.

The actor and credit scans follow Lucas-TY/D-PAL-SpecForge commit
23a628bd9d65b56c21f99692003ac5a3f844dd17 (selector-free dpard).
complete_block_mean follows this framework's block-loss convention;
sequence_anchor_mean reproduces the author's sequence-balanced reduction.
Neither mode divides by the dynamic credit sum or adds fixed position decay.
"""

# The reference tensor equations are adapted from MIT-licensed SpecForge:
# Copyright (c) 2025 sgl-project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from litespec.core.losses import _run, _should_compile
from litespec.core.outputs import RatioTerms, TrainingOutput
from .base import Objective
from ..vocab import distribution, IndexedTeacherDistribution

REFERENCE_COMMIT = "23a628bd9d65b56c21f99692003ac5a3f844dd17"


def renyi_half_and_overlap(draft_logits, target_logits):
    """FP32 local actor with a frozen teacher and detached acceptance overlap."""
    log_q = F.log_softmax(draft_logits.float(), dim=-1)
    log_p = F.log_softmax(target_logits.detach().float(), dim=-1)
    actor = -2.0 * torch.logsumexp(0.5 * (log_p + log_q), dim=-1)
    if not torch.isfinite(actor).all():
        raise ValueError("D-PAL requires a finite Renyi loss on every supervised position")
    with torch.no_grad():
        overlap = torch.exp(torch.minimum(log_p, log_q)).sum(-1)
    return actor, overlap


def local_actor_and_overlap(draft, teacher, active, local_loss):
    """Alternative actors share the same detached full-vocabulary credit."""
    keep = active[..., None]
    draft = torch.where(keep, draft.float(), 0)
    teacher = torch.where(keep, teacher.detach().float(), 0)
    if local_loss == "forward_kl":
        log_q = F.log_softmax(draft, dim=-1)
        log_p = F.log_softmax(teacher, dim=-1)
        p = log_p.exp()
        actor = torch.where(p > 0, p * (log_p - log_q), 0).sum(-1)
        with torch.no_grad():
            overlap = torch.minimum(log_p, log_q).exp().sum(-1)
    else:
        # Preserve native probability rounding and TV's zero subgradient.
        p, q = teacher.softmax(-1), draft.softmax(-1)
        actor = 0.5 * (p - q).abs().sum(-1)
        with torch.no_grad():
            overlap = torch.minimum(p, q).sum(-1)
    return actor.masked_fill(~active, 0), overlap.masked_fill(~active, 0)


def _log_probs_and_stats(logits, keep):
    # Mask before reduction: padded rows can contain non-finite logits.
    logits = torch.where(keep, logits.float(), 0)
    maximum = logits.amax(-1, keepdim=True)
    shifted = logits - maximum
    log_scale = shifted.exp().sum(-1, keepdim=True).log()
    return shifted - log_scale, maximum, log_scale


def _masked_vocab_terms(draft, teacher, active, rows=None):
    keep = active[..., None]
    log_q, q_max, q_scale = _log_probs_and_stats(draft, keep)
    if rows is None:
        log_p, p_max, p_scale = _log_probs_and_stats(teacher, keep)
    else:
        log_p = teacher.index_select(0, rows.flatten()).reshape(*rows.shape, -1)
        p_max = p_scale = q_max.new_empty(0)
    log_mix = 0.5 * (log_p + log_q)
    mix_normalizer = log_mix.logsumexp(-1, keepdim=True)
    actor = (-2.0 * mix_normalizer.squeeze(-1)).masked_fill(~active, 0)
    overlap = torch.minimum(log_p, log_q).exp().sum(-1).masked_fill(~active, 0)
    return actor, overlap, q_max, q_scale, p_max, p_scale, mix_normalizer


def _masked_vocab_gradient(draft, teacher, active, q_max, q_scale, p_max, p_scale,
                           mix_normalizer, actor_grad, rows=None):
    keep = active[..., None]
    log_q = (torch.where(keep, draft.float(), 0) - q_max) - q_scale
    log_p = ((torch.where(keep, teacher.float(), 0) - p_max) - p_scale if rows is None
             else teacher.index_select(0, rows.flatten()).reshape(*rows.shape, -1))
    log_mix = 0.5 * (log_p + log_q) - mix_normalizer
    # d D_{1/2}(p || q) / d logits_q = q - normalized(sqrt(p * q)).
    gradient = (log_q.exp() - log_mix.exp()) * actor_grad[..., None]
    return gradient.masked_fill(~keep, 0).to(draft.dtype)


class _MaskedDPALTerms(torch.autograd.Function):
    """Compile fixed-shape vocabulary work and save only inputs and row statistics."""

    @staticmethod
    def forward(ctx, draft, teacher, active, rows=None):
        # Computation starts in FP32, so there are no intermediate low-precision
        # rounding boundaries to emulate. Backward casts only the returned gradient.
        actor, overlap, *stats = _run(_masked_vocab_terms, draft, teacher, active, rows,
                                    emulate_precision_casts=False)
        ctx.save_for_backward(draft, teacher, active, *stats)
        ctx.rows = rows
        ctx.mark_non_differentiable(overlap)
        return actor, overlap

    @staticmethod
    @once_differentiable
    def backward(ctx, actor_grad, _overlap_grad):
        gradient = _run(_masked_vocab_gradient, *ctx.saved_tensors, actor_grad, ctx.rows,
                        emulate_precision_casts=False)
        return (gradient, None, None, None)[:len(ctx.needs_input_grad)]


@torch.no_grad()
def dpal_credit(overlap, mask, rho, *, divide_own_overlap=False):
    """Scan the prediction axis only; invalid slots are multiplicative no-ops."""
    smooth = torch.where(mask.bool(), rho + (1.0 - rho) * overlap.float(), 1.0)
    prefix = smooth.cumprod(-1)
    credit = (prefix * mask).flip([-1]).cumsum(-1).flip([-1])
    return credit / smooth if divide_own_overlap else credit


class DPALRenyiObjective(Objective):
    needs_teacher = True
    metric_names = ("dpal_renyi/local_loss", "dpal_renyi/overlap", "dpal_renyi/credit")

    def __init__(self, recipe, proposal_type="parallel"):
        super().__init__(recipe, proposal_type)
        self.rho = self.recipe["rho"]
        self.local_loss = self.recipe.get("local_loss", "renyi")
        self.divide_own_overlap = self.recipe["type"] == "dpal_plus_plus"
        self.sample_reduction = self.recipe["reduction"] == "sequence_anchor_mean"
        self.requires_complete_blocks = not self.sample_reduction

    def native_denominator(self, layout):
        mask = layout.prediction_mask.bool()
        if self.sample_reduction:
            return mask.any(-1).any(-1).sum().float()
        return mask.all(-1).sum().float()

    def forward(self, prediction, supervision):
        draft, teacher = prediction.draft_logits, supervision.target_logits
        if draft is None or teacher is None:
            raise ValueError("D-PAL requires full-vocabulary draft and frozen teacher logits")
        if getattr(prediction, "candidate_logits", None) is not None:
            raise ValueError("D-PAL does not support candidate-only final proposals")
        mask = supervision.prediction_mask.detach()
        if (mask.ndim != 3 or mask.size(-1) == 0 or draft.shape[:-1] != mask.shape
                or teacher.shape != draft.shape or draft.size(-1) == 0):
            raise ValueError("D-PAL expects matching [samples, anchors, predictions, vocabulary] logits")
        binary = ((mask == 0) | (mask == 1)).all()
        active = mask.bool()
        if self.requires_complete_blocks:
            active = active & active.all(-1, keepdim=True)
        if self.local_loss != "renyi":
            if not binary:
                raise ValueError("D-PAL prediction_mask must be binary")
            actor, overlap = local_actor_and_overlap(draft, teacher, active, self.local_loss)
            if not (torch.isfinite(actor).all() & torch.isfinite(overlap).all()):
                raise ValueError("D-PAL requires a finite local loss on every supervised position")
            local_actor, local_overlap = actor, overlap
        elif isinstance(distribution(teacher, supervision, frozen=True), IndexedTeacherDistribution):
            cached = distribution(teacher, supervision, frozen=True)
            actor, overlap = _MaskedDPALTerms.apply(
                draft, cached.unique_distribution.log_probabilities(), active, cached.rows)
            if not (binary & torch.isfinite(actor).all()):
                raise ValueError("D-PAL requires binary masks and finite supervised losses")
            local_actor, local_overlap = actor, overlap
        elif _should_compile(draft):
            actor, overlap = _MaskedDPALTerms.apply(draft, teacher.detach(), active)
            # Validate both conditions with one host synchronization on success.
            if not (binary & torch.isfinite(actor).all()):
                if not binary:
                    raise ValueError("D-PAL prediction_mask must be binary")
                raise ValueError("D-PAL requires a finite Renyi loss on every supervised position")
            local_actor, local_overlap = actor, overlap
        else:
            if not binary:
                raise ValueError("D-PAL prediction_mask must be binary")
            local_actor, local_overlap = renyi_half_and_overlap(draft[active], teacher[active])
            actor = torch.zeros_like(mask, dtype=torch.float32).masked_scatter(active, local_actor)
            overlap = torch.zeros_like(actor).masked_scatter(active, local_overlap)
        valid = active.float()
        credit = dpal_credit(overlap, valid, self.rho, divide_own_overlap=self.divide_own_overlap)
        block_loss = (actor * credit * valid).sum(-1)
        if self.sample_reduction:
            counts = supervision.sample_anchor_counts
            owners = supervision.block_sample_ids
            if counts is None or (owners is None and counts.shape != (mask.size(0),)):
                raise ValueError("sequence_anchor_mean requires full-layout sample_anchor_counts")
            counts = counts.detach().to(actor)
            actual = active.any(-1).sum(-1) if owners is None else torch.zeros_like(counts).scatter_add(
                0, owners.reshape(-1), active.any(-1).reshape(-1).to(counts))
            if (not torch.isfinite(counts).all() or not (counts == counts.round()).all()
                    or not (counts >= actual).all()):
                raise ValueError("sample_anchor_counts must count all valid anchors before chunking")
            divisor = counts[:, None] if owners is None else counts[owners]
            numerator = (block_loss / divisor.clamp_min(1)).sum()
            denominator = (counts > 0).sum().to(actor)
        else:
            numerator = block_loss.sum()
            denominator = active.all(-1).sum().to(actor)
        token_count = valid.sum().detach()
        metrics = {
            self.metric_names[0]: RatioTerms(local_actor.detach().sum(), token_count),
            self.metric_names[1]: RatioTerms(local_overlap.sum(), token_count),
            self.metric_names[2]: RatioTerms((credit * valid).sum(), token_count),
        }
        terms = RatioTerms(numerator, denominator.detach())
        return TrainingOutput(numerator / denominator.clamp_min(1e-12), terms, metrics)
