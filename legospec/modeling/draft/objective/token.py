from __future__ import annotations

import torch

from legospec.modeling.draft.objective.base import Objective
from legospec.modeling.draft.objective.vocab import distribution


class HardCEObjective(Objective):
    def forward(self, prediction, supervision):
        token_losses = distribution(prediction.draft_logits, supervision).cross_entropy(
            supervision.labels
        )
        loss_weights = self.loss_weights(supervision)
        return self.output((token_losses * loss_weights).sum(), loss_weights.sum())


from legospec.modeling.draft.objective.kernels import _run


def _probability_log_dot(probabilities, log_probabilities):
    # Fuse only multiply/reduce. Native log-softmax and its backward retain
    # their rounding boundaries; no full-vocabulary product needs storage.
    return (probabilities * log_probabilities).sum(-1)


class SoftCEObjective(Objective):
    needs_teacher = True

    def forward(self, prediction, supervision):
        if supervision.target_logits is None:
            raise ValueError("Soft objective requires frozen teacher logits")
        draft_log_probabilities = distribution(
            prediction.draft_logits, supervision
        ).log_probabilities()
        teacher = distribution(supervision.target_logits, supervision, frozen=True)
        target_log_probabilities = teacher.log_probabilities()
        target_probabilities = teacher.exp_log_probabilities()
        token_losses = -_run(
            _probability_log_dot, target_probabilities, draft_log_probabilities
        )
        if self.recipe["type"] == "forward_kl":
            token_losses = token_losses + _run(
                _probability_log_dot, target_probabilities, target_log_probabilities
            )
        loss_weights = self.loss_weights(supervision)
        return self.output((token_losses * loss_weights).sum(), loss_weights.sum())


class ForwardKLObjective(SoftCEObjective):
    pass


class TVObjective(Objective):
    needs_teacher = True

    def forward(self, prediction, supervision):
        draft_probabilities = distribution(
            prediction.draft_logits, supervision
        ).probabilities()
        target_probabilities = distribution(
            supervision.target_logits, supervision, frozen=True
        ).probabilities()
        # Standard total variation is half the full distribution L1 distance.
        token_losses = 0.5 * (target_probabilities - draft_probabilities).abs().sum(-1)
        loss_weights = self.loss_weights(supervision)
        return self.output((token_losses * loss_weights).sum(), loss_weights.sum())


"""Full distribution L1 distance, with the shared weighted-token reduction."""


class L1Objective(TVObjective):
    def forward(self, prediction, supervision):
        output = super().forward(prediction, supervision)
        return self.output(
            2.0 * output.loss_terms.numerator, output.loss_terms.denominator
        )


"""DFlash2 categorical CE on strict unary top-K support.

Gold tokens outside the support contribute zero Selector CE. The denominator
remains ALL valid weighted prediction tokens, matching the combined DFlash2
base + Selector training denominator, rather than reweighting by recall.
"""
from torch.nn import functional as F


class CandidateCEObjective(Objective):
    def forward(self, prediction, supervision):
        scores = prediction.candidate_logits.float()
        matches = prediction.candidate_ids == supervision.labels[..., None]
        covered = matches.any(-1)
        local_labels = matches.long().argmax(-1)
        losses = F.cross_entropy(
            scores.flatten(0, -2), local_labels.flatten(), reduction="none"
        ).reshape_as(supervision.prediction_mask)
        weights = self.loss_weights(supervision)
        # Multiplication retains a zero autograd path even for all-miss batches.
        return self.output(
            (losses * weights * covered).sum(),
            weights.sum(),
            candidate_recall=(weights * covered).sum(),
        )


# Per-token negative log overlap, normalized by valid decayed token weights.
class NLAObjective(Objective):
    needs_teacher = True

    def forward(self, prediction, supervision):
        draft_probabilities = distribution(
            prediction.draft_logits, supervision
        ).probabilities()
        target_probabilities = distribution(
            supervision.target_logits, supervision, frozen=True
        ).probabilities()
        token_losses = (
            -torch.minimum(target_probabilities, draft_probabilities)
            .sum(-1)
            .clamp_min(1e-5)
            .log()
        )
        weights = self.loss_weights(supervision)
        return self.output((token_losses * weights).sum(), weights.sum())
