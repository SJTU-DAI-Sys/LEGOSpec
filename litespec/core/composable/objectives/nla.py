"""Negative log overlap, globally normalized by decayed token weights."""

import torch

from .base import Objective
from ..vocab import distribution


class NLAObjective(Objective):
    needs_teacher = True

    def forward(self, prediction, supervision):
        draft_probabilities = distribution(prediction.draft_logits, supervision).probabilities()
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
