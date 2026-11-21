from .base import Objective
from ..vocab import distribution


class TVObjective(Objective):
    needs_teacher = True

    def forward(self, prediction, supervision):
        draft_probabilities = distribution(prediction.draft_logits, supervision).probabilities()
        target_probabilities = distribution(
            supervision.target_logits, supervision, frozen=True
        ).probabilities()
        # Standard total variation is half the full distribution L1 distance.
        token_losses = 0.5 * (target_probabilities - draft_probabilities).abs().sum(-1)
        loss_weights = self.loss_weights(supervision)
        return self.output((token_losses * loss_weights).sum(), loss_weights.sum())
