from .base import Objective
from ..vocab import distribution


class HardCEObjective(Objective):
    def forward(self, prediction, supervision):
        token_losses = distribution(prediction.draft_logits, supervision).cross_entropy(supervision.labels)
        loss_weights = self.loss_weights(supervision)
        return self.output((token_losses * loss_weights).sum(), loss_weights.sum())
