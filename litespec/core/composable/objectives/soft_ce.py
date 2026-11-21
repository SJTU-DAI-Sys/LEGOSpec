from .base import Objective
from ..vocab import distribution
from litespec.core.losses import _run


def _probability_log_dot(probabilities, log_probabilities):
    # Fuse only multiply/reduce. Native log-softmax and its backward retain
    # their rounding boundaries; no full-vocabulary product needs storage.
    return (probabilities * log_probabilities).sum(-1)


class SoftCEObjective(Objective):
    needs_teacher = True

    def forward(self, prediction, supervision):
        if supervision.target_logits is None:
            raise ValueError("Soft objective requires frozen teacher logits")
        draft_log_probabilities = distribution(prediction.draft_logits, supervision).log_probabilities()
        teacher = distribution(supervision.target_logits, supervision, frozen=True)
        target_log_probabilities = teacher.log_probabilities()
        target_probabilities = teacher.exp_log_probabilities()
        token_losses = -_run(_probability_log_dot, target_probabilities, draft_log_probabilities)
        if self.recipe["type"] == "forward_kl":
            token_losses = token_losses + _run(
                _probability_log_dot, target_probabilities, target_log_probabilities)
        loss_weights = self.loss_weights(supervision)
        return self.output((token_losses * loss_weights).sum(), loss_weights.sum())
