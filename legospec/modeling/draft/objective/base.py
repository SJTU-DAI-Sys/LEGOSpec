"""Parameter-free native objective terms; reduction belongs to the runner."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from legospec.modeling.draft.loss_config import normalize_objective
from legospec.modeling.draft.objective.outputs import ObjectiveOutput, RatioTerms
from legospec.modeling.draft.objective.vocab import VocabCache


@dataclass
class Supervision:
    labels: torch.Tensor
    prediction_mask: torch.Tensor
    target_logits: Optional[torch.Tensor] = None
    sample_prediction_counts: Optional[torch.Tensor] = None
    epoch_progress: float = 0.0
    # Full-layout anchor counts, preserved when the anchor axis is chunked.
    sample_anchor_counts: Optional[torch.Tensor] = None
    # Created by a caller for one loss/chunk only. Checkpoint replay gets a fresh cache.
    vocab_cache: Optional[VocabCache] = None
    block_sample_ids: Optional[torch.Tensor] = None


class Objective(nn.Module):
    needs_teacher = False
    sample_reduction = False
    requires_complete_blocks = False
    # Fixed before checkpointed chunk execution; every metric carries num/den.
    metric_names = ()

    def __init__(self, recipe, proposal_type="parallel"):
        super().__init__()
        self.recipe = normalize_objective(recipe)
        self.proposal_type = proposal_type

    def loss_weights(self, supervision):
        return position_weights(
            supervision.prediction_mask, self.recipe.get("loss_decay_gamma")
        )

    def native_denominator(self, layout):
        """Detached layout-only count for independent optimizer-window reductions."""
        if self.sample_reduction:
            return layout.prediction_mask.new_tensor(layout.prediction_mask.size(0))
        return self.loss_weights(layout).sum()

    def output(self, numerator, denominator, **metrics):
        terms = RatioTerms(numerator, denominator.detach().to(numerator))
        ratio_metrics = {
            name: RatioTerms(value.detach(), terms.denominator)
            for name, value in metrics.items()
        }
        return ObjectiveOutput(
            numerator / denominator.clamp_min(1e-12), terms, ratio_metrics
        )


def position_weights(mask, gamma):
    return apply_exponential_position_decay(
        mask.float(),
        gamma,
        torch.arange(mask.size(-1), device=mask.device),
        first_prediction_index=0,
    )


def apply_exponential_position_decay(
    raw_weights: torch.Tensor,
    gamma: Optional[float],
    position_indices: torch.Tensor,
    *,
    first_prediction_index: int = 1,
) -> torch.Tensor:
    """Apply decay using semantic block positions rather than tensor offsets."""

    if gamma is None or gamma <= 0:
        return raw_weights
    positions = position_indices.to(device=raw_weights.device, dtype=torch.float32)
    decay_steps = (positions - int(first_prediction_index)).clamp_min(0)
    shape = (1,) * (raw_weights.ndim - 1) + (-1,)
    return raw_weights * torch.exp(-decay_steps / float(gamma)).view(shape)
