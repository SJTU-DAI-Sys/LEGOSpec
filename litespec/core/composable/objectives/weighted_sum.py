"""Weighted token objectives with a shared mask, decay, and denominator."""

import math
from dataclasses import replace

import torch
from torch import nn

from .base import Objective
from .ce import HardCEObjective
from .kl import ForwardKLObjective
from .soft_ce import SoftCEObjective
from .tv import TVObjective
from .l1 import L1Objective
from .nla import NLAObjective
from ..vocab import VocabCache


class WeightedSumObjective(Objective):
    # Sample/path objectives have different native reductions and cannot be
    # inserted into this token-normalized sum without defining a new reduction.
    COMPONENTS = {
        "hard_ce": HardCEObjective,
        "soft_ce": SoftCEObjective,
        "forward_kl": ForwardKLObjective,
        "tv": TVObjective,
        "l1": L1Objective,
        "nla": NLAObjective,
        "lk_alpha": NLAObjective,
    }

    def __init__(self, recipe, proposal_type="parallel"):
        super().__init__(recipe, proposal_type)
        recipe = self.recipe
        components = recipe.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError("weighted_sum requires non-empty components")
        objectives, weights = [], []
        for component in components:
            item = dict(component)
            kind = item.get("type")
            if kind not in self.COMPONENTS:
                raise ValueError("weighted_sum requires token-normalized components")
            weight = float(item.pop("weight"))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Component weights must be finite and nonnegative")
            item["loss_decay_gamma"] = recipe["loss_decay_gamma"]
            objectives.append(self.COMPONENTS[kind](item, proposal_type))
            weights.append(weight)
        if not any(weights):
            raise ValueError("At least one component weight must be positive")
        self.components = nn.ModuleList(objectives)
        self.weights = tuple(weights)
        self.needs_teacher = any(
            weight > 0 and objective.needs_teacher
            for weight, objective in zip(self.weights, self.components)
        )

    def forward(self, prediction, supervision):
        if supervision.vocab_cache is None:
            supervision = replace(supervision, vocab_cache=VocabCache())
        numerator = None
        for weight, objective in zip(self.weights, self.components):
            if weight:
                component_supervision = supervision
                if prediction.draft_logits.requires_grad and prediction.draft_logits.dtype != torch.float32:
                    component_supervision = replace(
                        supervision, vocab_cache=supervision.vocab_cache.fork_draft())
                term = weight * objective(prediction, component_supervision).loss_terms.numerator
                numerator = term if numerator is None else numerator + term
        return self.output(numerator, self.loss_weights(supervision).sum())
