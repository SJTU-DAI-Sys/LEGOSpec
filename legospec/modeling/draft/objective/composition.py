from __future__ import annotations

"""Configured base/final branches for residual and strict top-K Selectors."""

from copy import deepcopy
from dataclasses import replace

from torch import nn

from legospec.modeling.draft.loss_config import (
    SELECTOR_TYPES,
    normalize_training_recipe,
)
from legospec.modeling.draft.objective.token import CandidateCEObjective


class LossBranches(nn.Module):
    def __init__(self, recipe):
        super().__init__()
        from legospec.modeling.draft.objective import build_objective

        self.recipe = normalize_training_recipe(recipe)
        prediction_type = self.recipe.get("proposal", {}).get("type", "autoregressive")
        self.is_selector = prediction_type in SELECTOR_TYPES
        outer = self.recipe["objective"]
        definitions = (
            [("final", outer["final"]), ("base", outer["base"])]
            if self.is_selector
            else [("final", {**outer, "weight": 1.0})]
        )
        self.weights = {}
        self.objectives = nn.ModuleDict()
        for name, definition in definitions:
            item = deepcopy(definition)
            weight = item.pop("weight")
            self.weights[name] = weight
            if not weight:
                continue
            item["loss_decay_gamma"] = outer["loss_decay_gamma"]
            proposal_type = (
                "parallel"
                if name == "base"
                else "selector_full_vocab"
                if self.is_selector
                else prediction_type
            )
            if (
                self.is_selector
                and name == "final"
                and self.recipe["proposal"]["selector_mode"] == "candidate_only"
            ):
                self.objectives[name] = CandidateCEObjective(item, "selector_topk")
            else:
                self.objectives[name] = build_objective(item, proposal_type)
        self.needs_teacher = any(o.needs_teacher for o in self.objectives.values())
        self.requires_complete_blocks = any(
            o.requires_complete_blocks for o in self.objectives.values()
        )

    def branch_output(self, name, prediction, supervision):
        if name == "base":
            prediction = replace(
                prediction,
                draft_logits=prediction.base_logits,
                candidate_logits=None,
                candidate_ids=None,
            )
        return self.objectives[name](prediction, supervision)

    def native_denominators(self, layout):
        zero = layout.prediction_mask.new_zeros(())
        return {
            name: (
                self.objectives[name].native_denominator(layout)
                if name in self.objectives
                else zero
            )
            for name in self.weights
        }


"""Weighted token objectives with a shared mask, decay, and denominator."""

import math

import torch

from legospec.modeling.draft.objective.base import Objective
from legospec.modeling.draft.objective.token import (
    ForwardKLObjective,
    HardCEObjective,
    L1Objective,
    NLAObjective,
    SoftCEObjective,
    TVObjective,
)
from legospec.modeling.draft.objective.vocab import VocabCache


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
                if (
                    prediction.draft_logits.requires_grad
                    and prediction.draft_logits.dtype != torch.float32
                ):
                    component_supervision = replace(
                        supervision, vocab_cache=supervision.vocab_cache.fork_draft()
                    )
                term = (
                    weight
                    * objective(prediction, component_supervision).loss_terms.numerator
                )
                numerator = term if numerator is None else numerator + term
        return self.output(numerator, self.loss_weights(supervision).sum())
