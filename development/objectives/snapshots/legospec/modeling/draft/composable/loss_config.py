"""Normalize objective weights, reductions and proposal-specific loss settings."""

from copy import deepcopy
import math

# Checkpoint semantic identity for the current normalized objective contract:
# objective-owned decay, explicit Selector base/final losses and reduction rules.
# This selects no implementation; resume rejects a missing or different marker.
# Keep the existing value so current checkpoints retain their semantic identity.
LOSS_CONTRACT_VERSION = 2
DEFAULT_LOSS_DECAY_GAMMA = 4.0
TOKEN_OBJECTIVES = frozenset(
    ("hard_ce", "soft_ce", "forward_kl", "tv", "l1", "nla", "lk_alpha")
)
DPAL_OBJECTIVES = frozenset(("dpal_renyi", "dpal_plus_plus"))
BLOCK_OBJECTIVES = DPAL_OBJECTIVES | {"dpace", "tv_survival", "opd_survival", "e2e_tv"}
SAMPLED_ANCHOR_OBJECTIVES = frozenset(("opd_survival", "e2e_tv"))
OBJECTIVES = TOKEN_OBJECTIVES | BLOCK_OBJECTIVES | {"weighted_sum"}
SELECTOR_TYPES = frozenset(("selector", "residual_selector", "topk_selector"))


def selector_mode(proposal):
    """The historical `selector` name remains a full-vocabulary alias."""
    kind = proposal.get("type")
    default = "candidate_only" if kind == "topk_selector" else "full_vocab_residual"
    mode = proposal.get("selector_mode", default)
    if mode not in ("candidate_only", "full_vocab_residual"):
        raise ValueError("Unknown Selector support mode")
    if kind in ("residual_selector", "topk_selector") and mode != default:
        raise ValueError("Selector name and support mode disagree")
    return mode


def resolve_gamma(recipe):
    gamma = recipe.get("loss_decay_gamma", DEFAULT_LOSS_DECAY_GAMMA)
    if gamma is None:
        return None
    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, (int, float))
        or not math.isfinite(gamma)
        or gamma <= 0
    ):
        raise ValueError(
            "objective.loss_decay_gamma must be a finite positive number or null"
        )
    return float(gamma)


def loss_weight(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Loss weight must be finite and nonnegative")
    return float(value)


def normalize_objective(recipe):
    if not isinstance(recipe, dict) or recipe.get("type") not in OBJECTIVES:
        raise ValueError("Unsupported composable objective: " + str(recipe))
    result = deepcopy(recipe)
    if result["type"] == "e2e_tv":
        result.setdefault("loss_decay_gamma", None)
        if result["loss_decay_gamma"] is not None:
            raise ValueError("E2E-TV requires loss_decay_gamma=null")
    if result["type"] in DPAL_OBJECTIVES:
        result.setdefault("loss_decay_gamma", None)
        if result["loss_decay_gamma"] is not None:
            raise ValueError(
                "D-PAL requires loss_decay_gamma=null; its position weights are dynamic"
            )
        rho = result.get("rho", 0.3)
        if (
            isinstance(rho, bool)
            or not isinstance(rho, (int, float))
            or not math.isfinite(rho)
            or not 0 <= rho <= 1
        ):
            raise ValueError("D-PAL rho must be finite and in [0,1]")
        if result["type"] == "dpal_plus_plus" and rho == 0:
            raise ValueError(
                "D-PAL++ requires rho > 0 so the smoothed overlap is positive"
            )
        result["rho"] = float(rho)
        local_loss = result.get("local_loss", "renyi")
        if local_loss not in ("renyi", "forward_kl", "tv"):
            raise ValueError("D-PAL local_loss must be renyi, forward_kl, or tv")
        if "local_loss" in result:
            result["local_loss"] = local_loss
        reduction = result.get("reduction", "complete_block_mean")
        if reduction not in ("complete_block_mean", "sequence_anchor_mean"):
            raise ValueError(
                "D-PAL reduction must be complete_block_mean or sequence_anchor_mean"
            )
        result["reduction"] = reduction
    result["loss_decay_gamma"] = resolve_gamma(result)
    if result["type"] == "opd_survival":
        # These are independent block objectives; no fixed token decay is used.
        result["beta_rej"] = loss_weight(result.get("beta_rej", 1.0))
        result["survival_weight"] = loss_weight(result.get("survival_weight", 1.0))
        decay = result.get("rejection_decay", 0.8)
        if (
            isinstance(decay, bool)
            or not isinstance(decay, (int, float))
            or not math.isfinite(decay)
            or not 0 < decay <= 1
        ):
            raise ValueError("objective.rejection_decay must be finite and in (0, 1]")
        result["rejection_decay"] = float(decay)
    if result["type"] == "dpace":
        reduction = result.get("reduction", "complete_block_mean")
        if reduction != "complete_block_mean":
            raise ValueError(
                "D-PACE requires reduction=complete_block_mean; set objective.reduction explicitly"
            )
        result["reduction"] = reduction
    if result["type"] == "weighted_sum":
        components = result.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError("weighted_sum requires non-empty components")
        for component in components:
            if (
                not isinstance(component, dict)
                or component.get("type") not in TOKEN_OBJECTIVES
            ):
                raise ValueError("weighted_sum requires token-normalized components")
            if "loss_decay_gamma" in component:
                raise ValueError(
                    "Set loss_decay_gamma only on the outer objective, not components"
                )
            component["weight"] = loss_weight(component.get("weight"))
        if not any(c["weight"] for c in components):
            raise ValueError("At least one component weight must be positive")
    return result


def normalize_training_recipe(recipe):
    result = deepcopy(recipe)
    proposal = result.get("proposal", {})
    if "loss_decay_gamma" in proposal:
        raise ValueError(
            "proposal.loss_decay_gamma was removed; migrate to objective.loss_decay_gamma"
        )
    objective = result["objective"]
    if proposal.get("type") not in SELECTOR_TYPES:
        result["objective"] = normalize_objective(objective)
        return result
    proposal["selector_mode"] = selector_mode(proposal)
    if objective.get("type") != "selector_losses":
        raise ValueError(
            "Selector training requires objective.type=selector_losses with base and final objectives"
        )
    gamma = resolve_gamma(objective)
    branches = {"type": "selector_losses", "loss_decay_gamma": gamma}
    for name, default_weight in (("base", 0.1), ("final", 1.0)):
        branch = deepcopy(objective.get(name, {"type": "hard_ce"}))
        if "loss_decay_gamma" in branch:
            raise ValueError(
                "Set loss_decay_gamma only on objective, not its " + name + " branch"
            )
        weight = loss_weight(branch.pop("weight", default_weight))
        branch = normalize_objective({**branch, "loss_decay_gamma": gamma})
        branch.pop("loss_decay_gamma")
        branches[name] = {**branch, "weight": weight}
    if not any(branches[name]["weight"] for name in ("base", "final")):
        raise ValueError("At least one Selector branch weight must be positive")
    if (
        proposal["selector_mode"] == "candidate_only"
        and branches["final"]["type"] != "hard_ce"
    ):
        raise ValueError("TopKSelector currently trains final candidate CE only")
    result["objective"] = branches
    return result
