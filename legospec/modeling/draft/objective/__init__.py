from legospec.modeling.draft.objective.block import DPACEObjective, E2ETVObjective
from legospec.modeling.draft.objective.composition import (
    WeightedSumObjective,
)
from legospec.modeling.draft.objective.dpal import DPALRenyiObjective
from legospec.modeling.draft.objective.token import (
    ForwardKLObjective,
    HardCEObjective,
    L1Objective,
    NLAObjective,
    SoftCEObjective,
    TVObjective,
)

OBJECTIVES = {
    "hard_ce": HardCEObjective,
    "soft_ce": SoftCEObjective,
    "forward_kl": ForwardKLObjective,
    "tv": TVObjective,
    "l1": L1Objective,
    "nla": NLAObjective,
    "lk_alpha": NLAObjective,
    "dpace": DPACEObjective,
    "dpal_renyi": DPALRenyiObjective,
    "e2e_tv": E2ETVObjective,
    "weighted_sum": WeightedSumObjective,
}


def build_objective(recipe, proposal_type="parallel"):
    return OBJECTIVES[recipe["type"]](recipe, proposal_type)
