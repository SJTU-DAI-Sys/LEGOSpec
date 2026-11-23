from legospec.modeling.draft.composable.objective.token import HardCEObjective
from legospec.modeling.draft.composable.objective.block import DPACEObjective
from legospec.modeling.draft.composable.objective.dpal import DPALRenyiObjective
from legospec.modeling.draft.composable.objective.block import E2ETVObjective
from legospec.modeling.draft.composable.objective.token import ForwardKLObjective
from legospec.modeling.draft.composable.objective.token import NLAObjective
from legospec.modeling.draft.composable.objective.token import SoftCEObjective
from legospec.modeling.draft.composable.objective.token import TVObjective
from legospec.modeling.draft.composable.objective.token import L1Objective
from legospec.modeling.draft.composable.objective.survival import TVSurvivalObjective
from legospec.modeling.draft.composable.objective.survival import OPDSurvivalObjective
from legospec.modeling.draft.composable.objective.composition import (
    WeightedSumObjective,
)

OBJECTIVES = {
    "hard_ce": HardCEObjective,
    "soft_ce": SoftCEObjective,
    "forward_kl": ForwardKLObjective,
    "tv": TVObjective,
    "l1": L1Objective,
    "tv_survival": TVSurvivalObjective,
    "opd_survival": OPDSurvivalObjective,
    "nla": NLAObjective,
    "lk_alpha": NLAObjective,
    "dpace": DPACEObjective,
    "dpal_renyi": DPALRenyiObjective,
    "dpal_plus_plus": DPALRenyiObjective,
    "e2e_tv": E2ETVObjective,
    "weighted_sum": WeightedSumObjective,
}


def build_objective(recipe, proposal_type="parallel"):
    return OBJECTIVES[recipe["type"]](recipe, proposal_type)
