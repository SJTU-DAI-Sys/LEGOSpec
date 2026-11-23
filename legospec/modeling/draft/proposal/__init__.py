from legospec.modeling.draft.proposal.markov import MarkovProposal, RNNProposal
from legospec.modeling.draft.proposal.parallel import ParallelProposal
from legospec.modeling.draft.proposal.selector import SelectorProposal
from legospec.modeling.draft.proposal.sequential import SequentialRNNProposal

PROPOSALS = {
    "parallel": ParallelProposal,
    "markov": MarkovProposal,
    "rnn": RNNProposal,
    "sequential_rnn": SequentialRNNProposal,
    "selector": SelectorProposal,
    "residual_selector": SelectorProposal,
    "topk_selector": SelectorProposal,
}


def build_proposal(config):
    return PROPOSALS[config.composable_config["proposal"]["type"]](config)
