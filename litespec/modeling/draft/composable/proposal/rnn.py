from ...markov import RNNMarkovHead
from .markov import MarkovProposal


class RNNProposal(MarkovProposal):
    head_class = RNNMarkovHead
