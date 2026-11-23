"""Shared LEGOSpec training entry."""


def run_training():
    from legospec.training.runner import configure_rank_caches

    configure_rank_caches()
    from legospec.training.config import parse_args
    from legospec.training.runner import run_training as run
    from torch.distributed.elastic.multiprocessing.errors import record

    record(run)(parse_args())
