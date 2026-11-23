"""Shared LEGOSpec training entry."""

from __future__ import annotations


def run_training():
    from legospec.training.runtime.runner import configure_rank_caches

    configure_rank_caches()
    from torch.distributed.elastic.multiprocessing.errors import record

    from legospec.training.config import parse_args
    from legospec.training.runtime.runner import run_training as run

    record(run)(parse_args())
