"""CPU gloo verification of the existing window normalizer with D-PAL."""

from copy import deepcopy
from pathlib import Path
import tempfile

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from legospec.modeling.draft.objective import build_objective
from legospec.modeling.draft.objective.base import Supervision
from legospec.modeling.draft.contracts import ProposalOutput
from legospec.training.optim.optimizer import normalize_accumulated_gradients


class LossModule(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.head = nn.Linear(5, 11, bias=False)
        self.objective = build_objective(dict(type="dpal_renyi", reduction=reduction))

    def forward(self, x, p, mask):
        q = self.head(x)
        out = self.objective(
            ProposalOutput(x, draft_logits=q),
            Supervision(
                torch.zeros_like(mask, dtype=torch.long),
                mask,
                p,
                sample_anchor_counts=mask.bool().any(-1).sum(-1),
            ),
        )
        return out.loss_terms.numerator, out.loss_terms.denominator


class GradientScaler:
    def __init__(self, parameters):
        self.parameters = list(parameters)

    def scale_model_gradients(self, scale):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.mul_(scale)


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2
    )
    try:
        for reduction in ("complete_block_mean", "sequence_anchor_mean"):
            torch.manual_seed(721)
            model = LossModule(reduction)
            reference = deepcopy(model)
            ddp = DistributedDataParallel(model)
            x, p = torch.randn(4, 3, 7, 5), torch.randn(4, 3, 7, 11)
            mask = torch.ones(4, 3, 7)
            mask[:2] = 0  # rank 0 has zero supervision in the entire window
            mask[2, 1:] = 0
            if reduction == "sequence_anchor_mean":
                mask[3, 1, 3:] = 0
            full_num, full_den = reference(x, p, mask)
            (full_num / full_den).backward()
            local_den = torch.tensor(0.0)
            for i in range(2):
                j = rank * 2 + i
                num, den = ddp(x[j : j + 1], p[j : j + 1], mask[j : j + 1])
                (num / 2).backward()
                local_den += den
            got_den = normalize_accumulated_gradients(
                GradientScaler(model.parameters()), local_den, 2
            )
            assert got_den == full_den.item()
            for a, b in zip(model.parameters(), reference.parameters()):
                assert torch.isfinite(a.grad).all()
                torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-7)
    finally:
        dist.destroy_process_group()


def test_cpu_ddp_uneven_counts_zero_rank_and_accumulation():
    folder = Path(__file__).resolve().parents[2] / "tmp_exp" / "artifacts" / "dpal_distributed"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as d:
        mp.spawn(worker, args=(str(Path(d) / "rendezvous"),), nprocs=2, join=True)


def test_empty_global_window_still_fails():
    with pytest.raises(ValueError, match="finite and positive"):
        normalize_accumulated_gradients(GradientScaler([]), torch.tensor(0.0), 2)
