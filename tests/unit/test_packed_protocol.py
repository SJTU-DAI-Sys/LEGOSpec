from pathlib import Path
import tempfile
from types import SimpleNamespace
import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from legospec.modeling.target.components import (
    SGLangTPEmbeddingAdapter,
    SGLangTPLMHeadAdapter,
    set_projection_suppress_tokens,
)
from legospec.modeling.draft.backbone.attention.access.visibility import (
    build_visibility,
)
from legospec.modeling.draft.backbone.attention.access.visibility import (
    indexed_visibility,
)
from legospec.training.execution.supervision import align_objective_blocks


@pytest.mark.parametrize("window", [None, 3])
@pytest.mark.parametrize("causal", [False, True])
def test_indexed_mask_matches_dense_and_padded_tiles(window, causal):
    inputs = SimpleNamespace(
        anchor_positions=torch.tensor([[0, 2, 4, 1]]),
        block_keep_mask=torch.tensor([[True, True, True, False]]),
        anchor_document_ids=torch.tensor([[0, 0, 1, 1]]),
    )
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4]])
    docs = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1]])
    valid = torch.ones_like(pos, dtype=torch.bool)
    dense = build_visibility(
        inputs, pos, valid, docs, 12, window, causal_queries=causal
    )
    live = dense.any(-1)
    dense[..., 0] |= ~live
    callback, _ = indexed_visibility(
        inputs, pos, valid, docs, 12, window, causal_queries=causal
    )
    q = torch.arange(16)[:, None]
    k = torch.arange(32)[None]
    got = callback(0, 0, q, k)
    torch.testing.assert_close(got[:12, :21], dense[0, 0])
    assert not got[12:].any() and not got[:, 21:].any()


class FakeEmbedding(nn.Module):
    def __init__(self, weight, rank):
        super().__init__()
        self.org_vocab_size = weight.size(0)
        self.embedding_dim = weight.size(1)
        self.rank = rank
        self.register_buffer("full", weight)

    def forward(self, ids):
        keep = (ids // 4) == self.rank
        value = self.full[ids] * keep[..., None]
        dist.all_reduce(value)
        return value


class FakeHead(nn.Module):
    def __init__(self, weight, rank):
        super().__init__()
        self.embedding_dim = weight.size(1)
        self.org_vocab_size = weight.size(0)
        self.weight = nn.Parameter(
            weight[rank * 4 : (rank + 1) * 4].clone(), requires_grad=False
        )

    def get_sharded_to_full_mapping(self):
        return None


def worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + path, rank=rank, world_size=2
    )
    try:
        torch.manual_seed(57)
        weight = torch.randn(8, 4)
        embedding = SGLangTPEmbeddingAdapter(
            FakeEmbedding(weight, rank), dist.group.WORLD, 0
        )
        head = SGLangTPLMHeadAdapter(FakeHead(weight, rank), dist.group.WORLD)
        for count in (rank * 3, 0, 2 + rank):
            ids = torch.arange(count)[None] % 8
            torch.testing.assert_close(embedding(ids), weight[ids])
            hidden = torch.randn(count, 4, requires_grad=True)
            actual = head(hidden)
            expected = hidden @ weight.T
            torch.testing.assert_close(actual, expected)
            actual.sum().backward()
            torch.testing.assert_close(hidden.grad, weight.sum(0).expand_as(hidden))
        # The current native head has no mapping helper. Verify full-vocabulary
        # softcap/suppression and ragged backward against an unsharded head.
        native = FakeHead(weight, rank)
        native.get_sharded_to_full_mapping = None
        native.num_added_embeddings = 0
        head = SGLangTPLMHeadAdapter(native, dist.group.WORLD, softcap=3.0)
        set_projection_suppress_tokens(head, [7])
        for count in (rank * 3, 0, 2 + rank):
            hidden = torch.randn(count, 4, requires_grad=True)
            reference = hidden.detach().clone().requires_grad_(True)
            actual = head(hidden)
            expected = (reference @ weight.T / 3.0).tanh() * 3.0
            expected[..., 7] = torch.finfo(expected.dtype).min
            torch.testing.assert_close(actual, expected)
            actual.softmax(-1).square().sum().backward()
            expected.softmax(-1).square().sum().backward()
            torch.testing.assert_close(hidden.grad, reference.grad)
        values = torch.randn(1, 1 + rank * 4, 2, requires_grad=True)
        (aligned,) = align_objective_blocks(values, head=head, chunk_size=2)
        assert (aligned.size(1) + 1) // 2 == 3
        aligned.sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))
    finally:
        dist.destroy_process_group()


def test_tp_ragged_empty_rows_and_chunk_rounds():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "packed_protocol"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as folder:
        mp.spawn(worker, args=(str(Path(folder) / "rendezvous"),), nprocs=2, join=True)
