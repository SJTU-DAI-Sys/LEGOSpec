"""Shared occurrence and packed-layout contracts.

Costs are analytic FLOP work estimates until a non-negative stage profile is
provided. A sampler never commits progress while DataLoader workers prefetch.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch


@dataclass(frozen=True)
class SamplePlan:
    occurrence_id: str
    sample_id: int
    length: int
    anchors: tuple[int, ...]
    supervision_positions: tuple[tuple[int, ...], ...] = ()
    teacher_positions: tuple[int, ...] = ()
    teacher_inverse: tuple[int, ...] = ()
    cost_features: dict[str, float] = field(default_factory=dict)
    shard: str | None = None
    source_position: int = 0

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("anchors", "teacher_positions", "teacher_inverse"):
            value[key] = tuple(value.get(key, ()))
        value["supervision_positions"] = tuple(
            tuple(x) for x in value.get("supervision_positions", ())
        )
        return cls(**value)


@dataclass(frozen=True)
class PackedBatchPlan:
    samples: tuple[SamplePlan, ...]
    context_offsets: torch.Tensor
    block_offsets: torch.Tensor
    block_sample_ids: torch.Tensor
    anchor_positions: torch.Tensor
    query_offsets: torch.Tensor
    query_position_ids: torch.Tensor
    context_position_ids: torch.Tensor
    kept_block_indices: torch.Tensor
    block_size: int
    query_len: int
    microbatches: tuple[tuple[int, int], ...] = ()
    routes: tuple = ()
    supervision_mask: torch.Tensor | None = None
    teacher_indices: torch.Tensor | None = None

    @property
    def block_to_sample(self):
        return self.block_sample_ids

    @property
    def anchor_positions_batched(self):
        return self.anchor_positions.unsqueeze(0)

    @property
    def position_ids(self):
        return self.query_position_ids.reshape(1, -1)

    @property
    def lengths(self):
        return self.context_offsets[1:] - self.context_offsets[:-1]

    @property
    def num_blocks(self):
        return self.anchor_positions.numel()

    @property
    def num_samples(self):
        return self.context_offsets.numel() - 1

    def to(self, device):
        return replace(
            self,
            **{
                name: value.to(device=device)
                for name, value in vars(self).items()
                if isinstance(value, torch.Tensor)
            },
        )

    def pin_memory(self):
        return replace(
            self,
            **{
                name: value.pin_memory()
                for name, value in vars(self).items()
                if isinstance(value, torch.Tensor)
            },
        )

    def context_gather_index(self, padded_length: int):
        pieces = [
            torch.arange(int(n), device=self.context_offsets.device)
            + i * int(padded_length)
            for i, n in enumerate(p.length for p in self.samples)
        ]
        return torch.cat(pieces) if pieces else self.context_offsets.new_empty(0)

    def pack_context(self, tensor):
        """Pack a batch-first padded tensor, keeping a singleton batch axis."""
        if tensor.shape[0] == 1 and tensor.shape[1] == sum(
            p.length for p in self.samples
        ):
            return tensor
        if tensor.shape[0] != self.num_samples:
            raise ValueError("context batch does not match packed plan")
        index = self.context_gather_index(tensor.shape[1]).to(tensor.device)
        return tensor.flatten(0, 1).index_select(0, index).unsqueeze(0)

    def select_samples(self, start: int, end: int):
        lengths = self.lengths[start:end].tolist()
        rows = []
        for sample in range(start, end):
            rows.append(self.anchor_positions[self.block_sample_ids == sample].tolist())
        width = max(map(len, rows), default=0)
        anchors = torch.zeros((len(rows), width), dtype=torch.long)
        keep = torch.zeros_like(anchors, dtype=torch.bool)
        for i, values in enumerate(rows):
            anchors[i, : len(values)] = torch.tensor(values, dtype=torch.long)
            keep[i, : len(values)] = True
        return build_packed_batch_plan(
            lengths,
            anchors,
            keep,
            self.block_size,
            query_len=self.query_len,
            samples=self.samples[start:end],
            device=self.context_offsets.device,
        )


def build_packed_batch_plan(
    lengths,
    anchor_positions,
    block_keep_mask,
    block_size,
    *,
    occurrence_ids=None,
    query_len=None,
    device=None,
    samples=None,
    microbatch_size=None,
):
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=device).reshape(-1)
    anchors = torch.as_tensor(anchor_positions, dtype=torch.long, device=lengths.device)
    keep = torch.as_tensor(block_keep_mask, dtype=torch.bool, device=lengths.device)
    if (
        anchors.ndim != 2
        or anchors.shape != keep.shape
        or anchors.shape[0] != lengths.numel()
    ):
        raise ValueError("anchors/keep must have shape [samples, anchor_capacity]")
    if bool((lengths < 0).any()):
        raise ValueError("negative sequence length")
    kept = keep.flatten().nonzero(as_tuple=False).flatten()
    owner = (
        torch.arange(lengths.numel(), device=lengths.device)
        .unsqueeze(1)
        .expand_as(anchors)
        .flatten()[kept]
    )
    local = anchors.flatten()[kept]
    if bool(((local < 0) | (local >= lengths[owner])).any()):
        raise ValueError("anchor lies outside its real sequence")
    counts = keep.sum(1)
    zeros = lengths.new_zeros(1)
    context_offsets = torch.cat((zeros, lengths.cumsum(0)))
    block_offsets = torch.cat((zeros, counts.cumsum(0)))
    query_len = int(query_len or block_size)
    query_positions = local[:, None] + torch.arange(query_len, device=lengths.device)
    context_positions = (
        torch.cat(
            [torch.arange(int(n), device=lengths.device) for n in lengths.tolist()]
        )
        if lengths.numel()
        else zeros[:0]
    )
    if samples is None:
        occurrence_ids = occurrence_ids or [str(i) for i in range(lengths.numel())]
        samples = tuple(
            SamplePlan(
                str(occurrence_ids[i]), i, int(n), tuple(anchors[i][keep[i]].tolist())
            )
            for i, n in enumerate(lengths.tolist())
        )
    microbatches = (
        ()
        if microbatch_size is None
        else tuple(
            (i, min(i + microbatch_size, len(samples)))
            for i in range(0, len(samples), microbatch_size)
        )
    )
    return PackedBatchPlan(
        tuple(samples),
        context_offsets,
        block_offsets,
        owner,
        local,
        block_offsets * query_len,
        query_positions,
        context_positions,
        kept,
        int(block_size),
        query_len,
        microbatches,
    )


def anchors_from_sample_plans(samples, *, device=None, capacity=None):
    width = (
        max((len(p.anchors) for p in samples), default=0)
        if capacity is None
        else int(capacity)
    )
    anchors = torch.zeros((len(samples), width), dtype=torch.long, device=device)
    keep = torch.zeros_like(anchors, dtype=torch.bool)
    for row, plan in enumerate(samples):
        n = len(plan.anchors)
        if n > width:
            raise ValueError("anchor capacity would discard planned blocks")
        anchors[row, :n] = torch.as_tensor(
            plan.anchors, dtype=torch.long, device=device
        )
        keep[row, :n] = True
    return anchors, keep

