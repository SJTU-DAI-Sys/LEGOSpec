"""Reusable row selection for chunked vocabulary objectives.

Source indices identify frozen prediction states, not labels or token values.
A caller supplies occurrence-aware absolute indices (e.g. packed context rows).
Only indices and hidden states survive between objective chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch


@dataclass(frozen=True)
class SupervisionRows:
    logical_shape: tuple[int, ...]
    active_index: torch.Tensor
    projection_index: torch.Tensor
    active_to_projection: torch.Tensor
    unique_teacher_index: Optional[torch.Tensor] = None
    teacher_inverse: Optional[torch.Tensor] = None
    teacher_first_index: Optional[torch.Tensor] = None

    def to(self, device):
        return replace(
            self,
            **{
                key: value.to(device, non_blocking=True)
                for key, value in vars(self).items()
                if isinstance(value, torch.Tensor)
            },
        )

    @classmethod
    def from_weights(cls, weights, projection_mask=None, teacher_source_index=None):
        active = weights.reshape(-1) != 0
        projection = (
            active
            if projection_mask is None
            else (projection_mask.reshape(-1).bool() | active)
        )
        active_index = active.nonzero(as_tuple=False).flatten()
        projection_index = projection.nonzero(as_tuple=False).flatten()
        active_to_projection = torch.searchsorted(projection_index, active_index)
        unique = inverse = first = None
        if teacher_source_index is not None:
            if tuple(teacher_source_index.shape) != tuple(weights.shape):
                raise ValueError(
                    "teacher source indices must have the supervision shape"
                )
            sources = teacher_source_index.reshape(-1).index_select(0, active_index)
            unique, inverse = torch.unique(sources, sorted=True, return_inverse=True)
            first = active_index.new_full((unique.numel(),), active_index.numel())
            first.scatter_reduce_(
                0,
                inverse,
                torch.arange(active_index.numel(), device=active_index.device),
                reduce="amin",
                include_self=True,
            )
            first = active_index.index_select(0, first)
        return cls(
            tuple(weights.shape),
            active_index,
            projection_index,
            active_to_projection,
            unique,
            inverse,
            first,
        )

    def _flat(self, tensor):
        rank = len(self.logical_shape)
        if tuple(tensor.shape[:rank]) != self.logical_shape:
            raise ValueError(
                f"expected logical shape {self.logical_shape}, got {tuple(tensor.shape)}"
            )
        return tensor.reshape(-1, *tensor.shape[rank:])

    def gather_active(self, tensor):
        return self._flat(tensor).index_select(0, self.active_index)

    def gather_projection(self, tensor):
        return self._flat(tensor).index_select(0, self.projection_index)

    def scatter_active(self, values, fill_value=0):
        return self._scatter(values, self.active_index, fill_value)

    def scatter_projection(self, values, fill_value=0):
        return self._scatter(values, self.projection_index, fill_value)

    def _scatter(self, values, index, fill_value):
        size = 1
        for width in self.logical_shape:
            size *= width
        output = values.new_full((size, *values.shape[1:]), fill_value)
        return output.index_copy(0, index, values).reshape(
            *self.logical_shape, *values.shape[1:]
        )

    @torch.no_grad()
    def project_teacher(self, aligned_hidden, head):
        """Project each unique frozen hidden once; return active occurrence order."""
        if self.teacher_first_index is None or self.teacher_inverse is None:
            raise ValueError("teacher_source_index is required for teacher projection")
        hidden = self._flat(aligned_hidden).index_select(0, self.teacher_first_index)
        logits = head(hidden)
        return logits.index_select(0, self.teacher_inverse)


__all__ = ["SupervisionRows"]


def align_objective_blocks(*tensors, head, chunk_size, dim=1):
    """Match TP collective rounds without adding backbone or projection rows.

    Synthetic blocks carry zero weights. Their empty row projections retain an
    autograd edge, so short ranks also enter TP backward and checkpoint replay.
    """
    import torch.distributed as dist

    group = getattr(head, "tp_group", None)
    tp_size = int(getattr(head, "tp_size", 1))
    if tp_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return tensors
    first = next(value for value in tensors if value is not None)
    width = first.shape[dim]
    chunk = int(chunk_size) or max(width, 1)
    # chunk_size=0 means one collective round on every rank already.
    if chunk_size == 0:
        return tensors
    rounds = torch.tensor(
        (width + chunk - 1) // chunk, device=first.device, dtype=torch.long
    )
    dist.all_reduce(rounds, op=dist.ReduceOp.MAX, group=group)
    required = int(rounds.item()) * chunk
    if (width + chunk - 1) // chunk == int(rounds.item()):
        return tensors
    result = []
    for value in tensors:
        if value is None:
            result.append(None)
            continue
        shape = list(value.shape)
        shape[dim] = required - width
        result.append(torch.cat((value, value.new_zeros(shape)), dim=dim))
    return tuple(result)


__all__.append("align_objective_blocks")
