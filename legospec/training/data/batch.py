"""Sample anchors and prepare labels, masks and packed execution batches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Optional

import torch

from legospec.training.data.layout import build_packed_batch_plan


@dataclass
class BlockLayout:
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    labels: torch.Tensor
    previous_token_ids: torch.Tensor
    prediction_mask: torch.Tensor
    label_indices: torch.Tensor
    sample_prediction_counts: torch.Tensor

    def to(self, device):
        return BlockLayout(
            **{f.name: getattr(self, f.name).to(device) for f in fields(self)}
        )

    def batch_slice(self, start, end):
        return BlockLayout(
            **{f.name: getattr(self, f.name)[start:end] for f in fields(self)}
        )


def prepare_layout(
    input_ids,
    loss_mask,
    block_size,
    num_anchors,
    *,
    complete_blocks=False,
    generator=None,
    supervision_mode="sampled",
    minimum_anchor=0,
):
    length = input_ids.size(1)
    eligible = (loss_mask[:, :-1] > 0.5) & (loss_mask[:, 1:] > 0.5)
    if complete_blocks:
        offsets = torch.arange(block_size, device=input_ids.device)
        indices = (
            torch.arange(max(0, length - block_size + 1), device=input_ids.device)[
                :, None
            ]
            + offsets
        )
        whole = (loss_mask[:, indices] > 0.5).all(-1)
        eligible = torch.nn.functional.pad(
            whole, (0, max(0, length - 1 - whole.size(1)))
        )
    eligible[:, :minimum_anchor] = False
    if supervision_mode not in ("sampled", "all"):
        raise ValueError("supervision_mode must be sampled or all")
    if supervision_mode == "all":
        width = max(int(num_anchors), eligible.size(1), 1)
        sentinel = eligible.size(1)
        positions = torch.arange(sentinel, device=input_ids.device)[None].expand_as(
            eligible
        )
        positions = positions.masked_fill(~eligible, sentinel).sort(-1).values
        positions = torch.nn.functional.pad(
            positions, (0, width - sentinel), value=sentinel
        )
        keep = positions != sentinel
        num_anchors = positions.masked_fill(~keep, 0)
    elif eligible.numel() and eligible.any():
        num_anchors, keep = sample_sorted_anchors(
            eligible, num_anchors, output_width=num_anchors, generator=generator
        )
    else:
        num_anchors = input_ids.new_zeros(input_ids.size(0), num_anchors)
        keep = torch.zeros_like(num_anchors, dtype=torch.bool)
    return layout_from_anchors(input_ids, loss_mask, num_anchors, keep, block_size)


def layout_from_anchors(input_ids, loss_mask, num_anchors, keep, block_size):
    """Materialize labels from a frozen metadata plan without resampling."""
    labels, in_bounds, indices = gather_block_values(input_ids, num_anchors, block_size)
    mask, _, _ = gather_block_values(loss_mask, num_anchors, block_size)
    mask = mask.float() * keep[..., None] * in_bounds
    mask = mask[..., 1:]
    labels = labels[..., 1:]
    previous = torch.cat(
        (input_ids.gather(1, num_anchors)[..., None], labels[..., :-1]), -1
    )
    return BlockLayout(
        num_anchors,
        keep,
        labels,
        previous,
        mask,
        indices[..., 1:],
        mask.flatten(1).sum(-1),
    )



def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    if is_dataclass(value):
        return type(value)(
            **{f.name: cpu_copy(getattr(value, f.name)) for f in fields(value)}
        )
    return value


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    if is_dataclass(value):
        return sum(tensor_bytes(getattr(value, f.name)) for f in fields(value))
    return 0


@dataclass
class PreparedLogicalBatch:
    data: object
    layout: BlockLayout
    window_denominators: torch.Tensor
    spill_path: Optional[str] = None

    def materialize(self):
        if self.spill_path and self.data is None:
            # Locally produced trusted tensor packet, never a dataset-provided pickle.
            self.data = torch.load(
                self.spill_path, map_location="cpu", weights_only=False
            )
        return self.data

    def release(self):
        if self.spill_path:
            Path(self.spill_path).unlink(missing_ok=True)
            self.spill_path = None
        self.data = None

    def spill(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="window-", suffix=".pt", dir=directory)
        os.close(fd)
        torch.save(self.data, name)
        self.spill_path, self.data = name, None


def prepare_packed_plan(valid, layout, block_size, *, query_len=None):
    valid = valid.bool()
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("Packed training requires contiguous right padding")
    lengths = valid.sum(-1)
    # Protocol-only block keeps empty ranks on the same differentiable path.
    keep = layout.block_keep_mask.clone()
    if not keep.any():
        keep[0, 0] = True
    plan = build_packed_batch_plan(
        lengths,
        layout.anchor_positions,
        keep,
        block_size,
        query_len=query_len,
        device=valid.device,
    )
    selected = plan.kept_block_indices
    offset = plan.context_offsets.index_select(0, plan.block_sample_ids)[None, :, None]
    return replace(
        plan,
        supervision_mask=layout.prediction_mask.flatten(0, 1).index_select(0, selected)[
            None
        ],
        teacher_indices=(
            layout.label_indices.flatten(0, 1).index_select(0, selected)[None]
            + offset
            - 1
        ).clamp_min(0),
    )


def pack_training_batch(
    input_ids,
    hidden,
    loss_mask,
    last,
    valid,
    layout,
    block_size,
    *,
    query_len=None,
    plan=None,
):
    valid = (
        torch.ones_like(input_ids, dtype=torch.bool) if valid is None else valid.bool()
    )
    plan = (
        prepare_packed_plan(valid, layout, block_size, query_len=query_len)
        if plan is None
        else plan.to(input_ids.device)
    )
    index = plan.kept_block_indices

    def blocks(value):
        return value.flatten(0, 1).index_select(0, index).unsqueeze(0)

    offset = plan.context_offsets.index_select(0, plan.block_sample_ids)[None, :, None]
    packed_layout = replace(
        layout,
        anchor_positions=blocks(layout.anchor_positions),
        block_keep_mask=blocks(layout.block_keep_mask),
        labels=blocks(layout.labels),
        previous_token_ids=blocks(layout.previous_token_ids),
        prediction_mask=blocks(layout.prediction_mask),
        label_indices=blocks(layout.label_indices) + offset,
    )
    values = tuple(
        None if x is None else plan.pack_context(x)
        for x in (input_ids, hidden, loss_mask, last, valid)
    )
    return (*values, packed_layout, plan)


def unpack_block_values(value, plan, original_shape):
    flat = value.flatten(0, 1)[: plan.kept_block_indices.numel()]
    shape = (original_shape[0] * original_shape[1], *flat.shape[1:])
    return (
        flat.new_zeros(shape)
        .index_copy(0, plan.kept_block_indices, flat)
        .reshape(*original_shape, *flat.shape[1:])
    )


def prepare_sample_epoch_layout(
    input_ids,
    loss_mask,
    attention_mask,
    *,
    block_size,
    num_anchors,
    seed,
    epoch,
    sample_indices=None,
    supervision_mode="sampled",
    minimum_anchor=0,
):
    ids = input_ids.detach().cpu()
    mask = loss_mask.detach().cpu().float()
    valid = attention_mask.detach().cpu().bool()
    if ids.ndim != 2 or ids.shape != mask.shape or ids.shape != valid.shape:
        raise ValueError("Anchor inputs must have matching [batch, sequence] shapes")
    if int(epoch) < 0:
        raise ValueError("Anchor epoch must be nonnegative")
    indices = (
        None
        if sample_indices is None
        else torch.as_tensor(sample_indices).reshape(-1).tolist()
    )
    if indices is not None and len(indices) != len(ids):
        raise ValueError("One stable sample_index is required per anchor sample")
    layouts = []
    if supervision_mode == "all":
        num_anchors = max(1, ids.size(1) - 1)
    for row in range(len(ids)):
        positions = valid[row].nonzero().reshape(-1)
        length = int(positions[-1]) + 1 if positions.numel() else 1
        tokens = ids[row : row + 1, :length]
        labels = mask[row : row + 1, :length] * valid[row : row + 1, :length]
        if indices is None:
            # Token-only online datasets do not necessarily carry sample_index.
            # Include masks and exclude batch padding from this stable identity.
            identity = [
                "content",
                hashlib.sha256(
                    json.dumps(
                        [
                            tokens.tolist(),
                            labels.tolist(),
                            valid[row, :length].tolist(),
                        ],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            ]
        else:
            identity = ["sample_index", int(indices[row])]
        key = json.dumps(
            [int(seed), int(epoch), identity],
            separators=(",", ":"),
        ).encode()
        row_seed = int.from_bytes(
            hashlib.blake2b(key, digest_size=8).digest(), "little"
        )
        generator = torch.Generator(device="cpu").manual_seed(row_seed)
        layouts.append(
            prepare_layout(
                tokens,
                labels,
                block_size,
                num_anchors,
                complete_blocks=True,
                generator=generator,
                supervision_mode=supervision_mode,
                minimum_anchor=minimum_anchor,
            )
        )
    if not layouts:
        raise ValueError("Cannot prepare anchors for an empty batch")
    return BlockLayout(
        **{
            field.name: torch.cat(
                [getattr(layout, field.name) for layout in layouts], dim=0
            )
            for field in fields(BlockLayout)
        }
    ).to(input_ids.device)


def anchor_layout_digest(layout):
    """Small audit fingerprint; includes inactive-slot masks and prediction masks."""
    digest = hashlib.sha256()
    for name in ("anchor_positions", "block_keep_mask", "prediction_mask"):
        value = getattr(layout, name).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def sample_sorted_anchors(
    valid_anchor_mask: torch.Tensor,
    num_anchors: int,
    *,
    output_width: Optional[int] = None,
    max_valid_anchors: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample valid anchors, sort them, and pad with inactive slots.

    Returns ``anchor_positions`` and ``block_keep_mask``. ``output_width`` only
    sets the physical padded width; it does not change which sequence positions
    are eligible or how many real anchors are sampled.
    """

    if valid_anchor_mask.ndim != 2:
        raise ValueError("valid_anchor_mask must have shape (batch, candidates)")
    valid_anchor_mask = valid_anchor_mask.bool()
    counts = valid_anchor_mask.sum(dim=1)
    if max_valid_anchors is None:
        max_valid_anchors = int(counts.max().item())
    sampled_width = min(int(num_anchors), max(0, int(max_valid_anchors)))
    if sampled_width <= 0:
        raise ValueError("No valid anchor positions in this batch.")
    width = sampled_width if output_width is None else int(output_width)
    if width < sampled_width:
        raise ValueError("output_width cannot be smaller than sampled width")

    scores = torch.rand(
        valid_anchor_mask.shape,
        device=valid_anchor_mask.device,
        dtype=torch.float32,
        generator=generator,
    )
    scores.masked_fill_(~valid_anchor_mask, 2.0)
    selected_width = min(sampled_width, valid_anchor_mask.size(1))
    anchor_positions = scores.argsort(dim=1)[:, :selected_width]
    selected_valid = torch.gather(valid_anchor_mask, 1, anchor_positions)
    sentinel = valid_anchor_mask.size(1)
    anchor_positions = anchor_positions.masked_fill(~selected_valid, sentinel)
    if selected_width < width:
        anchor_positions = torch.cat(
            [
                anchor_positions,
                anchor_positions.new_full(
                    (anchor_positions.size(0), width - selected_width), sentinel
                ),
            ],
            dim=1,
        )
    anchor_positions = anchor_positions.sort(dim=1).values
    block_keep_mask = anchor_positions != sentinel
    return (
        torch.where(
            block_keep_mask, anchor_positions, torch.zeros_like(anchor_positions)
        ),
        block_keep_mask,
    )


def gather_block_values(
    values: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather ``anchor + offset`` values and return values, bounds, safe indices."""

    if values.ndim < 2 or anchor_positions.ndim != 2:
        raise ValueError(
            "values and anchor_positions must have batch and sequence axes"
        )
    offsets = torch.arange(int(block_size), device=anchor_positions.device).view(
        1, 1, -1
    )
    indices = anchor_positions.unsqueeze(-1) + offsets
    in_bounds = indices < values.size(1)
    safe_indices = indices.clamp(min=0, max=max(values.size(1) - 1, 0))
    expanded = values.unsqueeze(1).expand(
        -1, anchor_positions.size(1), *values.shape[1:]
    )
    gather_index = safe_indices
    for _ in values.shape[2:]:
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(-1, -1, -1, *values.shape[2:])
    return torch.gather(expanded, 2, gather_index), in_bounds, safe_indices



def aligned_width(width: int, alignment: int) -> int:
    """Round a positive logical width to a backend-required physical width."""

    width, alignment = int(width), int(alignment)
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return math.ceil(width / alignment) * alignment


__all__ = [
    "aligned_width",
    "gather_block_values",
    "sample_sorted_anchors",
]


def select_tp_rank_batch(value, tp_rank: int):
    """Copy rank ``tp_rank``'s sample so full target tensors can be released."""
    if isinstance(value, dict):
        return {
            key: select_tp_rank_batch(item, tp_rank) for (key, item) in value.items()
        }
    if isinstance(value, tuple):
        return tuple((select_tp_rank_batch(item, tp_rank) for item in value))
    if isinstance(value, list):
        return [select_tp_rank_batch(item, tp_rank) for item in value]
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        raise TypeError(
            "TP draft batch values must be batch-first tensors or containers."
        )
    if not 0 <= int(tp_rank) < value.size(0):
        raise ValueError(
            f"TP rank {tp_rank} cannot select from batch size {value.size(0)}."
        )
    return value.narrow(0, int(tp_rank), 1).clone(memory_format=torch.contiguous_format)


def anchor_layout(session, data, *, microbatch_start=None, microbatch_end=None):
    if (
        session.topology is not None
        and getattr(data, "anchor_positions", None) is not None
    ):
        subset = slice(microbatch_start, microbatch_end)
        return layout_from_anchors(
            data.input_ids[subset],
            data.loss_mask[subset] * data.context_valid_mask[subset],
            data.anchor_positions[subset],
            data.block_keep_mask[subset],
            session.draft.block_size,
        )
    if isinstance(data, dict) and "_planned_layout" in data:
        layout = data["_planned_layout"]
        if session.tp_draft_rank is not None:
            layout = layout.batch_slice(
                session.tp_draft_rank, session.tp_draft_rank + 1
            )
        return layout.batch_slice(microbatch_start, microbatch_end)
    hidden_feedback = session.draft.uses_hidden_feedback
    supervision_mode = getattr(session.args, "supervision_mode", "sampled")

    if session.topology is None:
        (ids, mask, valid) = (
            data[key] for key in ("input_ids", "loss_mask", "attention_mask")
        )
        indices = data.get("sample_index")
        epoch = int(data.get("_anchor_epoch", session.state.epoch))
        if session.tp_draft_rank is not None:
            start = session.tp_draft_rank
            (ids, mask, valid) = (
                value[start : start + 1] for value in (ids, mask, valid)
            )
            if indices is not None:
                indices = indices[start : start + 1]
    else:
        (ids, mask, valid) = (data.input_ids, data.loss_mask, data.context_valid_mask)
        indices = None
        epoch = session.state.epoch
    subset = slice(microbatch_start, microbatch_end)
    return prepare_sample_epoch_layout(
        ids[subset],
        mask[subset],
        valid[subset],
        sample_indices=None if indices is None else indices[subset],
        block_size=session.draft.block_size,
        num_anchors=session.args.num_anchors,
        seed=session.args.seed,
        epoch=epoch,
        supervision_mode=supervision_mode if hidden_feedback else "sampled",
        minimum_anchor=1 if hidden_feedback else 0,
    )
