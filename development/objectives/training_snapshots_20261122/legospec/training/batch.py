from __future__ import annotations

"""Generic anchor/layout preparation, outside trainable math."""

import os
import tempfile
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Optional

import torch

from legospec.training.block import (
    apply_exponential_position_decay,
    gather_block_values,
    sample_sorted_anchors,
)


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


def position_weights(mask, gamma):
    return apply_exponential_position_decay(
        mask.float(),
        gamma,
        torch.arange(mask.size(-1), device=mask.device),
        first_prediction_index=0,
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
class OPDReplayMetadata:
    accepted_lengths: dict[str, torch.Tensor]
    rng_state: dict


@dataclass
class PreparedLogicalBatch:
    data: object
    layout: BlockLayout
    window_denominators: torch.Tensor
    spill_path: Optional[str] = None
    opd_metadata: Optional[dict] = None
    opd_component_denominators: Optional[dict] = None
    opd_reference_samples: Optional[torch.Tensor] = None
    opd_inputs_prepared: bool = False

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


"""Pack execution axes without changing occurrence or objective semantics."""
from dataclasses import replace


from legospec.training.layout import build_packed_batch_plan


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


"""Objective-independent anchors, keyed by sample identity and epoch.

All random draws use a private CPU generator for each unpadded sample. This
keeps positions invariant to CUDA RNG use, padding, rank and batch slicing.
The layout is collected before model execution and reused during replay.
"""

import hashlib
import json


ANCHOR_SAMPLING_VERSION = "sample_epoch_v1"


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
            [ANCHOR_SAMPLING_VERSION, int(seed), int(epoch), identity],
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
