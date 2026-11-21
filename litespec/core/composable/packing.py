"""Pack execution axes without changing occurrence or objective semantics."""
from dataclasses import replace

import torch

from litespec.core.packed_layout import build_packed_batch_plan


def prepare_packed_plan(valid, layout, block_size, *, query_len=None):
    valid = valid.bool()
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("Packed training requires contiguous right padding")
    lengths = valid.sum(-1)
    # Protocol-only block keeps empty ranks on the same differentiable path.
    keep = layout.block_keep_mask.clone()
    if not keep.any():
        keep[0, 0] = True
    plan = build_packed_batch_plan(lengths, layout.anchor_positions, keep, block_size,
                                   query_len=query_len, device=valid.device)
    selected = plan.kept_block_indices
    offset = plan.context_offsets.index_select(0, plan.block_sample_ids)[None, :, None]
    return replace(plan,
        supervision_mask=layout.prediction_mask.flatten(0, 1).index_select(0, selected)[None],
        teacher_indices=(layout.label_indices.flatten(0, 1).index_select(0, selected)[None] + offset - 1).clamp_min(0))


def pack_training_batch(input_ids, hidden, loss_mask, last, valid, layout, block_size,
                        *, query_len=None, plan=None):
    valid = torch.ones_like(input_ids, dtype=torch.bool) if valid is None else valid.bool()
    plan = (prepare_packed_plan(valid, layout, block_size, query_len=query_len)
            if plan is None else plan.to(input_ids.device))
    index = plan.kept_block_indices

    def blocks(value):
        return value.flatten(0, 1).index_select(0, index).unsqueeze(0)

    offset = plan.context_offsets.index_select(0, plan.block_sample_ids)[None, :, None]
    packed_layout = replace(layout,
        anchor_positions=blocks(layout.anchor_positions),
        block_keep_mask=blocks(layout.block_keep_mask),
        labels=blocks(layout.labels), previous_token_ids=blocks(layout.previous_token_ids),
        prediction_mask=blocks(layout.prediction_mask),
        label_indices=blocks(layout.label_indices) + offset)
    values = tuple(None if x is None else plan.pack_context(x)
                   for x in (input_ids, hidden, loss_mask, last, valid))
    return (*values, packed_layout, plan)


def unpack_block_values(value, plan, original_shape):
    flat = value.flatten(0, 1)[:plan.kept_block_indices.numel()]
    shape = (original_shape[0] * original_shape[1], *flat.shape[1:])
    return flat.new_zeros(shape).index_copy(0, plan.kept_block_indices, flat).reshape(
        *original_shape, *flat.shape[1:])
