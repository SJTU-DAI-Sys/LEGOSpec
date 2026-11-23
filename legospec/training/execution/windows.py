"""Prepare global optimizer-window denominators and manage batch lifetime."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist

from legospec.modeling.draft.objective.composition import LossBranches
from legospec.training.data.batch import (
    PreparedLogicalBatch,
    anchor_layout,
    cpu_copy,
    tensor_bytes,
)


def prepare_optimizer_window(session, first, iterator, count):
    recipe = session.draft.config.composable_config
    indexer = session.draft.config.indexer_config
    branches = LossBranches(recipe)
    if (
        session.topology is not None
        and getattr(first, "window_counts", None) is not None
    ):
        totals = first.window_counts.sum(0)
        if dist.is_initialized():
            dist.all_reduce(totals, group=session.process_group)
        for name in branches.objectives:
            if totals[1 if name == "final" else 2] <= 0:
                raise ValueError(
                    "Optimizer window has no supervised proposal positions"
                )
        session.planned_window_denominators = totals
        if not getattr(session, "_streamed_window_announced", False):
            print(
                "Streaming optimizer windows active: consume hidden packets directly from the receive ring; token-only lookahead, no hidden replay.",
                flush=True,
            )
            session._streamed_window_announced = True
        return None
    if (
        recipe.get("proposal", {}).get("type")
        not in ("selector", "residual_selector", "topk_selector")
        and indexer is None
    ):
        return None
    if isinstance(first, dict) and "_planned_window_counts" in first:
        totals = first["_planned_window_counts"]
        if session.tp_draft_rank is not None:
            totals = totals[session.tp_draft_rank : session.tp_draft_rank + 1]
        totals = totals.sum(0).cuda()
        if dist.is_initialized():
            dist.all_reduce(totals, group=session.process_group)
        for name in branches.objectives:
            if totals[1 if name == "final" else 2] <= 0:
                raise ValueError(
                    "Optimizer window has no supervised proposal positions"
                )
        session.planned_window_denominators = totals
        return None
    (prepared, used) = ([], 0)
    totals = torch.zeros(3, device="cuda", dtype=torch.float64)
    for index in range(count):
        raw = first if index == 0 else next(iterator)
        data = cpu_copy(raw)
        if session.topology is not None:
            (ids, mask) = (data.input_ids, data.loss_mask)
        else:
            (ids, mask) = (data["input_ids"], data["loss_mask"])
            if session.tp_draft_rank is not None:
                (ids, mask) = (
                    ids[session.tp_draft_rank : session.tp_draft_rank + 1],
                    mask[session.tp_draft_rank : session.tp_draft_rank + 1],
                )
        layout = anchor_layout(session, data)
        denominators = branches.native_denominators(layout)
        primary = denominators["final"].item()
        if indexer is None:
            auxiliary = denominators["base"].item()
        else:
            context_valid = (
                data.context_valid_mask
                if session.topology is not None
                else data["attention_mask"]
            )
            if session.tp_draft_rank is not None:
                context_valid = context_valid[
                    session.tp_draft_rank : session.tp_draft_rank + 1
                ]
            positions = torch.arange(ids.size(1))
            has_history = (
                context_valid[:, None].bool()
                & (positions[None, None] < layout.anchor_positions[..., None])
            ).any(-1)
            auxiliary = (
                (
                    layout.block_keep_mask
                    & (layout.prediction_mask > 0).any(-1)
                    & has_history
                )
                .sum()
                .item()
            )
        totals += totals.new_tensor([ids.size(0), primary, auxiliary])
        item = PreparedLogicalBatch(data, layout, totals)
        size = tensor_bytes(data)
        if used + size > session.args.window_buffer_mb * 1024**2:
            item.spill(Path(session.args.cache_dir) / "composable_windows")
        else:
            used += size
        prepared.append(item)
    if dist.is_initialized():
        dist.all_reduce(totals, group=session.process_group)
    invalid = (
        totals[1] <= 0
        if not branches.is_selector
        else any(
            (totals[1 if name == "final" else 2] <= 0 for name in branches.objectives)
        )
    )
    if invalid:
        for item in prepared:
            item.release()
        raise ValueError("Optimizer window has no supervised proposal positions")
    for item in prepared:
        item.window_denominators = totals
    return prepared


def release_batch(data):
    if isinstance(data, PreparedLogicalBatch):
        data.release()
