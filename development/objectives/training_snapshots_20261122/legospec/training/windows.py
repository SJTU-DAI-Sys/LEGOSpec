from __future__ import annotations

"""Collect once per optimizer window, replay fixed blocks without target reruns."""

from pathlib import Path

from legospec.training.batch import PreparedLogicalBatch, cpu_copy, tensor_bytes
from legospec.training.component_reduction import global_component_counts


def prepare_anchor_opd_window(session, first, iterator, count):
    from legospec.training import setup

    if session.args.accumulation_steps == 1:
        return None
    import torch

    prepared, used, counts, samples = [], 0, None, 0
    try:
        for index in range(count):
            raw = first if index == 0 else next(iterator)
            with torch.no_grad():
                metadata = session.fsdp_model(
                    **setup._anchor_values(session, raw),
                    opd_scorer=session.anchor_scorer,
                    metadata_only=True,
                )
            local = metadata["component_denominators"]
            if counts is None:
                counts = {
                    b: {k: v.detach().clone() for k, v in terms.items()}
                    for b, terms in local.items()
                }
            else:
                for branch, terms in local.items():
                    for key, value in terms.items():
                        counts[branch][key] += value
            samples += metadata["sample_count"]
            record = cpu_copy(metadata["sampled_opd_batch"])
            data = {
                "input_ids": cpu_copy(raw["input_ids"]),
                "loss_mask": cpu_copy(raw["loss_mask"]),
                "sampled_opd_batch": record,
            }
            item = PreparedLogicalBatch(data, record.layout, None)
            prepared.append(item)
            size = tensor_bytes(data)
            if used + size > session.args.window_buffer_mb * 1024**2:
                item.spill(Path(session.args.cache_dir) / "anchor_opd_windows")
            else:
                used += size
            del metadata, record, data
        totals, reference = global_component_counts(
            counts, samples, session.process_group
        )
        if totals["final"]["survival"] <= 0:
            raise ValueError("Sampled OPD optimizer window contains no complete blocks")
        for item in prepared:
            item.opd_component_denominators = totals
            item.opd_reference_samples = reference
        return prepared
    except BaseException:
        for item in prepared:
            item.release()
        raise


"""RNG-preserving detached OPD region metadata for multi-forward windows."""

from dataclasses import fields

import torch

from legospec.training.checkpoint import capture_rng_state, restore_rng_state
from legospec.training.batch import BlockLayout, OPDReplayMetadata, prepare_layout


def prepare_opd_window(session, first, iterator, count):
    from legospec.training import setup

    # Ordinary accumulation=1 is exactly one complete FSDP forward. Its three
    # counts can be reduced after all objective chunks without a metadata pass.
    if session.args.accumulation_steps == 1 and session.topology is None:
        return None
    from legospec.training.runner import iter_microbatch_slices

    entry_rng = capture_rng_state()
    prepared, used, local_counts, sample_count = [], 0, None, 0
    try:
        with torch.no_grad():
            for index in range(count):
                raw = cpu_copy(first if index == 0 else next(iterator))
                full = setup._step_values(session, raw)
                layouts, replay = [], {}
                for start, end in iter_microbatch_slices(session):
                    values = {
                        key: (None if value is None else value[start:end])
                        for key, value in full.items()
                    }
                    sampler = setup.anchor_layout
                    layout = (
                        None
                        if sampler is None
                        else sampler(
                            session, raw, microbatch_start=start, microbatch_end=end
                        )
                    )
                    if layout is None:
                        layout = prepare_layout(
                            values["input_ids"],
                            values["loss_mask"],
                            session.draft.block_size,
                            session.args.num_anchors,
                            complete_blocks=True,
                            minimum_anchor=1
                            if getattr(session.draft, "uses_hidden_feedback", False)
                            else 0,
                        )
                    else:
                        layout = layout.to(values["input_ids"].device)
                    rng = capture_rng_state()
                    meta = session.fsdp_model(
                        **values, layout=layout, metadata_only=True
                    )
                    replay[start, end] = OPDReplayMetadata(
                        cpu_copy(meta["accepted_lengths"]), rng
                    )
                    layouts.append(cpu_copy(layout))
                    if local_counts is None:
                        local_counts = {
                            branch: {
                                key: den.detach().clone() for key, den in terms.items()
                            }
                            for branch, terms in meta["component_denominators"].items()
                        }
                    else:
                        for branch, terms in meta["component_denominators"].items():
                            for key, den in terms.items():
                                local_counts[branch][key] += den.detach()
                    sample_count += meta["sample_count"]
                joined = BlockLayout(
                    **{
                        field.name: torch.cat(
                            [getattr(layout, field.name) for layout in layouts], dim=0
                        )
                        for field in fields(BlockLayout)
                    }
                )
                data = cpu_copy(full)
                item = PreparedLogicalBatch(
                    data, joined, None, opd_metadata=replay, opd_inputs_prepared=True
                )
                size = tensor_bytes(data)
                if used + size > session.args.window_buffer_mb * 1024**2:
                    item.spill(Path(session.args.cache_dir) / "composable_windows")
                else:
                    used += size
                prepared.append(item)
                del full, values, meta, layout
        totals, reference = global_component_counts(
            local_counts, sample_count, session.process_group
        )
        for branch, terms in totals.items():
            for key, den in terms.items():
                if key in ("survival", "token") and den <= 0:
                    raise ValueError(
                        "OPD optimizer window has no complete supervised blocks"
                    )
        for item in prepared:
            item.opd_component_denominators = cpu_copy(totals)
            item.opd_reference_samples = cpu_copy(reference)
        return prepared
    except BaseException:
        for item in prepared:
            item.release()
        raise
    finally:
        # Formal forwards replay their individual post-layout RNG snapshots.
        # At the end of the window the stochastic stream advances only once.
        restore_rng_state(entry_rng)


"""Plan only CPU supervision metadata; hidden packets stay a streaming input."""

from concurrent.futures import ThreadPoolExecutor
from itertools import islice


from legospec.training.batch import prepare_sample_epoch_layout


def _prepare_window(
    window,
    start,
    *,
    branches,
    block_size,
    num_anchors,
    seed,
    steps_per_epoch,
    indexer,
    supervision_mode,
    minimum_anchor,
):
    """CPU-only worker: no DataLoader iteration, hidden features, or CUDA calls."""
    layouts, counts = [], None
    for data in window:
        ids, mask, valid = (
            data[key] for key in ("input_ids", "loss_mask", "attention_mask")
        )
        if any(value.device.type != "cpu" for value in (ids, mask, valid)):
            raise ValueError("Target window lookahead requires CPU token metadata")
        layout = prepare_sample_epoch_layout(
            ids,
            mask,
            valid,
            block_size=block_size,
            num_anchors=num_anchors,
            seed=seed,
            epoch=start // steps_per_epoch,
            supervision_mode=supervision_mode,
            minimum_anchor=minimum_anchor,
        )
        if counts is None:
            counts = torch.zeros(ids.size(0), 3, dtype=torch.float64)
        if counts.size(0) != ids.size(0):
            raise ValueError("Producer sample slots changed inside an optimizer window")
        for row in range(ids.size(0)):
            part = layout.batch_slice(row, row + 1)
            native = branches.native_denominators(part)
            auxiliary = native.get("base", 0.0)
            if indexer:
                positions = torch.arange(ids.size(1))
                history = (
                    valid[row : row + 1, None].bool()
                    & (positions[None, None] < part.anchor_positions[..., None])
                ).any(-1)
                auxiliary = (
                    part.block_keep_mask & part.prediction_mask.bool().any(-1) & history
                ).sum()
            counts[row] += torch.tensor(
                [1.0, float(native["final"]), float(auxiliary)], dtype=torch.float64
            )
        layouts.append(layout)
    return [
        {
            **data,
            "_streamed_anchor_positions": layout.anchor_positions,
            "_streamed_block_keep_mask": layout.block_keep_mask,
            "_streamed_window_counts": counts,
        }
        for data, layout in zip(window, layouts, strict=True)
    ]


def plan_target_windows(
    batches,
    *,
    branches,
    block_size,
    num_anchors,
    seed,
    accumulation_steps,
    steps_per_epoch,
    indexer=False,
    supervision_mode="sampled",
    minimum_anchor=0,
    asynchronous=False,
):
    """Keep source order and the existing accumulation-window anchor epoch.

    Each producer sums counts independently for its fixed sample slots. Routing
    these per-slot totals with the first hidden packet lets draft ranks reduce
    the global denominator before their first backward. No teacher forward is
    performed during lookahead and no hidden tensor is retained here.

    Async mode prepares the next token-only window while the caller submits
    GPU work for the current window. Source iteration stays on the caller
    thread (important for DataLoader worker creation and epoch state); one
    background worker and one future bound lookahead to two metadata windows.
    """
    if accumulation_steps < 1 or steps_per_epoch < 1:
        raise ValueError("Window accumulation and epoch length must be positive")
    batches = iter(batches)
    settings = dict(
        branches=branches,
        block_size=block_size,
        num_anchors=num_anchors,
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        indexer=indexer,
        supervision_mode=supervision_mode,
        minimum_anchor=minimum_anchor,
    )
    start = 0
    if not asynchronous:
        while window := list(islice(batches, accumulation_steps)):
            yield from _prepare_window(window, start, **settings)
            start += len(window)
        return
    window = list(islice(batches, accumulation_steps))
    if not window:
        return
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="target-metadata"
    ) as pool:
        pending = pool.submit(_prepare_window, window, start, **settings)
        start += len(window)
        while pending is not None:
            prepared = pending.result()  # Worker errors must reach the producer.
            following = list(islice(batches, accumulation_steps))
            pending = (
                pool.submit(_prepare_window, following, start, **settings)
                if following
                else None
            )
            start += len(following)
            yield from prepared
