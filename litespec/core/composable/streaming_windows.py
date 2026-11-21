"""Plan only CPU supervision metadata; hidden packets stay a streaming input."""

from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import torch

from .anchor_sampling import prepare_sample_epoch_layout


def _prepare_window(window, start, *, branches, block_size, num_anchors, seed,
                    steps_per_epoch, indexer, supervision_mode, minimum_anchor):
    """CPU-only worker: no DataLoader iteration, hidden features, or CUDA calls."""
    layouts, counts = [], None
    for data in window:
        ids, mask, valid = (data[key] for key in
                            ("input_ids", "loss_mask", "attention_mask"))
        if any(value.device.type != "cpu" for value in (ids, mask, valid)):
            raise ValueError("Target window lookahead requires CPU token metadata")
        layout = prepare_sample_epoch_layout(
            ids, mask, valid, block_size=block_size, num_anchors=num_anchors,
            seed=seed, epoch=start // steps_per_epoch,
            supervision_mode=supervision_mode, minimum_anchor=minimum_anchor)
        if counts is None:
            counts = torch.zeros(ids.size(0), 3, dtype=torch.float64)
        if counts.size(0) != ids.size(0):
            raise ValueError("Producer sample slots changed inside an optimizer window")
        for row in range(ids.size(0)):
            part = layout.batch_slice(row, row + 1)
            native = branches.native_denominators(part)
            auxiliary = native.get("base", 0.)
            if indexer:
                positions = torch.arange(ids.size(1))
                history = (valid[row:row + 1, None].bool()
                           & (positions[None, None] < part.anchor_positions[..., None])).any(-1)
                auxiliary = (part.block_keep_mask & part.prediction_mask.bool().any(-1)
                             & history).sum()
            counts[row] += torch.tensor([1., float(native["final"]), float(auxiliary)],
                                        dtype=torch.float64)
        layouts.append(layout)
    return [{**data, "_streamed_anchor_positions": layout.anchor_positions,
             "_streamed_block_keep_mask": layout.block_keep_mask,
             "_streamed_window_counts": counts}
            for data, layout in zip(window, layouts, strict=True)]


def plan_target_windows(batches, *, branches, block_size, num_anchors, seed,
                        accumulation_steps, steps_per_epoch, indexer=False,
                        supervision_mode="sampled", minimum_anchor=0, asynchronous=False):
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
    settings = dict(branches=branches, block_size=block_size, num_anchors=num_anchors,
                    seed=seed, steps_per_epoch=steps_per_epoch, indexer=indexer,
                    supervision_mode=supervision_mode, minimum_anchor=minimum_anchor)
    start = 0
    if not asynchronous:
        while window := list(islice(batches, accumulation_steps)):
            yield from _prepare_window(window, start, **settings)
            start += len(window)
        return
    window = list(islice(batches, accumulation_steps))
    if not window:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='target-metadata') as pool:
        pending = pool.submit(_prepare_window, window, start, **settings)
        start += len(window)
        while pending is not None:
            prepared = pending.result()  # Worker errors must reach the producer.
            following = list(islice(batches, accumulation_steps))
            pending = (pool.submit(_prepare_window, following, start, **settings)
                       if following else None)
            start += len(following)
            yield from prepared
