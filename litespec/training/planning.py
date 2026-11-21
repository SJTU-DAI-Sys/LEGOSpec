"""Metadata-only scheduling shared by every composable recipe.

The source sampler remains the authority for order, repetition and drop-last.
W=1 only reassigns occurrences inside one optimizer update. W>1 explicitly
permutes updates inside a lookahead window. Worker prefetch never commits a
training cursor. Pending plans, including anchors, are checkpointed verbatim.
"""
from __future__ import annotations

from copy import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import threading
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from litespec.core.composable.anchor_sampling import prepare_sample_epoch_layout
from litespec.core.composable.batch import prepare_layout, layout_from_anchors

PLAN_VERSION = 'composable_occurrences_v1'
ROUND_ORDER_VERSION = 'balanced_shuffle_v1'


@dataclass(frozen=True)
class PlannedSample:
    index: int
    epoch: int
    source_position: int
    anchors: tuple[int, ...]
    length: int
    cost: float
    denominators: tuple[float, ...] = ()
    update_counts: tuple[tuple[float, ...], ...] = ()
    shard: str | None = None


def assign_microstep_batches(pool, replicas, batch_size):
    """Group similar-cost work into synchronized rounds, then balance each round.

    Gradient synchronization happens on every backward. The scheduling target is
    therefore the sum of per-round maximum rank costs, not maximum rank cost
    summed across the whole accumulation window. This is a deterministic cost
    heuristic, not a runtime/readiness-driven queue.
    """
    width = replicas * batch_size
    if replicas < 1 or batch_size < 1 or len(pool) % width:
        raise ValueError('Each microstep requires one full batch per rank')
    ordered = sorted(pool, key=lambda s: (-s.cost, s.source_position))
    batches = [[] for _ in range(replicas)]
    total_loads = [0.] * replicas
    for start in range(0, len(ordered), width):
        bins = [[] for _ in range(replicas)]
        loads = [0.] * replicas
        for sample in ordered[start:start + width]:
            available = [r for r in range(replicas) if len(bins[r]) < batch_size]
            rank = min(available, key=lambda r: (loads[r], len(bins[r]), total_loads[r],
                -sum(s.shard == sample.shard for s in bins[r]) if sample.shard is not None else 0, r))
            bins[rank].append(sample)
            loads[rank] += sample.cost
        for rank, samples in enumerate(bins):
            # Slot order may change within a batch, never across microsteps.
            batches[rank].append(sorted(samples, key=lambda s: (-s.length, s.source_position)))
            total_loads[rank] += loads[rank]
    return batches


def shuffle_balanced_microsteps(batches, *, seed, phase, window_start):
    """Permute whole synchronized rounds, preserving every rank's balanced batch.

    Use a private, rank-independent RNG so prefetch and model RNG consumption
    cannot change ordering. Saved pending windows remain authoritative on resume.
    """
    key = f'{ROUND_ORDER_VERSION}:{seed}:{phase}:{window_start}'
    round_seed = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), 'little')
    order = torch.randperm(len(batches[0]),
        generator=torch.Generator().manual_seed(round_seed)).tolist()
    return [[rank[step] for step in order] for rank in batches]


class TrainingCostModel:
    """Recipe-derived work estimates; optional measured stage coefficients.

    Estimates are scheduling weights, not advertised runtime predictions.
    Attention shape, sparse access, recurrent state, convolution, projection
    width and teacher execution contribute independently of recipe names.
    """
    def __init__(self, config, *, online=False, profile=None):
        self.config, self.online = config, online
        self.recipe = config.composable_config
        self.coefficients = {}
        if profile:
            value = json.loads(Path(profile).read_text()) if isinstance(profile, (str, Path)) else profile
            self.coefficients = value.get('coefficients', value)
            if (set(self.coefficients) - {'target', 'context', 'backbone', 'proposal', 'teacher'}
                    or any(not math.isfinite(float(x)) or float(x) < 0 for x in self.coefficients.values())):
                raise ValueError('Cost profile requires finite nonnegative stage coefficients')

    def features(self, length, layout):
        c, r = self.config, self.recipe
        h, v = c.hidden_size, c.vocab_size
        anchors = layout.anchor_positions[layout.block_keep_mask]
        a, k = len(anchors), c.block_size
        q = a * k
        rows = int(layout.prediction_mask.count_nonzero())
        teacher = int(torch.unique((layout.label_indices - 1)[layout.prediction_mask.bool()]).numel())
        features = dict(target=float((12 * length * h * h + 2 * length * length * h)
                                    * getattr(c, 'num_target_layers', 1)) if self.online else 0.,
                        context=float(length * h * h), backbone=0.,
                        proposal=float(rows * h * v), teacher=float(teacher * h * v))
        for layer in r['backbone']['layers']:
            attention, access = layer['attention'], layer.get('access', 'dense')
            pairs = float(k * anchors.sum() + a * k * k)
            if access in ('sliding_window', 'swa'):
                settings = r.get('access_configs', {}).get(access, {})
                window = settings.get('window_size', settings.get('window', 512))
                pairs = float(k * anchors.clamp(max=window).sum() + a * k * k)
            if access == 'dsa':
                settings = r.get('access_configs', {}).get('dsa', {})
                pairs = float(q * (settings.get('index_topk', 1024) + k))
            if attention in ('gdn', 'gated_deltanet'):
                # Ordered history updates and independent per-anchor query state.
                work = (length + q) * h * h
            elif attention == 'mla':
                settings = r.get('attention_configs', {}).get('mla', {})
                width = settings.get('kv_lora_rank', h)
                work = (length + q) * h * width + pairs * h
            else:
                work = (length + q) * h * h + pairs * h
            features['backbone'] += float(work + q * h * c.intermediate_size * 3)
            if layer.get('local_operator'):
                features['backbone'] += float(q * h * 16)
        proposal = r.get('proposal', {})
        rank = int(proposal.get('rank', 0))
        features['proposal'] += float(rows * rank * (h + rank + v))
        if proposal.get('type') == 'sequential_rnn':
            features['proposal'] -= float(rows * h * v)
        if 'selector' in proposal.get('type', ''):
            topk = int(proposal.get('top_k', 16))
            features['proposal'] += float(rows * topk * topk * max(rank, 1))
        if r['context']['type'] == 'eagle3':
            features['backbone'] = float((k - 1) * length * (h*h + length*h))
        return features

    def __call__(self, length, layout):
        return max(1., sum(value * float(self.coefficients.get(key, 1.))
                          for key, value in self.features(length, layout).items()))


class WindowPlanner:
    def __init__(self, loader, args, config, *, phase, total_batches, state=None):
        self.dataset, self.sampler = loader.dataset, loader.sampler
        self.batch_size = int(loader.batch_size)
        self.steps_per_epoch = len(loader)
        self.rank = int(getattr(self.sampler, 'rank', 0))
        self.replicas = int(getattr(self.sampler, 'num_replicas', 1))
        self.args, self.config, self.phase = args, config, phase
        self.total_batches = int(total_batches)
        self.accumulation = int(args.accumulation_steps)
        self.window = int(args.schedule_window_updates)
        if self.window < 1 or self.accumulation < 1:
            raise ValueError('Window and accumulation must be positive')
        self.cost = TrainingCostModel(config, online=not bool(args.train_hidden_states_path),
                                      profile=args.training_cost_profile)
        from litespec.core.composable.objectives.selector import LossBranches
        self.branches = LossBranches(config.composable_config)
        self.metadata_counts = not any(getattr(o, 'needs_component_normalization', False)
                                       for o in self.branches.objectives.values())
        self.orders, self.windows = {}, {}
        self._lock = threading.RLock()
        self._executor = None
        self._future = None
        self._future_start = None
        self._closed = False
        self.async_planning = bool(getattr(args, 'async_schedule_planning', False))
        if self.async_planning and not hasattr(self.dataset, 'training_metadata'):
            raise ValueError('Async planning requires metadata-only dataset access')
        self.committed = 0
        self.signature = dict(version=PLAN_VERSION, phase=phase, replicas=self.replicas,
            batch_size=self.batch_size, accumulation=self.accumulation,
            steps_per_epoch=self.steps_per_epoch, total_batches=self.total_batches,
            anchor_sampling=getattr(args, 'anchor_sampling', 'legacy'), seed=int(args.seed),
            cost_profile=self.cost.coefficients)
        if state is not None:
            if state['signature'] != self.signature:
                raise ValueError('Pending scheduler plan is incompatible with this run')
            self.committed = int(state['committed'])
            if not 0 <= self.committed <= self.total_batches:
                raise ValueError('Invalid scheduler cursor')
            if state.get('pending') is not None:
                pending = state['pending']
                if (not pending['start'] <= self.committed < pending['stop'] <= self.total_batches
                        or len(pending['batches']) != self.replicas
                        or any(len(batches) != pending['stop'] - pending['start']
                               for batches in pending['batches'])):
                    raise ValueError('Invalid pending scheduler window')
                self.windows[int(pending['start'])] = pending

    def _orders(self, epoch):
        if epoch not in self.orders:
            sampler = copy(self.sampler)
            sampler.set_epoch(epoch)
            if isinstance(sampler, DistributedSampler):
                generator = torch.Generator().manual_seed(sampler.seed + epoch)
                indices = (torch.randperm(len(self.dataset), generator=generator).tolist()
                           if sampler.shuffle else list(range(len(self.dataset))))
                if not sampler.drop_last:
                    extra = sampler.total_size - len(indices)
                    indices += (indices * math.ceil(extra / max(1, len(indices))))[:extra]
                else:
                    indices = indices[:sampler.total_size]
                orders = [indices[r:sampler.total_size:self.replicas] for r in range(self.replicas)]
            elif hasattr(sampler, '_global_indices'):
                indices = sampler._global_indices()
                orders = [indices[r*sampler.num_samples:(r+1)*sampler.num_samples] for r in range(self.replicas)]
            else:
                raise TypeError('Optimized scheduling requires a deterministic distributed sampler')
            self.orders = {epoch: orders}
        return self.orders[epoch]

    def _sample(self, index, epoch, source_position):
        metadata = getattr(self.dataset, 'training_metadata', None)
        data = metadata(index) if metadata else self.dataset[index]
        ids = torch.as_tensor(data['input_ids']).reshape(1, -1)
        mask = torch.as_tensor(data['loss_mask']).reshape(1, -1)
        valid = torch.as_tensor(data['attention_mask']).reshape(1, -1).bool()
        length = int(valid.sum())
        eagle = self.config.composable_config['context']['type'] == 'eagle3'
        mode = getattr(self.args, 'supervision_mode', 'sampled') if eagle else 'sampled'
        count = self.args.num_anchors
        if getattr(self.args, 'anchor_sampling', 'legacy') == 'sample_epoch' or mode == 'all':
            layout = prepare_sample_epoch_layout(ids, mask, valid,
                sample_indices=[data['sample_index']] if 'sample_index' in data else None,
                block_size=self.config.block_size, num_anchors=count, seed=self.args.seed,
                epoch=epoch, supervision_mode=mode, minimum_anchor=1 if eagle else 0)
        else:
            # Scheduling cannot use mutable per-rank CUDA RNG for anchor planning.
            key = f'{PLAN_VERSION}:{self.args.seed}:{self.phase}:{epoch}:{source_position}:{index}'
            seed = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), 'little')
            layout = prepare_layout(ids, mask * valid, self.config.block_size, count,
                complete_blocks=bool(getattr(self.args, 'complete_blocks', False)),
                minimum_anchor=1 if eagle else 0, generator=torch.Generator().manual_seed(seed))
        anchors = tuple(layout.anchor_positions[layout.block_keep_mask].tolist())
        counts = ()
        if self.metadata_counts:
            native = self.branches.native_denominators(layout)
            auxiliary = float(native.get('base', 0.))
            if self.config.indexer_config is not None:
                auxiliary = float((layout.block_keep_mask & layout.prediction_mask.bool().any(-1)
                                   & (layout.anchor_positions > 0)).sum())
            counts = (1., float(native['final']), auxiliary)
        shard = str(self.dataset.shard_key(index)) if hasattr(self.dataset, 'shard_key') else None
        return PlannedSample(index, epoch, source_position, anchors, length,
                             self.cost(length, layout), counts, shard=shard)

    def _build(self, start):
        stop = min(start + self.window * self.accumulation, self.total_batches)
        pool = []
        for step in range(start, stop):
            epoch, batch = divmod(step, self.steps_per_epoch)
            orders = self._orders(epoch)
            for rank in range(self.replicas):
                for slot in range(self.batch_size):
                    position = batch*self.batch_size + slot
                    pool.append(self._sample(orders[rank][position], epoch,
                        (step*self.replicas + rank)*self.batch_size + slot))
        planned = assign_microstep_batches(pool, self.replicas, self.batch_size)
        planned = shuffle_balanced_microsteps(planned, seed=self.args.seed,
                                             phase=self.phase, window_start=start)
        batches = [[] for _ in range(self.replicas)]
        for update_start in range(0, stop-start, self.accumulation):
            for rank in range(self.replicas):
                microsteps = planned[rank][update_start:update_start + self.accumulation]
                counts = tuple(tuple(sum(batch[slot].denominators[k] for batch in microsteps)
                                     for k in range(3)) for slot in range(self.batch_size)) if self.metadata_counts else ()
                batches[rank].extend([[{**asdict(s), 'update_counts': counts} for s in batch]
                                     for batch in microsteps])
        return dict(start=start, stop=stop, batches=batches,
                    round_order_version=ROUND_ORDER_VERSION)

    def _window_at(self, step):
        with self._lock:
            window = next((w for w in self.windows.values()
                           if w['start'] <= step < w['stop']), None)
            previous = max((w['stop'] for w in self.windows.values() if w['stop'] <= step), default=self.committed)
        if window is not None:
            self._prefetch(window['stop'])
            return window
        # After a saved pending window, a changed W only affects future work.
        start = previous
        while start + self.window*self.accumulation <= step:
            start += self.window*self.accumulation
        if self.async_planning:
            self._prefetch(start)
            if self._future_start != start:
                raise RuntimeError('Nonsequential async scheduler request')
            window = self._future.result()
            self._future = self._future_start = None
        else:
            window = self._build(start)
        with self._lock:
            self.windows[start] = window
        self._prefetch(window['stop'])
        return window

    def _prefetch(self, start):
        # Submitted only when the DataLoader asks for its first index: worker
        # processes have already been created, so no executor thread is forked.
        if not self.async_planning or self._closed or start >= self.total_batches or self._future is not None:
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='schedule-plan')
        self._future_start = start
        self._future = self._executor.submit(self._build, start)

    def close(self):
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def batches(self, start):
        for step in range(start, self.total_batches):
            window = self._window_at(step)
            yield [PlannedSample(**{**s, 'anchors': tuple(s['anchors'])})
                   for s in window['batches'][self.rank][step-window['start']]]

    def commit(self, step):
        with self._lock:
            self.committed = int(step)
            self.windows = {k: w for k, w in self.windows.items() if w['stop'] > step}

    def state_dict(self):
        with self._lock:
            pending = next((w for w in self.windows.values()
                            if w['start'] <= self.committed < w['stop']), None)
            return dict(signature=self.signature, committed=self.committed, pending=pending)


class PlannedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, request):
        return self.dataset[request.index], request


class PlannedCollator:
    def __init__(self, collator, block_size, num_anchors):
        self.collator, self.block_size, self.num_anchors = collator, block_size, num_anchors

    def __call__(self, rows):
        values, requests = zip(*rows)
        data = self.collator(list(values))
        width = max(self.num_anchors, max(len(r.anchors) for r in requests), 1)
        anchors = torch.zeros(len(rows), width, dtype=torch.long)
        keep = torch.zeros_like(anchors, dtype=torch.bool)
        for i, request in enumerate(requests):
            anchors[i, :len(request.anchors)] = torch.tensor(request.anchors)
            keep[i, :len(request.anchors)] = True
        data['_planned_layout'] = layout_from_anchors(data['input_ids'], data['loss_mask'],
                                                       anchors, keep, self.block_size)
        data['_planned_occurrences'] = requests
        if requests[0].update_counts:
            data['_planned_window_counts'] = torch.tensor(requests[0].update_counts, dtype=torch.float64)
        return data


def planned_batches(session, phase):
    loader, args = session.dataloader, session.args
    saved = (session.resume_state or {}).get('scheduler_plan')
    if saved is not None and saved['signature']['phase'] != phase.name:
        saved = None
    planner = WindowPlanner(loader, args, session.draft.config, phase=phase.name,
                            total_batches=phase.batch_count(session.steps_per_epoch), state=saved)
    if saved is not None and planner.committed != session.state.phase_step:
        raise ValueError('Training and scheduler cursors disagree')
    planner.commit(session.state.phase_step)
    session.training_planner = planner
    planned = DataLoader(PlannedDataset(loader.dataset),
        batch_sampler=planner.batches(session.state.phase_step),
        collate_fn=PlannedCollator(loader.collate_fn, session.draft.block_size, args.num_anchors),
        num_workers=loader.num_workers, pin_memory=loader.pin_memory,
        persistent_workers=loader.num_workers > 0,
        prefetch_factor=loader.prefetch_factor if loader.num_workers else None,
        generator=torch.Generator().manual_seed(args.seed))
    try:
        yield from planned
    finally:
        planner.close()
