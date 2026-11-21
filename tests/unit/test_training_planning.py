from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler
from test_vocab_reuse import wrapper
from litespec.training.planning import (WindowPlanner, PlannedCollator, PlannedDataset,
                                      PlannedSample, assign_microstep_batches,
                                      shuffle_balanced_microsteps)
from litespec.core.composable.anchor_sampling import prepare_sample_epoch_layout

class MetadataDataset:
    def __init__(self, n=47):
        self.n = n
        self.loaded = 0
    def __len__(self):
        return self.n
    def training_metadata(self, i):
        length = 20 + i % 15
        return dict(input_ids=torch.arange(length)[None] % 31,
                    attention_mask=torch.ones(1, length), loss_mask=torch.ones(1, length))
    def __getitem__(self, i):
        self.loaded += 1
        return self.training_metadata(i)

def collate(values):
    length = max(v['input_ids'].numel() for v in values)
    return {k: torch.cat([torch.nn.functional.pad(v[k], (0, length-v[k].numel()))
                         for v in values]) for k in values[0]}

def planner(rank=0, w=1, state=None, n=47):
    data = MetadataDataset(n)
    sampler = DistributedSampler(data, num_replicas=3, rank=rank, seed=7)
    loader = DataLoader(data, batch_size=2, sampler=sampler, drop_last=True, collate_fn=collate)
    args = SimpleNamespace(accumulation_steps=3, schedule_window_updates=w,
        train_hidden_states_path='cache', training_cost_profile=None,
        anchor_sampling='sample_epoch', seed=42, num_anchors=8, complete_blocks=True)
    config = wrapper('soft_ce').draft_model.config
    return WindowPlanner(loader, args, config, phase='train', total_batches=2*len(loader), state=state)


def cost_samples(costs, lengths=None):
    lengths = lengths if lengths is not None else costs
    return [PlannedSample(i, 0, i, (1,), lengths[i], cost)
            for i, cost in enumerate(costs)]


def synchronized_cost(batches):
    return sum(max(sum(s.cost for s in rank[step]) for rank in batches)
               for step in range(len(batches[0])))


def test_microstep_cost_not_rank_total_or_length_order():
    samples = cost_samples([9, 8, 2, 1], [400, 100, 200, 300])
    # Equal rank totals (10 each) still cost 17 with these synchronized rounds.
    old = [[[samples[0]], [samples[3]]], [[samples[2]], [samples[1]]]]
    planned = assign_microstep_batches(samples, 2, 1)
    assert synchronized_cost(old) == 17
    assert synchronized_cost(planned) == 11
    assert [{s.cost for rank in planned for s in rank[step]} for step in range(2)] == [{9, 8}, {2, 1}]


def test_microstep_balances_multisample_batch_costs():
    planned = assign_microstep_batches(cost_samples([10, 9, 8, 7, 4, 3, 2, 1]), 2, 2)
    loads = [[sum(s.cost for s in rank[step]) for rank in planned] for step in range(2)]
    assert loads == [[17, 17], [5, 5]]
    assert synchronized_cost(planned) == 22


def test_shuffle_preserves_balanced_rounds_cost_occurrences_and_rng():
    samples = cost_samples([1000-i for i in range(16*8*2)])
    balanced = assign_microstep_batches(samples, 8, 2)
    rng = torch.get_rng_state().clone()
    shuffled = shuffle_balanced_microsteps(balanced, seed=42, phase='train', window_start=0)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    signature = lambda rounds: [tuple(tuple(s.source_position for s in rank[step])
                                     for rank in rounds) for step in range(len(rounds[0]))]
    assert signature(shuffled) != signature(balanced)
    assert Counter(signature(shuffled)) == Counter(signature(balanced))
    assert synchronized_cost(shuffled) == synchronized_cost(balanced)
    assert shuffled == shuffle_balanced_microsteps(balanced, seed=42, phase='train', window_start=0)
    assert shuffled != shuffle_balanced_microsteps(balanced, seed=42, phase='train', window_start=128)
    assert shuffled != shuffle_balanced_microsteps(balanced, seed=43, phase='train', window_start=0)


def test_planner_applies_identical_round_permutation_on_all_ranks(monkeypatch):
    import litespec.training.planning as module
    p = planner(w=4)
    identity = lambda batches, **kw: batches
    with monkeypatch.context() as patch:
        patch.setattr(module, 'shuffle_balanced_microsteps', identity)
        baseline = p._build(0)
    actual = [planner(rank=r, w=4)._build(0) for r in range(3)]
    assert all(window == actual[0] for window in actual)
    signature = lambda window: [tuple(tuple(s['source_position'] for s in rank[step])
                                     for rank in window['batches'])
                                for step in range(window['stop'])]
    assert signature(actual[0]) != signature(baseline)
    assert Counter(signature(actual[0])) == Counter(signature(baseline))


@pytest.mark.parametrize('replicas,batch_size', [(1, 1), (3, 2), (16, 1), (4, 3)])
def test_microstep_assignment_is_deterministic_and_keeps_full_batches(replicas, batch_size):
    samples = cost_samples([1 + (i * 13) % 29 for i in range(replicas * batch_size * 8)])
    planned = assign_microstep_batches(samples, replicas, batch_size)
    assert planned == assign_microstep_batches(list(reversed(samples)), replicas, batch_size)
    assert all(len(rank) == 8 and all(len(b) == batch_size for b in rank) for rank in planned)
    assert Counter(s.source_position for rank in planned for batch in rank for s in batch) == Counter(range(len(samples)))


def test_pending_checkpoint_assignments_are_not_rescheduled():
    p = planner(w=2)
    next(p.batches(0))
    p.commit(3)
    saved = deepcopy(p.state_dict())
    saved['pending'].pop('round_order_version', None)
    # An already persisted plan is authoritative even if its ordering differs
    # from the current heuristic (e.g. a checkpoint from the older planner).
    for rank in saved['pending']['batches']:
        rank[3:] = list(reversed(rank[3:]))
    restored = planner(w=2, state=saved)
    count = saved['pending']['stop'] - 3
    actual = list(restored.batches(3))[:count]
    assert [[s.source_position for s in batch] for batch in actual] == [
        [s['source_position'] for s in batch] for batch in saved['pending']['batches'][0][3:]]

@pytest.mark.parametrize('w', [1, 2, 4])
def test_preserves_occurrences_drop_last_and_cross_epoch(w):
    p = planner(w=w)
    ranks = [list(planner(rank=r, w=w).batches(0)) for r in range(3)]
    span = w*3
    for start in range(0, p.total_batches, span):
        stop = min(p.total_batches, start+span)
        actual = Counter((s.index,s.epoch,s.source_position)
                         for batches in ranks for batch in batches[start:stop] for s in batch)
        expected = Counter()
        for step in range(start, stop):
            epoch, batch = divmod(step, p.steps_per_epoch)
            for r, order in enumerate(p._orders(epoch)):
                for slot in range(2):
                    expected[order[batch*2+slot], epoch, (step*3+r)*2+slot] += 1
        assert actual == expected
    assert p.dataset.loaded == 0

@pytest.mark.parametrize('w', [1,2,4])
def test_resume_prefetch_and_changed_window(w):
    p = planner(w=w)
    iterator = iter(p.batches(0))
    consumed = [next(iterator) for _ in range(3)]
    # Simulate workers looking ahead; only the runner commits 3 batches.
    prefetched = [next(iterator) for _ in range(4)]
    p.commit(3)
    state = deepcopy(p.state_dict())
    expected = list(planner(w=w).batches(0))[3:]
    resumed = planner(w=w, state=state)
    assert list(resumed.batches(3)) == expected
    changed = planner(w=3, state=state)
    actual = list(changed.batches(3))
    if state['pending']:
        count = state['pending']['stop'] - 3
        assert actual[:count] == expected[:count]
    assert len(actual) == p.total_batches - 3
    assert p.dataset.loaded == 0


def test_planned_collation_keeps_anchors_after_reordering():
    p = planner(w=2)
    batch = next(p.batches(0))
    loader = PlannedDataset(p.dataset)
    data = PlannedCollator(collate, p.config.block_size, 8)([loader[r] for r in batch])
    for i, r in enumerate(batch):
        source = p.dataset.training_metadata(r.index)
        expected = prepare_sample_epoch_layout(source['input_ids'], source['loss_mask'], source['attention_mask'],
            block_size=p.config.block_size, num_anchors=8, seed=42, epoch=r.epoch)
        torch.testing.assert_close(data['_planned_layout'].anchor_positions[i], expected.anchor_positions[0])
        torch.testing.assert_close(data['_planned_layout'].prediction_mask[i], expected.prediction_mask[0])


def test_rejects_topology_change_on_resume():
    p=planner(w=4)
    next(p.batches(0)); p.commit(3)
    state=p.state_dict(); state['signature']['replicas']=4
    with pytest.raises(ValueError, match='incompatible'):
        planner(state=state)


@pytest.mark.parametrize('w', [1, 2, 4])
def test_async_preserves_assignment_rng_and_resume(w):
    synchronous = planner(w=w)
    expected = list(synchronous.batches(0))
    asynchronous = planner(w=w)
    asynchronous.async_planning = True
    rng = torch.get_rng_state().clone()
    try:
        iterator = asynchronous.batches(0)
        actual = [next(iterator) for _ in range(7)]
        asynchronous.commit(3)
        saved = deepcopy(asynchronous.state_dict())
        assert saved['committed'] == 3
        actual.extend(iterator)
        assert actual == expected
        torch.testing.assert_close(torch.get_rng_state(), rng)
        restored = planner(w=3, state=saved)
        restored.async_planning = True
        try:
            resumed = list(restored.batches(3))
            count = saved['pending']['stop'] - 3 if saved['pending'] else 0
            assert resumed[:count] == expected[3:3+count]
            assert len(resumed) == len(expected) - 3
        finally:
            restored.close()
    finally:
        asynchronous.close()


def test_async_error_reaches_consumer():
    p = planner()
    p.async_planning = True
    def fail(start):
        raise ValueError('planning failed')
    p._build = fail
    try:
        with pytest.raises(ValueError, match='planning failed'):
            next(p.batches(0))
    finally:
        p.close()


def test_metadata_window_counts_match_materialized_layouts():
    p=planner(w=2)
    batches=list(p.batches(0))
    for start in range(0,len(batches),3):
        actual=torch.zeros(2,3,dtype=torch.float64)
        for requests in batches[start:start+3]:
            data=PlannedCollator(collate,p.config.block_size,8)([PlannedDataset(p.dataset)[r] for r in requests])
            for slot in range(2):
                layout=data['_planned_layout'].batch_slice(slot,slot+1)
                native=p.branches.native_denominators(layout)
                actual[slot] += torch.tensor([1.,float(native['final']),float(native.get('base',0.))])
        torch.testing.assert_close(actual,torch.tensor(batches[start][0].update_counts,dtype=torch.float64))


def test_resume_skips_hidden_io_with_worker_prefetch():
    p=planner(w=3)
    reference=list(p.batches(0))
    # Save while a window is pending, after an optimizer update.
    p.commit(3)
    state=p.state_dict()
    restored=planner(w=3,state=state)
    loader=DataLoader(PlannedDataset(restored.dataset),batch_sampler=restored.batches(3),
        collate_fn=PlannedCollator(collate,restored.config.block_size,8),num_workers=2)
    received=[list(batch['_planned_occurrences']) for batch in loader]
    assert received==reference[3:]

@pytest.mark.parametrize('chunk',[0,16])
def test_checkpointing_defaults_follow_chunking(chunk):
    from litespec.training.recipes.composable import parse_args
    argv=['--target-model-path','target','--train-data-path','data.jsonl',
          '--output-dir','out','--report-to','none','--objective-chunk-blocks',str(chunk)]
    args=parse_args(argv=argv)
    assert args.objective_gradient_checkpointing == bool(chunk)
    assert args.backbone_gradient_checkpointing == bool(chunk)
    args=parse_args(argv=argv+['--objective-gradient-checkpointing','--no-backbone-gradient-checkpointing'])
    assert args.objective_gradient_checkpointing and not args.backbone_gradient_checkpointing


def test_documented_launcher_options_preserve_full_resume():
    from litespec.training.recipes.composable import parse_args
    argv=['--target-model-path','target','--train-data-path','data.jsonl',
          '--output-dir','out','--report-to','none','--training-mode','optimized',
          '--schedule-window-updates','1','--objective-chunk-blocks','0',
          '--save-interval','15000','--save-strategy','interval','--save-training-state']
    args=parse_args(argv=argv)
    assert args.save_training_state and args.save_interval==15000 and args.save_strategy=='interval'
    with pytest.raises(SystemExit):
        parse_args(argv=argv+['--no-save-training-state'])
