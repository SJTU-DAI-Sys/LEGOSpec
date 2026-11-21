"""Stream hidden batches with the same window loss and bounded receive lifetime."""
from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch
import threading

import pytest
import torch

from test_vocab_reuse import wrapper
from litespec.core.composable.anchor_sampling import prepare_sample_epoch_layout
from litespec.core.composable.batch import layout_from_anchors, PreparedLogicalBatch
from litespec.core.composable.objectives.selector import LossBranches
from litespec.core.composable.packets import ComposableBatchPacket, ComposablePacketSpec
from litespec.core.composable.streaming_windows import plan_target_windows
from litespec.distributed.transport import PacketReceiver, NodePacketTransport, build_node_routes
from litespec.training.recipes.composable import ComposableTrainingAdapter
from litespec.training.runner import run_target_phase


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def model_and_branches():
    model = wrapper('dpal_renyi', 'selector')
    recipe = model.draft_model.config.composable_config
    recipe['objective']['base'] = dict(type='soft_ce', weight=.2)
    recipe['objective']['final']['rho'] = .3
    model.loss_branches = LossBranches(recipe)
    return model, LossBranches(model.draft_model.config.composable_config)


def batches(count, batch_size=1):
    for step in range(count):
        ids = (torch.arange(24)[None].expand(batch_size, -1) + step) % 31
        valid = torch.ones_like(ids)
        mask = torch.ones_like(ids, dtype=torch.float32)
        for row in range(batch_size):
            length = 16 + (step + row) % 9
            valid[row, length:] = 0
            mask[row, length:] = 0
            mask[row, :2 + step % 3] = 0
        yield dict(input_ids=ids, attention_mask=valid, loss_mask=mask, order=step)


@pytest.mark.parametrize('accumulation', [3, 8])
@pytest.mark.parametrize('producers,drafts,node_batch', [(6, 2, 6), (2, 3, 6), (4, 2, 8)])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_counts_routing_tail_cross_epoch_and_rng_match_existing_windows(accumulation, producers, drafts, node_batch, asynchronous):
    _, branches = model_and_branches()
    count = accumulation + 2
    source = list(batches(count, node_batch))
    routes = build_node_routes(producers=producers, drafts=drafts, node_batch_size=node_batch)
    width, draft_width = node_batch // producers, node_batch // drafts
    rng = torch.get_rng_state().clone()
    plans = []
    for producer in range(producers):
        rows = [{k: (v[producer * width:(producer + 1) * width] if torch.is_tensor(v) else v)
                 for k, v in row.items()} for row in source]
        plans.append(list(plan_target_windows(rows, branches=branches, block_size=8,
            num_anchors=4, seed=42, accumulation_steps=accumulation, steps_per_epoch=2,
            asynchronous=asynchronous)))
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert all([row['order'] for row in plan] == list(range(count)) for plan in plans)
    assert all('_streamed_window_counts' not in row for row in source)
    for start in range(0, count, accumulation):
        stop = min(count, start + accumulation)
        global_counts = torch.zeros(3, dtype=torch.float64)
        for draft in range(drafts):
            expected, received = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
            for route in routes:
                if route.draft == draft:
                    received += plans[route.producer][start]['_streamed_window_counts'][
                        route.producer_start:route.producer_end].sum(0)
            for step in range(start, stop):
                row = source[step]
                sl = slice(draft * draft_width, (draft + 1) * draft_width)
                old = prepare_sample_epoch_layout(row['input_ids'][sl], row['loss_mask'][sl],
                    row['attention_mask'][sl], block_size=8, num_anchors=4, seed=42,
                    epoch=start // 2)
                native = branches.native_denominators(old)
                expected += torch.tensor([draft_width, float(native['final']), float(native['base'])])
            torch.testing.assert_close(received, expected, rtol=0, atol=0)
            global_counts += received
        assert global_counts[0] == (stop - start) * node_batch
    # An optimizer-boundary resume recreates the same pending metadata by replay.
    replay = list(plan_target_windows(list(batches(count, width)), branches=branches,
        block_size=8, num_anchors=4, seed=42, accumulation_steps=accumulation,
        steps_per_epoch=2, asynchronous=asynchronous))
    for a, b in zip(replay[accumulation:], plans[0][accumulation:], strict=True):
        # Producer zero has the same input and masks as these standalone rows.
        torch.testing.assert_close(a['_streamed_anchor_positions'], b['_streamed_anchor_positions'])


def packet(row, device='cpu'):
    b, length = row['input_ids'].shape
    return ComposableBatchPacket(
        row['input_ids'].to(device), row['loss_mask'].to(device),
        torch.randn(b, length, 32, device=device), torch.randn(b, length, 16, device=device),
        row['attention_mask'].bool().to(device),
        row['_streamed_anchor_positions'].to(device), row['_streamed_block_keep_mask'].to(device),
        row['_streamed_window_counts'].to(device))


def test_no_hidden_host_copy_spill_or_lookahead_on_consumer(monkeypatch):
    model, branches = model_and_branches()
    row = next(plan_target_windows(batches(8, 3), branches=branches, block_size=8,
        num_anchors=4, seed=42, accumulation_steps=8, steps_per_epoch=11))
    data = packet(row)
    session = SimpleNamespace(args=SimpleNamespace(opd_data_mode='teacher_forced'),
        draft=SimpleNamespace(config=model.draft_model.config, block_size=8),
        topology=object(), process_group=None)
    adapter = ComposableTrainingAdapter('composable', 'ComposableDraftModel')
    def forbidden(*a, **kw):
        raise AssertionError('Streaming consumer touched host storage or requested another packet')
    class NoLookahead:
        def __next__(self):
            return forbidden()
    monkeypatch.setattr(torch.Tensor, 'cpu', forbidden)
    monkeypatch.setattr(PreparedLogicalBatch, 'spill', forbidden)
    assert adapter.prepare_optimizer_window(session, data, NoLookahead(), 8) is None
    expected = row['_streamed_window_counts'].sum(0).clone()
    data.window_counts.zero_()  # Receive-ring reuse cannot overwrite the saved denominator.
    torch.testing.assert_close(session.planned_window_denominators, expected)
    layout = adapter.anchor_layout(session, data, microbatch_start=1, microbatch_end=2)
    torch.testing.assert_close(layout.anchor_positions, data.anchor_positions[1:2])


def test_streaming_preserves_selector_dpal_loss_and_gradients():
    old_model, branches = model_and_branches()
    new_model = deepcopy(old_model)
    rows = list(plan_target_windows(batches(3), branches=branches, block_size=8,
        num_anchors=4, seed=42, accumulation_steps=3, steps_per_epoch=2, asynchronous=True))
    totals = rows[0]['_streamed_window_counts'].sum(0)
    losses = [[], []]
    for row in rows:
        data = packet(row)
        old_layout = prepare_sample_epoch_layout(data.input_ids, data.loss_mask,
            data.context_valid_mask, block_size=8, num_anchors=4, seed=42, epoch=0)
        streamed_layout = layout_from_anchors(data.input_ids, data.loss_mask * data.context_valid_mask,
            data.anchor_positions, data.block_keep_mask, 8)
        for i, (model, layout) in enumerate([(old_model, old_layout), (new_model, streamed_layout)]):
            out = model(input_ids=data.input_ids, loss_mask=data.loss_mask,
                target_hidden_states=data.target_context_hidden,
                target_last_hidden_states=data.target_last_hidden, context_valid_mask=data.context_valid_mask,
                layout=layout, window_denominators=totals, collect_detailed_metrics=False)
            losses[i].append(out.loss.detach())
            out.loss.backward()
    torch.testing.assert_close(torch.stack(losses[0]), torch.stack(losses[1]), rtol=0, atol=0)
    for (name, a), (_, b) in zip(old_model.named_parameters(), new_model.named_parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)


def test_metadata_prefetch_overlaps_caller_and_is_bounded(monkeypatch):
    import litespec.core.composable.streaming_windows as module
    _, branches = model_and_branches()
    main_thread = threading.get_ident()
    worker_entered, release_worker = threading.Event(), threading.Event()
    original, worker_threads, fetched = module._prepare_window, [], []
    def source():
        for row in batches(11):
            assert threading.get_ident() == main_thread
            fetched.append(row['order'])
            yield row
    def prepare(window, start, **kw):
        worker_threads.append(threading.get_ident())
        if start == 3:
            worker_entered.set()
            assert release_worker.wait(10), 'Test failed to release background metadata worker'
        return original(window, start, **kw)
    monkeypatch.setattr(module, '_prepare_window', prepare)
    stream = plan_target_windows(source(), branches=branches, block_size=8,
        num_anchors=4, seed=42, accumulation_steps=3, steps_per_epoch=2, asynchronous=True)
    try:
        first = next(stream)
        assert first['order'] == 0
        assert worker_entered.wait(5)
        assert len(fetched) == 6  # Current plus one future CPU window; no hidden lookahead.
        assert all(thread != main_thread for thread in worker_threads)
        release_worker.set()
        assert [first['order'], *[row['order'] for row in stream]] == list(range(11))
    finally:
        release_worker.set()
        stream.close()


def test_background_metadata_error_is_propagated(monkeypatch):
    import litespec.core.composable.streaming_windows as module
    _, branches = model_and_branches()
    original = module._prepare_window
    def prepare(window, start, **kw):
        if start == 3:
            raise RuntimeError('metadata worker fixture failure')
        return original(window, start, **kw)
    monkeypatch.setattr(module, '_prepare_window', prepare)
    stream = plan_target_windows(batches(6), branches=branches, block_size=8,
        num_anchors=4, seed=42, accumulation_steps=3, steps_per_epoch=2, asynchronous=True)
    assert [next(stream)['order'] for _ in range(3)] == [0, 1, 2]
    with pytest.raises(RuntimeError, match='metadata worker fixture failure'):
        next(stream)


def test_wire_schema_and_target_send_order_resume_and_partial_window():
    _, branches = model_and_branches()
    old_spec = ComposablePacketSpec(1, 24, 32, 16, True)
    spec = ComposablePacketSpec(1, 24, 32, 16, True, 4)
    assert old_spec.schema_id == 31 and spec.schema_id == 33
    empty = ComposableBatchPacket.empty(spec, device='cpu')
    assert len(list(empty.tensors())) == 8
    assert len(list(ComposableBatchPacket.empty(old_spec, device='cpu').tensors())) == 5
    class Loader:
        def __init__(self):
            self.epoch = 0
            self.sampler = SimpleNamespace(set_epoch=lambda e: setattr(self, 'epoch', e))
        def __iter__(self):
            return iter(list(batches(6))[self.epoch * 2:self.epoch * 2 + 2])
    produced, sent = [], []
    transport = SimpleNamespace(wait_send=lambda _: 0.,
        send=lambda p, **kw: sent.append((kw['batch_id'], p)) or object())
    def produce(data, index):
        produced.append(index)
        return packet(data)
    result = run_target_phase(args=SimpleNamespace(pipeline_depth=2),
        topology=SimpleNamespace(is_target_leader=True), dataloader=Loader(),
        steps=2, epochs=3, epoch_offset=0, start_id=3, batch_id=0, limit=5,
        transport=transport, phase_id=0, schema_id=33, broadcast=lambda d: d, produce=produce,
        replay_skipped=True, prepare_batches=lambda it: plan_target_windows(it, branches=branches,
            block_size=8, num_anchors=4, seed=42, accumulation_steps=3, steps_per_epoch=2))
    assert result == (5, 0.) and produced == list(range(5))
    assert [index for index, _ in sent] == [3, 4]
    assert all(p.window_counts[0, 0] == 2 for _, p in sent)


@pytest.mark.parametrize('leader', [False, True])
@pytest.mark.parametrize('replay', [False, True])
@pytest.mark.parametrize('limit', [0, 5, 9])
def test_other_target_backends_keep_order_and_resume_without_planning(leader, replay, limit):
    class Loader:
        def __init__(self):
            self.epoch = 0
            self.sampler = SimpleNamespace(set_epoch=lambda e: setattr(self, 'epoch', e))
        def __iter__(self):
            return iter(range(self.epoch * 3, self.epoch * 3 + 3))
    produced, sent = [], []
    transport = SimpleNamespace(wait_send=lambda _: 0.,
        send=lambda p, **kw: sent.append((kw['batch_id'], p)) or object())
    def produce(data, index):
        produced.append((index, data))
        return data
    end, _ = run_target_phase(args=SimpleNamespace(pipeline_depth=2),
        topology=SimpleNamespace(is_target_leader=leader), dataloader=Loader() if leader else None,
        steps=3, epochs=3, epoch_offset=0, start_id=14, batch_id=12, limit=limit,
        transport=transport if leader else None, phase_id=0, schema_id=31,
        broadcast=lambda d: d, produce=produce, replay_skipped=replay)
    assert end == 12 + limit
    assert produced == [(12 + i, i if leader else None) for i in range(limit)
                        if replay or 12 + i >= 14]
    assert sent == ([(12 + i, i) for i in range(limit) if 12 + i >= 14] if leader else [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA receive-stream lifetime check')
def test_async_receive_ring_waits_for_consumer_kernels():
    class Transport:
        release_slot = NodePacketTransport.release_slot
        def __init__(self):
            self.stream = torch.cuda.Stream(priority=-1)
        def receive(self, slot, *, batch_id, **unused):
            with torch.cuda.stream(self.stream):
                slot.fill_(batch_id)
                ready = torch.cuda.Event()
                ready.record()
            return ready
        def wait_receive(self, ready):
            ready.synchronize()
            return 0.
    slots = [torch.empty(1024, device='cuda') for _ in range(2)]
    snapshots = []
    for index, value in PacketReceiver(Transport(), slots, start=0, stop=9, phase_id=0, schema_id=33):
        torch.cuda._sleep(4_000_000)
        snapshots.append((index, value.clone()))
    torch.cuda.synchronize()
    for index, value in snapshots:
        assert torch.all(value == index)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Pinned metadata H2D transfer check')
def test_metadata_is_pinned_and_transferred_without_host_synchronization(monkeypatch):
    model, branches = model_and_branches()
    row = next(plan_target_windows(batches(1), branches=branches, block_size=8,
        num_anchors=4, seed=42, accumulation_steps=8, steps_per_epoch=2, asynchronous=True))
    data = packet(row, device='cuda')
    session = SimpleNamespace(packet_metadata={'streamed_window_recipe': branches.recipe})
    adapter = ComposableTrainingAdapter('composable', 'ComposableDraftModel')
    transfers, original = [], torch.Tensor.to
    def transfer(value, *args, **kwargs):
        transfers.append((value.is_pinned(), kwargs.get('non_blocking')))
        return original(value, *args, **kwargs)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), patch.object(torch.Tensor, 'to', transfer):
        torch.cuda._sleep(2_000_000)
        adapter.annotate_target_packet(session, None, data, row)
    stream.synchronize()
    assert transfers == [(True, True)] * 3
    for field in ('anchor_positions', 'block_keep_mask', 'window_counts'):
        torch.testing.assert_close(getattr(data, field).cpu(), row['_streamed_' + field])
