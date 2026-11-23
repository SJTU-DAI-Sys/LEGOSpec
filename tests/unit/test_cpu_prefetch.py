import legospec.training.runtime.runner as runtime_runner
import threading
from types import SimpleNamespace

import pytest

from legospec.training.data.prefetch import OrderedCPUPrefetch, ordered_cpu_batches


def test_order_and_first_fetch_on_caller():
    owners = []

    def source():
        for i in range(20):
            owners.append(threading.current_thread().ident)
            yield i

    caller = threading.current_thread().ident
    assert list(ordered_cpu_batches(source(), 2)) == list(range(20))
    assert owners[0] == caller
    assert all(owner != caller for owner in owners[1:])


def test_error_propagates_and_source_closes():
    closed = threading.Event()

    def source():
        try:
            yield 1
            raise ValueError("bad input")
        finally:
            closed.set()

    iterator = OrderedCPUPrefetch(source(), 2)
    assert next(iterator) == 1
    with pytest.raises(ValueError, match="bad input"):
        next(iterator)
    assert closed.wait(1)
    assert not iterator.thread.is_alive()


def test_early_close_is_bounded():
    produced = []
    reached = threading.Event()

    def source():
        try:
            for i in range(100):
                produced.append(i)
                if len(produced) == 4:
                    reached.set()
                yield i
        finally:
            reached.set()

    iterator = OrderedCPUPrefetch(source(), 2)
    assert reached.wait(2)
    iterator.close()
    assert not iterator.thread.is_alive()
    # One caller-held item, two queued items and at most one blocked put.
    assert len(produced) <= 4


def test_empty_and_disabled():
    assert list(ordered_cpu_batches(iter(()), 1)) == []
    assert list(ordered_cpu_batches(iter(range(3)), 0)) == [0, 1, 2]


def test_legacy_prefetch_uses_snapshot_cursor_and_phase_limit(monkeypatch):
    from legospec.training.runtime import runner as runner

    def epoch_rows(loader, epoch, *, start_batch, description):
        for index in range(start_batch, 5):
            yield index, {"id": (epoch, index)}

    monkeypatch.setattr(runner, "iter_epoch", epoch_rows)
    state = SimpleNamespace(epoch=1, next_batch=2, phase_step=7)
    session = SimpleNamespace(
        state=state,
        steps_per_epoch=5,
        topology=None,
        dataloader=None,
        args=SimpleNamespace(
            cpu_batch_prefetch=2,
            train_hidden_states_path="cache",
            training_mode="legacy",
            anchor_sampling="sample_epoch",
        ),
    )
    phase = SimpleNamespace(name="train", epochs=3, batch_count=lambda steps: 12)
    iterator = runtime_runner.iter_logical_batches(session, phase)
    first = next(iterator)
    # Consumer cursor changes must neither truncate nor extend the producer.
    state.epoch = 99
    state.next_batch = 99
    state.phase_step = 99
    rows = [first, *iterator]
    assert [r["id"] for r in rows] == [(1, 2), (1, 3), (1, 4), (2, 0), (2, 1)]
    assert [r["_anchor_epoch"] for r in rows] == [1, 1, 1, 2, 2]
