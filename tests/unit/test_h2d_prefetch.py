import pytest
import torch

from legospec.training.data.prefetch import cuda_prefetched_batches
from legospec.training.config import resolve_execution_options
from types import SimpleNamespace


def test_profile_and_override():
    for mode, offline, expected in [('optimized', True, False), ('legacy', True, False), ('optimized', False, False)]:
        args = SimpleNamespace(training_mode=mode, train_hidden_states_path='cache' if offline else None)
        resolve_execution_options(args)
        assert args.h2d_prefetch is expected
    args.h2d_prefetch = True
    with pytest.raises(ValueError, match='optimized collocated offline'):
        resolve_execution_options(args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA H2D lifetime test')
def test_variable_batches_keep_cpu_metadata_and_cuda_storage_alive():
    batches = []
    for length in [13, 29, 7, 19]:
        batch = {name: torch.arange(length).reshape(1, -1).pin_memory()
                 for name in ('input_ids', 'attention_mask', 'loss_mask')}
        batch['hidden_states'] = {i: torch.full((1, length, 4), float(length+i)).pin_memory() for i in range(3)}
        batch['final_hidden_state'] = torch.full((1, length, 4), float(length)).pin_memory()
        batches.append(batch)
    outputs = []
    for expected, actual in zip(batches, cuda_prefetched_batches(batches, [0, 2], True), strict=True):
        assert actual['input_ids'] is expected['input_ids']
        assert actual['hidden_states'] is expected['hidden_states']
        assert set(actual['_cuda_inputs']['hidden_states']) == {0, 2}
        outputs.append(actual['_cuda_inputs']['hidden_states'][2].square())
    torch.cuda.synchronize()
    for expected, actual in zip(batches, outputs, strict=True):
        torch.testing.assert_close(actual.cpu(), expected['hidden_states'][2].square())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA iterator cleanup')
def test_early_close_releases_source_and_empty_input():
    closed = []
    def source():
        try:
            for _ in range(5):
                value = torch.ones(2, 7).pin_memory()
                yield dict(input_ids=value, attention_mask=value, loss_mask=value, hidden_states={0: value})
        finally:
            closed.append(True)
    iterator = cuda_prefetched_batches(source(), [0], False)
    next(iterator)
    iterator.close()
    assert closed == [True]
    assert list(cuda_prefetched_batches([], [0], False)) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runner integration")
def test_runner_uses_cuda_prefetch(monkeypatch):
    from legospec.training.runtime import runner
    value = torch.ones(1, 8).pin_memory()
    data = dict(input_ids=value, attention_mask=value, loss_mask=value,
                hidden_states={0: value}, final_hidden_state=value)
    session = SimpleNamespace(
        topology=None, draft=SimpleNamespace(target_layer_ids=[0]),
        args=SimpleNamespace(cpu_batch_prefetch=0, train_hidden_states_path="cache",
                             h2d_prefetch=True, require_target_last_hidden=True))
    monkeypatch.setattr(runner, "_iter_logical_batches", lambda *_: iter([data, data]))
    batches = list(runner.iter_logical_batches(session, None))
    assert len(batches) == 2
    assert all(batch["_cuda_inputs"]["input_ids"].is_cuda for batch in batches)
    assert all(batch["input_ids"] is value for batch in batches)
