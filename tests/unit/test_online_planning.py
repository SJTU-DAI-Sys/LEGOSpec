"""Online Arrow lookahead preserves optimizer-window and resume semantics."""
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, DistributedSampler

from legospec.training.config import resolve_execution_options
from legospec.training.data.planning import WindowPlanner
from test_vocab_reuse import wrapper


def test_online_async_is_opt_in_and_requires_optimized_collocation():
    args = SimpleNamespace(training_mode='optimized', train_hidden_states_path=None,
                           disaggregate=False, async_schedule_planning=True)
    assert resolve_execution_options(args).async_schedule_planning
    for mode, disaggregate in [('legacy', False), ('optimized', True)]:
        args = SimpleNamespace(training_mode=mode, train_hidden_states_path=None,
                               disaggregate=disaggregate, async_schedule_planning=True)
        with pytest.raises(ValueError, match='optimized collocated'):
            resolve_execution_options(args)


def test_online_arrow_plans_next_window_during_current_window_and_resumes():
    dataset = Dataset.from_list([
        dict(input_ids=[list(range(20+i%15))], attention_mask=[[1]*(20+i%15)],
             loss_mask=[[1]*(20+i%15)], sample_index=i)
        for i in range(96)
    ]).with_format('torch')
    sampler = DistributedSampler(dataset, num_replicas=3, rank=0, seed=7)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, drop_last=True)
    config = wrapper('soft_ce').draft_model.config
    args = SimpleNamespace(accumulation_steps=3, schedule_window_updates=2,
        train_hidden_states_path=None, training_cost_profile=None,
        anchor_sampling='sample_epoch', seed=42, num_anchors=8,
        complete_blocks=True, async_schedule_planning=False)
    synchronous = WindowPlanner(loader,args,config,phase='train',total_batches=16)
    expected = list(synchronous.batches(0))
    args = deepcopy(args); args.async_schedule_planning=True
    asynchronous = WindowPlanner(loader,args,config,phase='train',total_batches=16)
    started, release = Event(), Event()
    build = asynchronous._build
    def gated_build(start):
        if start == 6:
            started.set()
            assert release.wait(10)
        return build(start)
    asynchronous._build = gated_build
    rng = torch.get_rng_state().clone()
    try:
        iterator = asynchronous.batches(0)
        actual = [next(iterator)]
        assert started.wait(10)  # Next window is running before current is consumed.
        actual.extend(next(iterator) for _ in range(5))
        assert not asynchronous._future.done()
        asynchronous.commit(3)
        saved = deepcopy(asynchronous.state_dict())
        release.set()
        actual.extend(iterator)
        assert actual == expected
        torch.testing.assert_close(torch.get_rng_state(),rng)
        restored = WindowPlanner(loader,args,config,phase='train',total_batches=16,state=saved)
        try:
            assert list(restored.batches(3)) == expected[3:]
        finally:
            restored.close()
    finally:
        release.set()
        asynchronous.close()


def test_online_loader_uses_configured_prefetch(monkeypatch):
    from legospec.training.data import loader as module
    recorded={}
    class Rows:
        def filter(self,*args,**kwargs): return self
    def capture(dataset,batch_size,**kwargs):
        recorded.update(kwargs)
        return [None]
    monkeypatch.setattr(module,'get_dp_group',lambda:None)
    monkeypatch.setattr(module,'prepare_dp_dataloaders',capture)
    args=SimpleNamespace(block_size=8,batch_size=2,dataloader_num_workers=1,
        build_dataset_num_proc=128,
        dataloader_prefetch_factor=3,pad_to_max_length=False)
    module._prepare_dataloader(args,Rows(),train_data_path='tokens')
    assert recorded['prefetch_factor']==3
