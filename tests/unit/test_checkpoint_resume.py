"""A resumed optimizer update must match uninterrupted execution exactly."""

import copy
from dataclasses import asdict

import torch
import pytest

from legospec.training.runtime.checkpoint import (
    capture_rng_state,
    restore_rng_state,
    save_distributed_training_state,
    load_distributed_training_state,
)
from legospec.training.optim.optimizer import BF16Optimizer
from legospec.training.runtime.state import RunState


@pytest.mark.parametrize("muon_config", [None, {}], ids=["adamw", "muon_adamw"])
def test_full_optimizer_rng_and_cursor_resume(tmp_path, muon_config):
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 3).to(torch.bfloat16)
    optimizer = BF16Optimizer(
        model, lr=0.01, total_steps=10, warmup_ratio=0.2, muon_config=muon_config
    )

    def update(model, optimizer):
        inputs = torch.randn(5, 4).bfloat16()
        model(inputs).float().square().mean().backward()
        optimizer.step()

    update(model, optimizer)
    cursor = RunState(
        epoch=2, next_batch=7, phase_step=7, global_step=7, optimizer_step=1
    )
    save_distributed_training_state(
        str(tmp_path),
        {
            **optimizer.state_dict(),
            "cursor": asdict(cursor),
            "rng_state": capture_rng_state(),
        },
    )
    update(model, optimizer)
    expected = copy.deepcopy(optimizer.state_dict())
    resumed_model = torch.nn.Linear(4, 3).to(torch.bfloat16)
    resumed_optimizer = BF16Optimizer(
        resumed_model,
        lr=0.01,
        total_steps=10,
        warmup_ratio=0.2,
        muon_config=muon_config,
    )
    state = load_distributed_training_state(str(tmp_path))
    assert state["cursor"] == asdict(cursor)
    resumed_optimizer.load_state_dict(state)
    restore_rng_state(state["rng_state"])
    update(resumed_model, resumed_optimizer)
    actual = resumed_optimizer.state_dict()
    assert actual["scheduler_state_dict"] == expected["scheduler_state_dict"]
    for left, right in zip(
        actual["fp32_master_params"], expected["fp32_master_params"]
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(resumed_model.parameters(), model.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    expected_opts = expected["optimizer_state_dict"]
    actual_opts = actual["optimizer_state_dict"]
    pairs = (
        [(actual_opts[name], expected_opts[name]) for name in ("adamw", "muon")]
        if muon_config is not None
        else [(actual_opts, expected_opts)]
    )
    for actual_opt, expected_opt in pairs:
        assert actual_opt["param_groups"] == expected_opt["param_groups"]
        for key, values in expected_opt["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(
                    actual_opt["state"][key][name],
                    value,
                    rtol=0,
                    atol=0,
                )


def test_each_rank_restores_its_own_tensors_and_rng(tmp_path):
    for rank in (0, 1):
        torch.manual_seed(42 + rank)
        state = {
            "cursor": asdict(RunState(global_step=7, optimizer_step=1)),
            "optimizer_state_dict": {"moment": torch.randn(3)},
            "scheduler_state_dict": {"last_epoch": 1},
            "fp32_master_params": [torch.randn(3)],
            "rng_state": capture_rng_state(),
        }
        save_distributed_training_state(str(tmp_path), state, rank=rank)
        loaded = load_distributed_training_state(str(tmp_path), rank=rank)
        assert loaded["cursor"] == state["cursor"]
        assert loaded["scheduler_state_dict"] == state["scheduler_state_dict"]
        torch.testing.assert_close(
            loaded["optimizer_state_dict"],
            state["optimizer_state_dict"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            loaded["fp32_master_params"], state["fp32_master_params"], rtol=0, atol=0
        )
        expected = torch.rand(4)
        restore_rng_state(loaded["rng_state"])
        torch.testing.assert_close(torch.rand(4), expected, rtol=0, atol=0)
    common = torch.load(tmp_path / "training_state.pt", weights_only=False)
    assert (
        not {"optimizer_state_dict", "fp32_master_params", "rng_state"} & common.keys()
    )
