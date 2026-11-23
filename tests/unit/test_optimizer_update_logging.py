"""CPU regressions for optimizer-step axes and complete accumulation metrics."""

import legospec.training.logger.logging as logger_logging

from types import SimpleNamespace

import pytest
import torch

from legospec.modeling.draft.objective.outputs import RatioTerms, ObjectiveOutput
from legospec.training.runtime.state import Phase, RunState, TrainingSession
from legospec.training.logger.logging import TrainingMetricAccumulator, WandbTracker
from legospec.training.runtime.runner import run_draft


def output(correct, total):
    loss = torch.tensor(1.0, requires_grad=True)
    return ObjectiveOutput(
        loss,
        RatioTerms(loss, torch.tensor(1.0)),
        {
            "hard_label/token_accuracy": RatioTerms(
                torch.tensor(float(correct)), torch.tensor(float(total))
            )
        },
    )


def test_accuracy_uses_all_accumulated_tokens_not_last_or_mean_of_percentages():
    metrics = TrainingMetricAccumulator()
    # 7/7 followed by 0/1: aggregate accuracy is 7/8, not 0 or 1/2.
    metrics.update(output(7, 7))
    metrics.update(output(0, 1))
    assert metrics.reduce()["hard_label/token_accuracy"].item() == 0.875
    metrics.reset()
    metrics.update(output(1, 4))
    assert metrics.reduce()["hard_label/token_accuracy"].item() == 0.25


def test_accuracy_reduces_counts_across_ranks_before_dividing(monkeypatch):

    metrics = TrainingMetricAccumulator()
    metrics.update(output(7, 7))
    metrics.update(output(0, 1))
    monkeypatch.setattr(logger_logging.dist, "is_available", lambda: True)
    monkeypatch.setattr(logger_logging.dist, "is_initialized", lambda: True)

    def remote_counts(pair, **unused):
        # The other rank contributes 1/4 token accuracy and 2/2 loss.
        pair.add_(
            torch.tensor([1.0, 4.0])
            if pair[1].item() == 8
            else torch.tensor([2.0, 2.0])
        )

    monkeypatch.setattr(logger_logging.dist, "all_reduce", remote_counts)
    assert metrics.reduce()["hard_label/token_accuracy"].item() == pytest.approx(8 / 12)


@pytest.mark.parametrize("resumed", [False, True])
def test_runner_logs_updates_and_entire_accumulation_including_tail(
    monkeypatch, resumed
):
    import legospec.training.optim.optimizer as optimizer_module

    monkeypatch.setattr(
        optimizer_module, "normalize_accumulated_gradients", lambda *a: None
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    flags, logged = [], []

    class Optimizer:
        use_muon = False

        def step(self):
            parameter.grad = None
            return 0.0

        def get_learning_rate(self):
            return 0.01

    class Adapter:
        algorithm, modes = "composable", ()

        def enter_phase(self, *unused):
            return {}

        def compute_step(self, session, data, *, collect_detailed_metrics, **unused):
            flags.append((data, collect_detailed_metrics))
            return ObjectiveOutput(
                parameter,
                RatioTerms(parameter, torch.tensor(1.0)),
                {
                    "hard_label/token_accuracy": RatioTerms(
                        torch.tensor(float(data % 3)), torch.tensor(3.0)
                    )
                },
            )

        def log_metrics(self, session, values):
            return values

        def detail_metrics(self, session):
            return {}

    from legospec.training.runtime import runner
    from legospec.training.execution import step as execution_step, windows
    from legospec.training.logger import logging

    adapter = Adapter()
    monkeypatch.setattr(runner, "enter_phase", adapter.enter_phase)
    monkeypatch.setattr(execution_step, "compute_step", adapter.compute_step)
    monkeypatch.setattr(logging, "log_metrics", adapter.log_metrics)
    monkeypatch.setattr(logging, "detail_metrics", adapter.detail_metrics)
    monkeypatch.setattr(windows, "prepare_optimizer_window", lambda *a: None)

    class Tracker:
        def log(self, payload, *, step):
            logged.append((step, payload))

    state = (
        RunState(phase_step=8, global_step=8, optimizer_step=1, next_batch=8)
        if resumed
        else RunState()
    )
    session = TrainingSession(
        args=SimpleNamespace(
            accumulation_steps=8,
            log_interval=3,
            save_interval=15000,
            save_every_epoch=False,
            final_save=False,
        ),
        phases=[Phase("train", 0, 1)],
        state=state,
        optimizer=Optimizer(),
        tracker=Tracker(),
        steps_per_epoch=26,
        total_steps=4,
    )
    source = lambda s, phase: iter(
        range(s.state.phase_step, phase.batch_count(s.steps_per_epoch))
    )
    run_draft(session, batch_source=source, save=lambda *a: None)
    assert [step for step, payload in logged if "train/loss" in payload] == (
        [3, 4] if resumed else [1, 3, 4]
    )
    for step, payload in logged:
        if "train/loss" not in payload:
            continue
        start, stop = (step - 1) * 8, min(step * 8, 26)
        assert payload["train/optimizer_step"] == step
        assert payload["detail/composable/optimizer_step"] == step
        assert payload["detail/composable/micro_step"] == stop
        assert payload["train/token_accuracy"] == pytest.approx(
            sum(i % 3 for i in range(start, stop)) / ((stop - start) * 3)
        )
    assert all(flag == (data < 8 or data >= 16) for data, flag in flags)


def test_wandb_binds_train_and_detail_axes_to_explicit_optimizer_step(monkeypatch):
    import legospec.training.logger.logging as module

    definitions, calls = [], []
    fake = SimpleNamespace(
        init=lambda **kw: None,
        define_metric=lambda *a, **kw: definitions.append((a, kw)),
        log=lambda payload, **kw: calls.append((payload, kw)),
    )
    monkeypatch.setattr(module, "wandb", fake)
    monkeypatch.setattr(logger_logging.dist, "get_rank", lambda: 0)
    monkeypatch.setenv("WANDB_MODE", "offline")
    tracker = WandbTracker(
        SimpleNamespace(wandb_project="test", wandb_name="test", wandb_run_id="test"),
        ".",
    )
    payload = {"train/token_accuracy": 0.875, "detail/composable/micro_step": 400}
    tracker.log(payload, step=50)
    assert payload == {
        "train/token_accuracy": 0.875,
        "detail/composable/micro_step": 400,
    }
    assert calls == [
        ({**payload, "train/optimizer_step": 50}, {"step": 50, "commit": True})
    ]
    for prefix in ("train/*", "detail/*"):
        assert ((prefix,), {"step_metric": "train/optimizer_step"}) in definitions
