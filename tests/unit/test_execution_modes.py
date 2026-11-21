"""Profile selection, explicit overrides and fail-fast validation."""
from types import SimpleNamespace
import pytest
from litespec.training.execution import resolve_execution_options
from litespec.training.recipes.composable import parse_args


@pytest.mark.parametrize("mode,offline,disaggregate,expected", [
    ("legacy", True, False, (False, 0, False)),
    ("optimized", True, False, (True, 2, True)),
    ("optimized", False, False, (True, 0, False)),
    ("optimized", False, True, (True, 0, False)),
])
def test_profile(mode, offline, disaggregate, expected):
    args = SimpleNamespace(training_mode=mode,
        train_hidden_states_path="cache" if offline else None, disaggregate=disaggregate)
    resolve_execution_options(args)
    assert (args.hidden_cache_selective_read, args.cpu_batch_prefetch,
            args.async_schedule_planning) == expected


def command(mode="optimized", *extra):
    return parse_args(argv=["--target-model-path", "unused-model", "--output-dir", "unused-output",
        "--train-hidden-states-path", "unused-cache", "--training-mode", mode, "--report-to", "none", *extra])


def test_cli_profile_and_explicit_overrides():
    args = command()
    assert (args.cpu_batch_prefetch, args.async_schedule_planning,
            args.hidden_cache_selective_read, args.schedule_window_updates) == (2, True, True, 1)
    args = command("optimized", "--cpu-batch-prefetch", "0",
                   "--no-async-schedule-planning", "--no-hidden-cache-selective-read")
    assert (args.cpu_batch_prefetch, args.async_schedule_planning,
            args.hidden_cache_selective_read) == (0, False, False)
    assert command("legacy").cpu_batch_prefetch == 0


@pytest.mark.parametrize("extra", [
    ("--cpu-batch-prefetch", "-1"), ("--dataloader-prefetch-factor", "0"),
    ("--no-save-training-state",),
])
def test_invalid_cli_rejected(extra):
    with pytest.raises(SystemExit):
        command("optimized", *extra)


def test_legacy_cannot_silently_enable_planner():
    for extra in [("--async-schedule-planning",), ("--schedule-window-updates", "2")]:
        with pytest.raises(SystemExit):
            command("legacy", *extra)


@pytest.mark.parametrize("saved_mode,requested", [("legacy", "optimized"), ("composable_packed_v1", "legacy")])
def test_resume_cannot_cross_execution_contract(saved_mode, requested):
    from litespec.training.checkpoint import validate_training_identity
    state = dict(training_state_format="litespec_training_v2", execution_contract=saved_mode)
    with pytest.raises(ValueError, match="Changing training_mode requires init_from"):
        validate_training_identity(state, SimpleNamespace(training_mode=requested))
