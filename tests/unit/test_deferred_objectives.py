"""Deferred objectives cannot enter active model/training recipes."""

import json
from pathlib import Path

import pytest

from legospec.modeling.draft import LegoSpecConfig
from legospec.modeling.draft.loss_config import normalize_objective


@pytest.mark.parametrize("kind", ["dpal_plus_plus", "tv_survival", "opd_survival"])
def test_deferred_objective_rejected_at_configuration_boundary(kind):
    with pytest.raises(ValueError, match="Unsupported composable objective"):
        normalize_objective({"type": kind})
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/composable_parallel_dpal_renyi.json").read_text()
    )
    config["composable_config"]["objective"] = {"type": kind}
    with pytest.raises(ValueError):
        LegoSpecConfig(**config)


@pytest.mark.parametrize(
    "arguments",
    [["--opd-data-mode", "sampled_anchor"], ["--opd-target-chunk-anchors", "0"]],
)
def test_deferred_training_options_are_not_cli_arguments(arguments, capsys):
    from legospec.training.config import parse_args

    with pytest.raises(SystemExit) as error:
        parse_args(
            argv=[
                "--target-model-path",
                "target",
                "--train-data-path",
                "data.jsonl",
                "--output-dir",
                "out",
                "--report-to",
                "none",
                *arguments,
            ]
        )
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_removed_training_mode_cannot_silently_fall_back():
    from legospec.training.execution.step import TrainingStep

    with pytest.raises(TypeError, match="Unexpected TrainingStep arguments"):
        TrainingStep(None, None, None, 0, opd_data_mode="sampled_anchor")
