"""Public YAML launchers parse through the one shared CLI."""

from pathlib import Path
import pytest
from scripts.config.training_launcher import (
    load_yaml_config,
    arguments_to_cli,
    launcher_environment,
    training_base_cli,
)
from legospec.training.config import parse_args

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "path", sorted((ROOT / "scripts/config").glob("*.yaml")), ids=lambda p: p.stem
)
def test_launcher_checkpoint_policy_and_arguments(path):
    launcher, arguments, snapshot = load_yaml_config(
        path, identity_key="algorithm", identity_value="composable"
    )
    assert arguments["save_interval"] == 15000
    assert arguments["save_strategy"] == "interval"
    assert arguments["save_training_state"] is True
    environment = launcher_environment(launcher, yaml_snapshot=snapshot)
    parsed = parse_args(
        argv=training_base_cli(environment) + arguments_to_cli(arguments)
    )
    assert parsed.save_training_state and parsed.save_interval == 15000
