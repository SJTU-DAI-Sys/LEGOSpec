"""Capture CLI and collators can be used without a distributed runtime."""

import pytest
import torch

from legospec.data.dataset import DataCollatorWithPadding
from scripts.data.capture_hidden_states import parse_args


def test_collator_uses_explicit_sequence_parallel_layout():
    features = [
        {
            "input_ids": torch.arange(length)[None],
            "attention_mask": torch.ones(1, length, dtype=torch.long),
            "loss_mask": torch.ones(1, length),
            "position_ids": torch.arange(2 * length)[None],
        }
        for length in (3, 5)
    ]
    batch = DataCollatorWithPadding(sp_degree=4, ulysses_degree=2)(features)
    assert batch["input_ids"].shape == (2, 8)
    assert batch["position_ids"].shape == (2, 16)
    assert batch["attention_mask"].sum(-1).tolist() == [3, 5]
    torch.testing.assert_close(batch["loss_mask"], batch["attention_mask"].float())


def test_capture_cli_combines_model_pipeline_and_profiling_options(tmp_path):
    base = [
        "--target-model-path",
        "target",
        "--data-path",
        "answers.jsonl",
        "--output-path",
        str(tmp_path / "cache"),
        "--target-layer-ids",
        "1,3",
        "--pipeline-depth",
        "3",
        "--profiling-output",
        str(tmp_path / "timings"),
    ]
    args = parse_args(base)
    assert args.target_model_backend == "sglang" and args.packed_requests
    assert args.target_layer_ids == [1, 3] and args.pipeline_depth == 3
    assert args.profiling_output == tmp_path / "timings"
    hf = parse_args(base + ["--target-model-backend", "hf", "--no-packed-requests"])
    assert hf.target_model_backend == "hf" and not hf.packed_requests
    with pytest.raises(SystemExit):
        parse_args(base + ["--target-model-backend", "hf"])
    with pytest.raises(SystemExit):
        parse_args(base + ["--profiling-output", str(tmp_path / "cache" / "timings")])
