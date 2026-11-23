from __future__ import annotations

METADATA_MEMBER = "_metadata.json"
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class HiddenStateError(RuntimeError):
    pass


def parse_layer_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layer IDs must be comma-separated integers"
        ) from exc
    if (
        not result
        or len(set(result)) != len(result)
        or any((item < 0 for item in result))
    ):
        raise argparse.ArgumentTypeError(
            "layer IDs must be unique non-negative integers"
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture target hidden states to raw tar"
    )
    model = parser.add_argument_group("model")
    model.add_argument("--target-model-path", required=True)
    model.add_argument("--target-model-backend", choices=("hf", "sglang"), default="hf")
    model.add_argument("--target-layer-ids", type=parse_layer_ids)
    model.add_argument("--trust-remote-code", action="store_true")
    model.add_argument("--model-download-dir")
    data = parser.add_argument_group("data")
    data.add_argument("--data-path", required=True)
    data.add_argument(
        "--output-path",
        help="Full-cache output directory. Defaults to ./cache/data/regen_full/<input>_<model>.",
    )
    data.add_argument("--chat-template", default="qwen")
    data.add_argument("--is-preformatted", action="store_true")
    data.add_argument("--max-length", type=int, default=4096)
    data.add_argument("--num-samples", type=int)
    data.add_argument("--build-dataset-num-proc", type=int, default=8)
    data.add_argument("--cache-dir", default="./cache/hidden_states")
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--tp-size", type=int, default=1)
    runtime.add_argument("--batch-size", type=int, default=1)
    runtime.add_argument("--dist-timeout", type=int, default=2000)
    runtime.add_argument("--max-retries", type=int, default=2)
    runtime.add_argument("--fail-on-error", action="store_true")
    runtime.add_argument("--resume", action="store_true")
    runtime.add_argument("--retry-failures", action="store_true")
    sglang = parser.add_argument_group("sglang backend")
    sglang.add_argument("--sglang-attention-backend", default="flashinfer")
    sglang.add_argument("--sglang-mem-fraction-static", type=float, default=0.4)
    sglang.add_argument("--sglang-context-length", type=int)
    sglang.add_argument("--sglang-enable-nccl-nvls", action="store_true")
    sglang.add_argument("--sglang-enable-symm-mem", action="store_true")
    sglang.add_argument("--sglang-enable-torch-compile", action="store_true")
    sglang.add_argument("--sglang-enable-dp-attention", action="store_true")
    sglang.add_argument("--sglang-enable-dp-lm-head", action="store_true")
    sglang.add_argument("--sglang-enable-piecewise-cuda-graph", action="store_true")
    sglang.add_argument(
        "--sglang-piecewise-cuda-graph-max-tokens", type=int, default=4096
    )
    sglang.add_argument("--sglang-piecewise-cuda-graph-tokens", type=int, nargs="+")
    sglang.add_argument("--sglang-ep-size", type=int, default=1)
    sglang.add_argument("--sglang-max-running-requests", type=int)
    sglang.add_argument("--sglang-max-total-tokens", type=int)
    args = parser.parse_args(argv)
    for name in ("max_length", "batch_size", "tp_size", "build_dataset_num_proc"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if args.retry_failures and (not args.resume):
        parser.error("--retry-failures requires --resume")
    return args


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def normalize_hidden_states(
    hidden_states: Any, layer_ids: list[int] | None
) -> dict[int, Any]:
    if isinstance(hidden_states, dict):
        result = {int(key): value for (key, value) in hidden_states.items()}
    elif isinstance(hidden_states, (tuple, list)):
        keys = layer_ids if layer_ids is not None else list(range(len(hidden_states)))
        if len(keys) != len(hidden_states):
            raise HiddenStateError("hidden-state count does not match target layer IDs")
        result = dict(zip(keys, hidden_states))
    else:
        raise HiddenStateError("target returned an unsupported hidden-state container")
    if not result:
        raise HiddenStateError("target returned no hidden states")
    return result


def collate_records(records: list[dict[str, Any]], pad_token_id: int):
    from torch.nn.utils.rnn import pad_sequence

    def values(key, padding):
        tensors = [item[key].reshape(-1) for item in records]
        return pad_sequence(tensors, batch_first=True, padding_value=padding)

    return {
        "input_ids": values("input_ids", pad_token_id),
        "attention_mask": values("attention_mask", 0),
        "loss_mask": values("loss_mask", 0),
    }


def contiguous_range(total: int, rank: int, size: int) -> tuple[int, int]:
    (base, remainder) = divmod(total, size)
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return (start, end)
