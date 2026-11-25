"""Capture target hidden states into raw-tar cache shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from legospec.data.hidden_capture import CaptureMetrics, run_generation


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
    model.add_argument(
        "--target-model-backend", choices=("hf", "sglang"), default="sglang"
    )
    model.add_argument("--target-layer-ids", type=parse_layer_ids)
    model.add_argument("--trust-remote-code", action="store_true")
    model.add_argument("--model-download-dir")
    data = parser.add_argument_group("data")
    data.add_argument("--data-path", required=True)
    data.add_argument(
        "--output-path",
        help="Required output directory for the completed hidden cache.",
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
    parser.add_argument("--pipeline-depth", type=int, default=2)
    parser.add_argument("--writer-workers", type=int, default=2)
    parser.add_argument("--max-pending-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--max-batch-tokens", type=int, default=0)
    parser.add_argument(
        "--packed-requests", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--batch-order", choices=["original", "bucket"], default="original"
    )
    parser.add_argument(
        "--rank-assignment",
        choices=["contiguous", "token-balanced"],
        default=None,
        help="Explicit DP assignment; unset preserves the coupled bucket policy",
    )
    parser.add_argument("--tar-shard-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--tar-commit-workers", type=int, default=1)
    parser.add_argument("--tar-pending-commits", type=int, default=2)
    parser.add_argument("--tar-buffer-bytes", type=int, default=4 * 1024**2)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument(
        "--recompute-gemma4-final-norm",
        action="store_true",
        help="Store Gemma 4 Unified decoder states only in raw tar",
    )
    parser.add_argument(
        "--check-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check generated hidden values before writing; benchmarks may audit outside timing",
    )
    parser.add_argument(
        "--profiling-output",
        type=Path,
        help="Optional timing directory outside the hidden cache.",
    )
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
    if (
        min(
            args.pipeline_depth,
            args.writer_workers,
            args.max_pending_bytes,
            args.tar_buffer_bytes,
            args.tar_commit_workers,
            args.tar_pending_commits,
        )
        <= 0
    ):
        parser.error("pipeline and writer limits must be positive")
    if (
        args.packed_requests or args.max_batch_tokens
    ) and args.target_model_backend != "sglang":
        parser.error(
            "Packed capture requires SGLang; select --target-model-backend sglang or disable packed capture with --no-packed-requests and --max-batch-tokens 0"
        )
    if args.max_batch_tokens < 0:
        parser.error("token budget must be nonnegative")
    if args.tar_shard_bytes < 0:
        parser.error("Tar shard byte limit must be nonnegative")
    if not args.output_path:
        parser.error("--output-path is required for capture")
    if args.profiling_output is not None:
        if args.profiling_output.resolve().is_relative_to(
            Path(args.output_path).resolve()
        ):
            parser.error("--profiling-output must be outside --output-path")
    return args


def main(argv=None):
    args = parse_args(argv)
    metrics = CaptureMetrics(profiling=args.profiling_output is not None)
    run_generation(args, metrics)
    if args.profiling_output is not None:
        metrics.finish(args.profiling_output)


if __name__ == "__main__":
    main()
