"""Bounded asynchronous hidden capture and capture metrics."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import queue
import resource
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from legospec.data.hidden_cache import atomic_write_json, persist_cache, sha256_file

DEFAULT_LAYER_IDS = [1, 9, 17, 25, 33]


class Slot:
    def __init__(self):
        self.layers = {}
        self.valid = torch.empty((), dtype=torch.bool, pin_memory=True)

    def preallocate(
        self, tokens, hidden_size, layer_ids, bulk, separate_final, recompute_final
    ):
        count = tokens * hidden_size
        num_layers = len(layer_ids) + int(separate_final)
        self.flat = torch.empty(
            count * num_layers, dtype=torch.bfloat16, pin_memory=True
        )
        if bulk and recompute_final:
            self.layers = {"aux": self.flat}
        elif bulk:
            boundary = count * (num_layers - 1)
            self.layers = {"aux": self.flat[:boundary], "last": self.flat[boundary:]}
        else:
            self.layers = {
                layer: self.flat[i * count : (i + 1) * count]
                for (i, layer) in enumerate(layer_ids)
            }
            if separate_final:
                self.layers["teacher_final"] = self.flat[-count:]

    @property
    def allocated_bytes(self):
        return sum((t.numel() * t.element_size() for t in self.layers.values()))

    def allocate(self, hidden, limit):
        required = sum((t.numel() * t.element_size() for t in hidden.values()))
        if required > limit:
            raise MemoryError(f"batch needs {required} pinned bytes; budget {limit}")
        for key, tensor in hidden.items():
            existing = self.layers.get(key)
            if (
                existing is None
                or existing.numel() < tensor.numel()
                or existing.dtype != tensor.dtype
            ):
                self.layers.pop(key, None)
                self.layers[key] = torch.empty(
                    tensor.numel(), dtype=tensor.dtype, pin_memory=True
                )
        return {
            key: self.layers[key][: value.numel()].view(value.shape)
            for (key, value) in hidden.items()
        }


def metadata(record):
    active = record["loss_mask"].reshape(-1) > 0.5
    padded = torch.cat(
        (
            torch.zeros(1, dtype=torch.int8),
            active.to(torch.int8),
            torch.zeros(1, dtype=torch.int8),
        )
    )
    changes = padded[1:] - padded[:-1]
    starts = torch.nonzero(changes == 1).reshape(-1).tolist()
    ends = torch.nonzero(changes == -1).reshape(-1).tolist()
    return {
        "sequence_length": record["input_ids"].numel(),
        "supervised_token_count": int(active.sum()),
        "loss_mask_runs": [[int(a), int(b)] for (a, b) in zip(starts, ends)],
    }


def check_capture_space(root, processed, committed, layer_count, hidden_size):
    from pyarrow import compute as pc

    lengths = pc.list_value_length(pc.list_flatten(processed.data.column("input_ids")))
    if len(lengths) != len(processed):
        raise ValueError("Cannot count one token sequence per cache row")
    pending = [
        length
        for (index, length) in enumerate(lengths.to_pylist())
        if index not in committed
    ]
    required = sum(pending) * (2 * hidden_size * layer_count + 3 * 8) + 4096 * len(
        pending
    )
    free = shutil.disk_usage(root).free
    reserve = 64 * 1024**3
    print(
        f"Pending capture cache: {len(pending)} samples, {required} bytes, {free} bytes free",
        flush=True,
    )
    if pending and required + reserve > free:
        raise OSError(
            f"Capture needs {required + reserve} bytes including disk reserve, but only {free} bytes are free"
        )


class Pipeline:
    def __init__(self, args, sink, metrics):
        (self.args, self.sink, self.metrics) = (args, sink, metrics)
        self.stream = torch.cuda.Stream()
        self.free = queue.Queue()
        self.slots = [Slot() for _ in range(args.pipeline_depth)]
        budget = args.max_batch_tokens or args.batch_size * args.max_length
        tokens = min(budget, args.batch_size * args.max_length)
        hidden_size = args.capture_hidden_size
        size = (
            tokens
            * hidden_size
            * (len(args.capture_layer_ids) + int(args.separate_final))
            * 2
        )
        rounded = 1 << (size - 1).bit_length()
        reserved = torch.cuda.memory.host_memory_stats().get(
            "reserved_bytes.current", 0
        )
        if rounded * args.pipeline_depth + reserved > args.max_pending_bytes:
            raise MemoryError("Rounded pinned pool allocation exceeds byte quota")
        for slot in self.slots:
            slot.preallocate(
                tokens,
                hidden_size,
                args.capture_layer_ids,
                bool(args.packed_requests or args.max_batch_tokens),
                args.separate_final,
                args.recompute_gemma4_final_norm,
            )
        for slot in self.slots:
            self.free.put(slot)
        self.work = queue.Queue(maxsize=args.pipeline_depth)
        self.error = None
        self.pool = ThreadPoolExecutor(max_workers=args.writer_workers)
        self.thread = threading.Thread(
            target=self.consume, name="hidden-writer", daemon=True
        )
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError("Background hidden writer failed") from self.error

    def submit(self, output, records):
        self.check()
        with self.metrics.time("queue_wait"):
            while True:
                self.check()
                try:
                    slot = self.free.get(timeout=0.1)
                    break
                except queue.Empty:
                    pass
        hidden = normalize_hidden_states(output.hidden_states, None)
        if sorted(hidden) != self.args.capture_layer_ids:
            raise HiddenStateError(f"unexpected capture layers: {sorted(hidden)}")
        recompute_final = self.args.recompute_gemma4_final_norm
        normalized_final = None if recompute_final else output.final_hidden_state
        if normalized_final is not None:
            hidden = dict(hidden, teacher_final=normalized_final)
        bulk = bool(getattr(output, "packed", False))
        transport = {"aux": output.packed_aux} if bulk else hidden
        if bulk and (not recompute_final):
            transport["last"] = output.packed_last
        try:
            buffers = slot.allocate(
                transport, self.args.max_pending_bytes // self.args.pipeline_depth
            )
            if bulk:
                aux_ids = [layer for layer in hidden if layer != "teacher_final"]
                viewed = buffers["aux"].view(
                    buffers["aux"].shape[0], len(aux_ids), self.args.capture_hidden_size
                )
                cpu = {layer: viewed[:, i, :] for (i, layer) in enumerate(aux_ids)}
                if not recompute_final:
                    cpu["teacher_final"] = buffers["last"]
            else:
                cpu = buffers
            allocated = sum((s.allocated_bytes for s in self.slots))
            self.metrics.peak_pinned_bytes = max(
                getattr(self.metrics, "peak_pinned_bytes", 0), allocated
            )
        except BaseException:
            self.free.put(slot)
            raise
        with self.metrics.gpu("finite_gpu"):
            good = torch.ones((), device="cuda", dtype=torch.bool)
            if getattr(self.args, "check_finite", True):
                if getattr(output, "packed", False):
                    for value in transport.values():
                        good.logical_and_(torch.isfinite(value).all())
                else:
                    valid = output.attention_mask.bool().unsqueeze(-1)
                    for value in hidden.values():
                        good.logical_and_((torch.isfinite(value) | ~valid).all())
        produced = torch.cuda.Event()
        produced.record()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(produced)
            with self.metrics.gpu("d2h_async", self.stream):
                for key, value in transport.items():
                    buffers[key].copy_(value.detach(), non_blocking=True)
                    value.record_stream(self.stream)
                slot.valid.copy_(good, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        offsets = output.token_offsets if getattr(output, "packed", False) else None
        packet = (slot, cpu, ready, records, hidden, good, offsets)
        self.work.put(packet)

    def serialize(self, cpu, record, index, offsets=None):
        length = record["input_ids"].numel()
        with self.metrics.time("serialize"):
            compact = False
            hidden = {
                key: value[index, :length]
                if offsets is None
                else value[offsets[index] : offsets[index + 1]]
                for (key, value) in cpu.items()
            }
            if compact:
                hidden = {key: value.clone() for (key, value) in hidden.items()}
            payload = {
                "schema_version": 2,
                "sample_index": record["source_index"],
                "input_ids": record["input_ids"].reshape(-1).clone(),
                "attention_mask": record["attention_mask"].reshape(-1).clone(),
                "loss_mask": record["loss_mask"].reshape(-1).clone(),
                "hidden_states": hidden,
            }
            if not self.args.recompute_gemma4_final_norm:
                payload["final_hidden_state"] = hidden.pop("teacher_final")
            from legospec.data.hidden_cache import RawSample

            value = RawSample(payload)
            return (record["source_index"], value, metadata(record))

    def consume(self):
        while True:
            packet = self.work.get()
            if packet is None:
                self.work.task_done()
                break
            (slot, cpu, ready, records, gpu_refs, good, offsets) = packet
            futures = []
            try:
                with self.metrics.time("d2h_wait"):
                    ready.synchronize()
                if self.error is not None:
                    continue
                if not bool(slot.valid):
                    raise HiddenStateError("captured NaN or Inf")
                futures = [
                    self.pool.submit(self.serialize, cpu, record, i, offsets)
                    for (i, record) in enumerate(records)
                ]
                for future in futures:
                    (index, value, meta) = future.result()
                    self.sink.add(index, value, meta)
            except BaseException as error:
                self.error = error
            finally:
                for future in futures:
                    try:
                        future.result()
                    except BaseException:
                        pass
                self.free.put(slot)
                self.work.task_done()

    def finish(self):
        self.work.join()
        self.work.put(None)
        self.thread.join()
        self.pool.shutdown()
        self.check()
        self.sink.finish()


def plan_capture_indices(
    lengths, world, rank, *, assignment="contiguous", batch_order="original"
):
    """Static capture assignment and batch ordering, with independent controls."""
    if world < 1 or not 0 <= rank < world:
        raise ValueError("Invalid capture rank topology")
    if assignment == "contiguous":
        (start, end) = contiguous_range(len(lengths), rank, world)
        indices = list(range(start, end))
    elif assignment == "token-balanced":
        bins = [[] for _ in range(world)]
        loads = [0] * world
        for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
            owner = min(range(world), key=lambda r: (loads[r], r))
            bins[owner].append(index)
            loads[owner] += lengths[index]
        indices = sorted(bins[rank])
    else:
        raise ValueError("Unknown capture assignment")
    if batch_order == "bucket":
        indices.sort(key=lambda i: (lengths[i], i))
    elif batch_order != "original":
        raise ValueError("Unknown capture batch order")
    return indices


def run_generation(args, metrics):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from legospec.args import SGLangBackendArgs
    from legospec.data import build_training_dataset
    from legospec.distributed import destroy_distributed, get_dp_group, init_distributed
    from legospec.modeling.config_utils import load_text_model_config
    from legospec.modeling.target import get_training_target_model
    from legospec.modeling.target.adapters.base import get_model_adapter

    text_config = load_text_model_config(
        args.target_model_path,
        cache_dir=args.model_download_dir,
        trust_remote_code=args.trust_remote_code,
    )
    adapter = get_model_adapter(text_config)
    if args.recompute_gemma4_final_norm and not adapter.supports_final_norm_recompute:
        raise ValueError(
            "The selected target adapter does not support final norm recomputation"
        )
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    rank = dist.get_rank(get_dp_group())
    world = dist.get_world_size(get_dp_group())
    root = Path(args.output_path)
    root.mkdir(parents=True, exist_ok=True)
    cpu_group = dist.new_group(backend="gloo")
    identity = input_fingerprint(Path(args.data_path))
    try:
        args.final_norm_layer_id = text_config.num_hidden_layers - 1
        requested_layer_ids = args.target_layer_ids or DEFAULT_LAYER_IDS
        if max(requested_layer_ids) > args.final_norm_layer_id:
            raise ValueError(
                f"Layer IDs must be between 0 and {args.final_norm_layer_id}"
            )
        args.capture_layer_ids = sorted(
            set(requested_layer_ids) | {args.final_norm_layer_id}
        )
        args.capture_hidden_size = text_config.hidden_size
        args.separate_final = not args.recompute_gemma4_final_norm
        contract = "decoder_outputs_pre_final_norm_v1"
        header = {
            "input_identity": identity,
            "model": args.target_model_path,
            "layers": args.capture_layer_ids,
            "max_length": args.max_length,
            "format": "tar-raw-v1",
            "world_size": world,
        }
        if args.separate_final or args.recompute_gemma4_final_norm:
            header["context_feature_contract"] = contract
        if args.recompute_gemma4_final_norm:
            header["recompute_gemma4_final_norm"] = True
        header_path = root / "capture_identity.json"
        if rank == 0:
            if header_path.exists():
                assert args.resume and json.loads(header_path.read_text()) == header, (
                    "Resume identity mismatch"
                )
            elif any(root.iterdir()):
                raise ValueError("Nonempty output directory")
            else:
                atomic_write_json(header_path, header)
        dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        kwargs = (
            SGLangBackendArgs.from_args(args).to_kwargs()
            if args.target_model_backend == "sglang"
            else {}
        )
        if args.target_model_backend == "sglang":
            kwargs["context_length"] = (
                kwargs.get("context_length") or args.max_length + 1
            )
            kwargs["max_running_requests"] = (
                kwargs.get("max_running_requests") or args.batch_size
            )
            kwargs["max_total_tokens"] = (
                kwargs.get("max_total_tokens") or args.batch_size * args.max_length
            )
        target = get_training_target_model(
            args.target_model_path,
            backend=args.target_model_backend,
            capture_layer_ids=args.capture_layer_ids,
            torch_dtype=torch.bfloat16,
            device="cuda" if args.target_model_backend == "hf" else None,
            cache_dir=args.model_download_dir,
            trust_remote_code=args.trust_remote_code,
            **kwargs,
        )
        raw = load_dataset("json", data_files=args.data_path)["train"]
        if args.num_samples is not None:
            raw = raw.select(range(min(len(raw), args.num_samples)))
        key = hashlib.sha256(
            json.dumps(
                {
                    "identity": identity,
                    "max_length": args.max_length,
                    "chat_template": args.chat_template,
                    "model": args.target_model_path,
                    "preformatted": args.is_preformatted,
                    "num_samples": args.num_samples,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        def build():
            return build_training_dataset(
                raw,
                tokenizer,
                args.chat_template,
                max_length=args.max_length,
                shuffle_seed=None,
                num_proc=args.build_dataset_num_proc,
                cache_dir=str(Path(args.cache_dir) / "processed_dataset"),
                cache_key=key,
                is_preformatted=args.is_preformatted,
                capture_errors=True,
            )

        processed = build() if dist.get_rank() == 0 else None
        dist.barrier()
        if processed is None:
            processed = build()
        (start, end) = contiguous_range(len(processed), rank, world)
        from legospec.data.hidden_cache import StreamTarSink

        sink = StreamTarSink(args, root, identity, rank, metrics)
        if args.recompute_gemma4_final_norm:
            check_capture_space(
                root,
                processed,
                sink.done,
                len(args.capture_layer_ids),
                args.capture_hidden_size,
            )
        pipeline = Pipeline(args, sink, metrics)
        pending = []
        captured = 0

        def capture(records):
            nonlocal captured
            if not records:
                return
            if args.packed_requests or args.max_batch_tokens:
                total = sum((record["input_ids"].numel() for record in records))
                metrics.batch_counts(total, total)
                with metrics.time("target_host_total"):
                    output = target.generate_capture_data(records)
            else:
                batch = collate_records(records, tokenizer.pad_token_id)
                with metrics.time("h2d_host"), metrics.gpu("h2d"):
                    gpu = {
                        key: value.cuda(non_blocking=True)
                        for (key, value) in batch.items()
                    }
                with metrics.time("target_host_total"):
                    output = target.generate_training_data(**gpu, return_logits=False)
            pipeline.submit(output, records)
            captured += len(records)
            if captured % 64 < len(records):
                print(
                    f"[rank {rank}] captured {captured}/{len(indices)}; queued={pipeline.work.qsize()}",
                    flush=True,
                )
            if args.clear_cache:
                torch.cuda.empty_cache()

        indices = list(range(start, end))
        assignment = args.rank_assignment or (
            "token-balanced" if args.batch_order == "bucket" else "contiguous"
        )
        if args.batch_order == "bucket" or assignment == "token-balanced":
            with metrics.time("batch_plan"):
                lengths = [
                    processed[i]["input_ids"].numel() for i in range(len(processed))
                ]
                indices = plan_capture_indices(
                    lengths,
                    world,
                    rank,
                    assignment=assignment,
                    batch_order=args.batch_order,
                )
        pending_tokens = 0
        for index in indices:
            if index in sink.done:
                continue
            with metrics.time("dataset_read"):
                row = processed[index]
            if row.get("preprocessing_error"):
                raise HiddenStateError(row["preprocessing_error"])
            record = {
                key: row[key].reshape(-1)
                for key in ("input_ids", "attention_mask", "loss_mask")
            }
            record["source_index"] = index
            length = record["input_ids"].numel()
            if (
                args.max_batch_tokens
                and pending
                and (pending_tokens + length > args.max_batch_tokens)
            ):
                capture(pending)
                pending = []
                pending_tokens = 0
            pending.append(record)
            pending_tokens += length
            if len(pending) >= args.batch_size:
                capture(pending)
                pending = []
                pending_tokens = 0
        capture(pending)
        with metrics.time("writer_drain"):
            pipeline.finish()
        metrics.capture_end = time.perf_counter()
        print(f"[rank {rank}] capture drained; waiting for finalization", flush=True)
        dist.barrier(group=cpu_group)
        run_metadata = {
            "save_mode": "regen_full",
            "sample_index_version": 1,
            "target_model": args.target_model_path,
            "target_backend": args.target_model_backend,
            "chat_template": args.chat_template,
            "max_length": args.max_length,
            "requested_hidden_layer_ids": requested_layer_ids,
            "stored_hidden_layer_ids": args.capture_layer_ids,
            "final_norm_layer_id": args.final_norm_layer_id,
            "num_hidden_layers": text_config.num_hidden_layers,
            "hidden_size": text_config.hidden_size,
            "hidden_dtype": "torch.bfloat16",
            "compressed": False,
        }
        run_metadata["context_feature_contract"] = contract
        if args.recompute_gemma4_final_norm:
            run_metadata["recompute_gemma4_final_norm"] = True
        if rank == 0:
            print("[rank 0] finalization started", flush=True)
            from legospec.data.hidden_cache import finalize

            with metrics.time("finalize"):
                finalize(root, identity, run_metadata, len(processed), metrics)

            persist_cache(root, metrics)
            print("[rank 0] manifest and data persisted", flush=True)
        dist.barrier(group=cpu_group)
    finally:
        destroy_distributed()


BOOT = time.perf_counter()


class CaptureMetrics:
    def __init__(self, profiling=False):
        self.profiling = profiling
        self.cpu = defaultdict(float)
        self.counts = defaultdict(int)
        self.events = []
        self.lock = threading.Lock()
        self.first_batch = None
        self.last_batch = None
        self.capture_end = None
        self.tokens = 0
        self.padded_tokens = 0
        self.peak_pinned_bytes = 0
        self.raw_cpu = []

    @contextmanager
    def time(self, key):
        start = time.perf_counter()
        trace = os.environ.get("OPT_NVTX") == "1"
        if trace:
            import torch

            torch.cuda.nvtx.range_push(key)
        try:
            yield
        finally:
            if trace:
                torch.cuda.nvtx.range_pop()
            with self.lock:
                elapsed = time.perf_counter() - start
                self.cpu[key] += elapsed
                self.counts[key] += 1
                if self.profiling:
                    self.raw_cpu.append((key, start, elapsed, threading.get_ident()))

    def wrap(self, obj, name, key):
        original = getattr(obj, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.time(key):
                return original(*args, **kwargs)

        setattr(obj, name, wrapped)

    @contextmanager
    def gpu(self, key, stream=None):
        if not self.profiling:
            yield
            return
        import torch

        enqueued = time.perf_counter()
        first, last = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        first.record(stream)
        try:
            yield
        finally:
            last.record(stream)
            with self.lock:
                self.events.append((key, first, last, enqueued))

    def batch(self, values):
        self.batch_counts(
            int(values["attention_mask"].sum()), values["input_ids"].numel()
        )

    def batch_counts(self, valid, padded):
        now = time.perf_counter()
        if self.first_batch is None:
            self.first_batch = now
        self.last_batch = now
        self.tokens += valid
        self.padded_tokens += padded
        self.counts["batches"] += 1
        if os.environ.get("OPT_PROFILE") == "1":
            import torch

            if self.counts["batches"] == 2:
                torch.cuda.cudart().cudaProfilerStart()
            if self.counts["batches"] == 10:
                torch.cuda.cudart().cudaProfilerStop()

    def finish(self, directory, extra=None):
        import torch

        gpu = defaultdict(float)
        raw_gpu = []
        for key, first, last, enqueued in self.events:
            last.synchronize()
            elapsed = first.elapsed_time(last) / 1000
            gpu[key] += elapsed
            raw_gpu.append((key, enqueued, elapsed))
        value = {
            "rank": int(os.environ.get("RANK", 0)),
            "pid": os.getpid(),
            "wall_s": time.perf_counter() - BOOT,
            "cpu_stage_s": dict(self.cpu),
            "stage_calls": dict(self.counts),
            "cuda_stage_s": dict(gpu),
            "valid_tokens": self.tokens,
            "padded_tokens": self.padded_tokens,
            "first_batch_offset_s": None
            if self.first_batch is None
            else self.first_batch - BOOT,
            "first_batch_monotonic": self.first_batch,
            "capture_end_monotonic": self.capture_end or self.last_batch,
            "last_batch_offset_s": None
            if self.last_batch is None
            else self.last_batch - BOOT,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            **(extra or {}),
        }
        value["peak_pinned_bytes"] = self.peak_pinned_bytes
        value["host_allocator"] = torch.cuda.memory.host_memory_stats()
        value["peak_process_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        value["runtime_backend"] = getattr(self, "runtime_backend", {})
        directory.mkdir(parents=True, exist_ok=True)
        raw_path = directory / f"rank_{value['rank']}_raw_timings.json"
        raw_path.write_text(
            json.dumps(
                {
                    "cpu_columns": [
                        "stage",
                        "monotonic_start_s",
                        "elapsed_s",
                        "thread_id",
                    ],
                    "cuda_columns": [
                        "stage",
                        "monotonic_enqueue_s",
                        "cuda_event_elapsed_s",
                    ],
                    "cpu": self.raw_cpu,
                    "cuda": raw_gpu,
                }
            )
            + "\n"
        )
        value["raw_timings"] = str(raw_path)
        (directory / f"rank_{value['rank']}_metrics.json").write_text(
            json.dumps(value, indent=2) + "\n"
        )


class HiddenStateError(RuntimeError):
    pass


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
