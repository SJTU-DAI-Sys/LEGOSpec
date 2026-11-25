"""Hidden-cache format, manifests, streaming writes and durable commits."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tarfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

MAGIC = b"LGRAW001"
DTYPES = {
    str(t): t
    for t in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    )
}


class RawSample:
    def __init__(self, payload):
        self.parts = []
        descriptors = []
        offset = 0
        fields = [
            (key, payload[key]) for key in ("input_ids", "attention_mask", "loss_mask")
        ]
        fields += [
            (f"hidden/{int(key)}", value)
            for key, value in sorted(payload["hidden_states"].items())
        ]
        if "final_hidden_state" in payload:
            fields += [("final_hidden_state", payload["final_hidden_state"])]
        for name, tensor in fields:
            if tensor.device.type != "cpu" or str(tensor.dtype) not in DTYPES:
                raise ValueError("Raw tar requires a supported CPU tensor")
            tensor = tensor.detach().contiguous()
            value = memoryview(tensor.view(torch.uint8).numpy()).cast("B")
            descriptors.append(
                {
                    "name": name,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "offset": offset,
                    "nbytes": len(value),
                }
            )
            self.parts.append(value)
            offset += len(value)
        header = json.dumps(
            {
                "version": 1,
                "sample_index": payload["sample_index"],
                "tensors": descriptors,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.header = MAGIC + struct.pack("<Q", len(header)) + header
        self.header += b"\0" * (-len(self.header) % 64)
        self.size = len(self.header) + offset


def load_raw(
    handle,
    start,
    size,
    *,
    headers_only=False,
    expect_final_hidden=True,
    layer_ids=None,
    max_length=None,
):
    """Validate every descriptor, optionally read selected layers/token prefixes."""
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be positive")
    layer_ids = None if layer_ids is None else set(layer_ids)
    handle.seek(start)
    prefix = handle.read(16)
    if len(prefix) != 16 or prefix[:8] != MAGIC:
        raise ValueError("Invalid raw tensor member magic")
    length = struct.unpack("<Q", prefix[8:])[0]
    if not 0 < length <= min(1024 * 1024, size - 16):
        raise ValueError("Invalid raw tensor header length")
    info = json.loads(handle.read(length))
    if info.get("version") != 1:
        raise ValueError("Unsupported raw tensor member version")
    data_start = (16 + length + 63) // 64 * 64
    output = {
        "schema_version": 2,
        "sample_index": info["sample_index"],
        "hidden_states": {},
    }
    names = set()
    previous_end = 0
    for desc in info["tensors"]:
        name, dtype, shape = desc["name"], DTYPES[desc["dtype"]], desc["shape"]
        if name in names or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("Invalid raw tensor shape or duplicate field")
        names.add(name)
        offset, nbytes = desc["offset"], desc["nbytes"]
        expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if (
            type(offset) is not int
            or type(nbytes) is not int
            or offset != previous_end
            or nbytes != expected
            or data_start + offset + nbytes > size
        ):
            raise ValueError("Invalid raw tensor byte range")
        previous_end = offset + nbytes
        layer = None
        if name.startswith("hidden/"):
            layer = name.removeprefix("hidden/")
            if not layer.isascii() or not layer.isdecimal() or str(int(layer)) != layer:
                raise ValueError("Invalid raw hidden layer ID")
            layer = int(layer)
        elif name not in (
            "input_ids",
            "attention_mask",
            "loss_mask",
            "final_hidden_state",
        ):
            raise ValueError("Unknown raw tensor field")
        if layer is not None and layer_ids is not None and layer not in layer_ids:
            continue
        read_shape = list(shape)
        if max_length is not None:
            read_shape[0] = min(read_shape[0], max_length)
        read_bytes = nbytes // shape[0] * read_shape[0]
        if headers_only and (layer is not None or name == "final_hidden_state"):
            value = torch.empty(read_shape, dtype=dtype, device="meta")
        else:
            handle.seek(start + data_start + offset)
            data = bytearray(read_bytes)
            if handle.readinto(data) != read_bytes:
                raise ValueError("Truncated raw tensor")
            value = torch.frombuffer(data, dtype=dtype).reshape(read_shape)
        if layer is not None:
            output["hidden_states"][layer] = value
        elif name in ("input_ids", "attention_mask", "loss_mask", "final_hidden_state"):
            output[name] = value
        else:
            raise ValueError("Unknown raw tensor field")
    required = {"input_ids", "attention_mask", "loss_mask"}
    if (
        not required.issubset(names)
        or ("final_hidden_state" in names) != expect_final_hidden
    ):
        raise ValueError("Raw tensor fields do not match the cache layout")
    if data_start + previous_end != size:
        raise ValueError("Trailing raw tensor bytes")
    return output


def sync_dir(path, metrics):
    with metrics.time("commit_directory_fsync"):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def durable_json(path, value, metrics):
    temporary = path.with_name("." + path.name + ".tmp")
    with metrics.time("commit_index_encode"):
        encoded = json.dumps(value, sort_keys=True).encode() + b"\n"
    with metrics.time("commit_index_write"):
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            with metrics.time("commit_index_fsync"):
                os.fsync(handle.fileno())
    with metrics.time("commit_index_rename"):
        os.replace(temporary, path)
    sync_dir(path.parent, metrics)


class StreamTar:
    def __init__(self, root, rank, sequence, args, metrics):
        (self.metrics, self.args) = (metrics, args)
        self.final = root / "shards" / f"rank_{rank:05d}_part_{sequence:08d}.tar"
        self.temp = self.final.with_name("." + self.final.name + ".tmp")
        self.handle = self.temp.open("wb", buffering=args.tar_buffer_bytes)
        self.digest = hashlib.sha256()
        self.offset = 0
        (self.indices, self.metadata, self.names, self.ranges) = ([], [], [], [])

    def write(self, value):
        value = memoryview(value).cast("B")
        for begin in range(0, len(value), self.args.tar_buffer_bytes):
            part = value[begin : begin + self.args.tar_buffer_bytes]
            with self.metrics.time("tar_write_bytes"):
                count = self.handle.write(part)
            if count != len(part):
                raise OSError("Short tar write")
            with self.metrics.time("sha256_stream"):
                self.digest.update(part)
            self.offset += count

    def member(self, name, value):
        size = value.size if isinstance(value, RawSample) else len(value)
        header = tarfile.TarInfo(name)
        (header.size, header.mtime) = (size, 0)
        self.write(header.tobuf(format=tarfile.USTAR_FORMAT))
        start = self.offset
        if isinstance(value, RawSample):
            self.write(value.header)
            for part in value.parts:
                self.write(part)
        else:
            self.write(value)
        self.write(b"\x00" * (-size % 512))
        return [start, size]

    def seal(self, meta):
        with self.metrics.time("commit_tar_footer"):
            self.member(METADATA_MEMBER, json.dumps(meta, sort_keys=True).encode())
            self.write(b"\x00" * 1024)
            self.write(b"\x00" * (-self.offset % 10240))

    def publish(self, meta, sidecar):
        try:
            with self.metrics.time("shard_commit"):
                with self.metrics.time("commit_flush"):
                    self.handle.flush()
                with self.metrics.time("commit_file_fsync"):
                    os.fsync(self.handle.fileno())
                with self.metrics.time("commit_close"):
                    self.handle.close()
                meta["sha256"] = self.digest.hexdigest()
                with self.metrics.time("commit_data_rename"):
                    os.replace(self.temp, self.final)
                sync_dir(self.final.parent, self.metrics)
                durable_json(sidecar, meta, self.metrics)
            return meta
        finally:
            self.handle.close()


class StreamTarSink:
    def __init__(self, args, root, identity, rank, metrics):
        (self.args, self.root, self.identity, self.rank, self.metrics) = (
            args,
            Path(root),
            identity,
            rank,
            metrics,
        )
        self.directory = self.root / "ranks" / f"rank_{rank:05d}"
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.root / "shards").mkdir(exist_ok=True)
        for path in (
            self.directory,
            self.directory.parent,
            self.root / "shards",
            self.root,
        ):
            sync_dir(path, metrics)
        self.done = set()
        self.sequence = 0
        for path in sorted(self.directory.glob("part_*.json")):
            meta = json.loads(path.read_text())
            if not args.resume or meta["input_identity"] != identity:
                raise ValueError("Committed tar identity mismatch")
            if sha256_file(self.root / meta["path"]) != meta["sha256"]:
                raise ValueError("Committed tar checksum mismatch")
            if self.done.intersection(meta["source_indices"]):
                raise ValueError("Duplicate committed source ID")
            self.done.update(meta["source_indices"])
            self.sequence = max(self.sequence, meta["sequence"] + 1)
        if args.resume:
            for path in (self.root / "shards").glob(f".rank_{rank:05d}_part_*.tar.tmp"):
                path.unlink()
        self.seen = set(self.done)
        self.writer = None
        self.pending = deque()
        self.executor = ThreadPoolExecutor(max_workers=args.tar_commit_workers)

    def reap(self, wait=False):
        for future in self.pending:
            if future.done():
                future.result()
        while self.pending and (wait or self.pending[0].done()):
            self.done.update(self.pending.popleft().result()["source_indices"])
            if not wait:
                continue

    def add(self, index, value, metadata):
        self.reap()
        if index in self.seen:
            raise ValueError("Duplicate source ID")
        size = value.size if isinstance(value, RawSample) else len(value)
        if (
            self.writer is not None
            and self.writer.offset + 512 + size > self.args.tar_shard_bytes
        ):
            self.commit()
        if self.writer is None:
            self.writer = StreamTar(
                self.root, self.rank, self.sequence, self.args, self.metrics
            )
        extension = "bin"
        name = f"samples/{index:012d}.{extension}"
        with self.metrics.time("tar_write"):
            span = self.writer.member(name, value)
        self.writer.indices.append(index)
        self.writer.metadata.append(metadata)
        self.writer.names.append(name)
        self.writer.ranges.append(span)
        self.seen.add(index)

    def commit(self):
        writer = self.writer
        if writer is None:
            return
        meta = {
            "schema_version": 2,
            "kind": "final_rank",
            "rank": self.rank,
            "sequence": self.sequence,
            "input_identity": self.identity,
            "path": str(writer.final.relative_to(self.root)),
            "sample_count": len(writer.indices),
            "source_indices": writer.indices,
            "sample_metadata": writer.metadata,
            "member_names": writer.names,
            "member_ranges": writer.ranges,
        }
        writer.seal(meta)
        sidecar = self.directory / f"part_{self.sequence:08d}.json"
        self.writer = None
        self.sequence += 1
        if self.executor:
            try:
                with self.metrics.time("commit_queue_wait"):
                    while len(self.pending) >= self.args.tar_pending_commits:
                        self.done.update(
                            self.pending.popleft().result()["source_indices"]
                        )
                    self.reap()
                self.pending.append(self.executor.submit(writer.publish, meta, sidecar))
            except BaseException:
                writer.handle.close()
                raise
        else:
            self.done.update(writer.publish(meta, sidecar)["source_indices"])

    def finish(self):
        try:
            self.commit()
            with self.metrics.time("commit_drain"):
                self.reap(wait=True)
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)


def finalize(root, identity, metadata, count, metrics):
    root = Path(root)
    (shards, indices, ordinal) = ([], [], 0)
    with metrics.time("final_index_read"):
        for path in sorted((root / "ranks").glob("rank_*/part_*.json")):
            shard = json.loads(path.read_text())
            assert (
                shard["input_identity"] == identity and (root / shard["path"]).is_file()
            )
            size = shard["sample_count"]
            assert (
                size
                == len(shard["source_indices"])
                == len(shard["member_names"])
                == len(shard["member_ranges"])
            )
            (shard["success_start"], shard["success_end"]) = (
                ordinal,
                ordinal + size - 1,
            )
            ordinal += size
            indices.extend(shard["source_indices"])
            shards.append(shard)
    assert len(indices) == count and sorted(indices) == list(range(count)), (
        "Incomplete or duplicate source coverage"
    )
    manifest = {
        **metadata,
        "schema_version": 2,
        "cache_format": "tar-raw-v1",
        "logical_order": "source_index",
        "complete": True,
        "input_identity": identity,
        "successful_samples": count,
        "failed_samples": 0,
        "shard_size": None,
        "shards": shards,
    }
    durable_json(root / "manifest.json", manifest, metrics)
    return manifest


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


def persist_cache(root, metrics):
    with metrics.time("fsync"):
        for path in root.rglob("*"):
            if path.is_file() and not path.name.endswith(".tmp"):
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        for directory in [root / "shards", root]:
            if directory.exists():
                fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)


METADATA_MEMBER = "_metadata.json"
REGEN_FULL_SCHEMA_VERSION = 2
SAMPLE_INDEX_VERSION = 1


class HiddenCacheError(ValueError):
    """Raised when a regen_full cache cannot safely be used for training."""


def _same_model(expected: str, actual: str) -> bool:
    expected_path = Path(expected)
    actual_path = Path(actual)
    if expected_path.exists() and actual_path.exists():
        return expected_path.resolve() == actual_path.resolve()
    return expected.rstrip("/\\") == actual.rstrip("/\\")


def load_hidden_cache_manifest(
    root: str | Path,
    *,
    target_model: str,
    num_hidden_layers: int,
    hidden_size: int,
    required_layer_ids: set[int],
    max_length: int,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise HiddenCacheError(f"regen_full manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HiddenCacheError(f"cannot read regen_full manifest: {exc}") from exc
    if (
        manifest.get("cache_format") != "tar-raw-v1"
        or manifest.get("schema_version") != 2
    ):
        raise HiddenCacheError("Only tar-raw-v1 hidden caches are supported")
    if manifest.get("save_mode") != "regen_full" or not manifest.get("complete"):
        raise HiddenCacheError("regen_full cache is not marked complete")
    if manifest.get("sample_index_version") != SAMPLE_INDEX_VERSION:
        raise HiddenCacheError(
            "regen_full cache has no lightweight sample index; regenerate the cache with the current capture_hidden_states.py"
        )
    if not _same_model(target_model, str(manifest.get("target_model", ""))):
        raise HiddenCacheError(
            f"regen_full target model does not match training target: cache={manifest.get('target_model')!r}, training={target_model!r}"
        )
    if int(manifest.get("num_hidden_layers", -1)) != int(num_hidden_layers):
        raise HiddenCacheError("regen_full target layer count does not match training")
    if int(manifest.get("hidden_size", -1)) != int(hidden_size):
        raise HiddenCacheError("regen_full hidden size does not match training")
    if int(max_length) > int(manifest.get("max_length", -1)):
        raise HiddenCacheError(
            f"training max_length={max_length} exceeds cache max_length={manifest.get('max_length')}"
        )
    final_layer_id = int(manifest.get("final_norm_layer_id", -1))
    if final_layer_id != int(num_hidden_layers) - 1:
        raise HiddenCacheError("regen_full final-norm layer ID is invalid")
    stored_layer_ids = {
        int(value) for value in manifest.get("stored_hidden_layer_ids", [])
    }
    missing_layer_ids = sorted(set(required_layer_ids) - stored_layer_ids)
    if (
        manifest.get("recompute_gemma4_final_norm")
        and final_layer_id not in stored_layer_ids
    ):
        missing_layer_ids = sorted(set(missing_layer_ids) | {final_layer_id})
    if missing_layer_ids:
        needed_layer_ids = sorted(set(required_layer_ids) | set(missing_layer_ids))
        raise HiddenCacheError(
            f"regen_full is missing required hidden layers {missing_layer_ids}; regenerate with --target-layer-ids {','.join(map(str, needed_layer_ids))}"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise HiddenCacheError("regen_full manifest contains no shards")
    expected_start = 0
    counted = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise HiddenCacheError("regen_full shard entry must be an object")
        path = root / str(shard.get("path", ""))
        if not path.is_file():
            raise HiddenCacheError(f"regen_full shard does not exist: {path}")
        start = int(shard.get("success_start", -1))
        end = int(shard.get("success_end", -1))
        count = int(shard.get("sample_count", -1))
        if start != expected_start or end != start + count - 1 or count <= 0:
            raise HiddenCacheError(f"regen_full shard range is invalid: {path}")
        source_indices = shard.get("source_indices", [])
        if len(source_indices) != count:
            raise HiddenCacheError(
                f"regen_full shard source index count is invalid: {path}"
            )
        sample_metadata = shard.get("sample_metadata", [])
        if len(sample_metadata) != count:
            raise HiddenCacheError(
                f"regen_full shard sample metadata count is invalid: {path}"
            )
        for value in sample_metadata:
            _validate_sample_metadata(value, path)
        names = shard.get("member_names", [])
        ranges = shard.get("member_ranges", [])
        if len(names) != count or len(ranges) != count or len(set(names)) != count:
            raise HiddenCacheError(f"invalid indexed tar members: {path}")
        previous_end = 0
        file_size = path.stat().st_size
        for name, span, source in zip(names, ranges, source_indices):
            extension = "bin"
            if (
                name != f"samples/{int(source):012d}.{extension}"
                or not isinstance(span, list)
                or len(span) != 2
            ):
                raise HiddenCacheError(f"invalid indexed tar entry: {path}")
            (offset, size) = span
            if (
                not all((isinstance(v, int) for v in span))
                or offset % 512
                or offset < previous_end
                or (size <= 0)
                or (offset + size > file_size)
            ):
                raise HiddenCacheError(f"invalid indexed tar byte range: {path}")
            previous_end = offset + size
        if verify_checksums and sha256_file(path) != shard.get("sha256"):
            raise HiddenCacheError(f"regen_full shard checksum mismatch: {path}")
        expected_start = end + 1
        counted += count
    if counted != int(manifest.get("successful_samples", -1)):
        raise HiddenCacheError("regen_full successful sample count is inconsistent")
    indices = [i for shard in shards for i in shard["source_indices"]]
    if len(set(indices)) != len(indices):
        raise HiddenCacheError("regen_full contains duplicate source indices")
    return manifest


def _validate_sample_metadata(value: Any, path: Path | None = None) -> None:
    location = f" in {path}" if path is not None else ""
    if not isinstance(value, dict):
        raise HiddenCacheError(
            f"regen_full sample metadata must be an object{location}"
        )
    length = value.get("sequence_length")
    count = value.get("supervised_token_count")
    runs = value.get("loss_mask_runs")
    if not isinstance(length, int) or length <= 0:
        raise HiddenCacheError(f"invalid indexed sequence length{location}")
    if not isinstance(count, int) or count < 0 or count > length:
        raise HiddenCacheError(f"invalid indexed supervision count{location}")
    if not isinstance(runs, list):
        raise HiddenCacheError(f"invalid indexed loss-mask runs{location}")
    previous_end = 0
    observed_count = 0
    for run in runs:
        if (
            not isinstance(run, list)
            or len(run) != 2
            or (not all((isinstance(item, int) for item in run)))
        ):
            raise HiddenCacheError(f"invalid indexed loss-mask run{location}")
        (start, end) = run
        if start < previous_end or start < 0 or end <= start or (end > length):
            raise HiddenCacheError(f"invalid indexed loss-mask range{location}")
        observed_count += end - start
        previous_end = end
    if observed_count != count:
        raise HiddenCacheError(f"indexed supervision count mismatch{location}")
