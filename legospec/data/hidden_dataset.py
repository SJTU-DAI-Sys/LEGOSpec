"""Hidden-cache datasets, collation and full finite-value audits."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from legospec.data.hidden_cache import (
    HiddenCacheError,
    _validate_sample_metadata,
    load_raw,
    sha256_file,
)


def sample_metadata_has_valid_anchor(
    metadata: dict[str, Any],
    *,
    block_size: int,
    max_length: int,
    complete_blocks: bool = False,
) -> bool:
    """Evaluate anchor eligibility using only the lightweight cache index."""
    _validate_sample_metadata(metadata)
    block_size = int(block_size)
    effective_length = min(int(metadata["sequence_length"]), int(max_length))
    if effective_length < block_size:
        return False
    runs = metadata["loss_mask_runs"]
    supervised_count = sum(
        (
            max(0, min(end, effective_length) - start)
            for (start, end) in runs
            if start < effective_length
        )
    )
    if supervised_count < 2 * block_size:
        return False
    if complete_blocks:
        return any(
            (
                min(end, effective_length) - start >= block_size
                for (start, end) in runs
                if start < effective_length
            )
        )
    max_anchor = effective_length - block_size
    return any(
        (
            start <= max_anchor and min(end, effective_length) - start >= 2
            for (start, end) in runs
            if start < effective_length
        )
    )


class RegenFullDataset(Dataset):
    """Map-style dataset backed by regen_full tar members."""

    def __init__(
        self,
        root: str | Path,
        manifest: dict[str, Any],
        *,
        required_layer_ids: set[int],
        max_length: int,
        max_open_archives: int = 4,
        validation_mode: str = "per-read",
        audit_path: str | Path | None = None,
        selective_read: bool = True,
    ) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self.required_layer_ids = set(required_layer_ids)
        self.final_layer_id = int(manifest["final_norm_layer_id"])
        self.hidden_size = int(manifest["hidden_size"])
        self._final_norm = None
        if manifest.get("recompute_gemma4_final_norm"):
            if (
                manifest.get("context_feature_contract")
                != "decoder_outputs_pre_final_norm_v1"
            ):
                raise HiddenCacheError(
                    "Unsupported Gemma 4 final norm recomputation layout"
                )
            from legospec.modeling.config_utils import get_pretrained_config_dict
            from legospec.modeling.target.adapters.base import get_model_adapter

            adapter = get_model_adapter(
                get_pretrained_config_dict(manifest["target_model"])
            )
            config = adapter.config
            if (
                config.hidden_size != self.hidden_size
                or config.num_hidden_layers != self.final_layer_id + 1
            ):
                raise HiddenCacheError(
                    "Target checkpoint does not match cached feature dimensions"
                )
            self._final_norm = adapter.load_final_norm(manifest["target_model"])
        self.max_length = int(max_length)
        if validation_mode not in ("per-read", "audited"):
            raise ValueError("Unknown hidden-cache validation mode")
        self.validate_finite = validation_mode == "per-read"
        self.selective_read = bool(selective_read)
        if not self.validate_finite:
            validate_audit(
                self.root,
                manifest,
                audit_path,
                self.required_layer_ids | {self.final_layer_id},
            )
        self.max_open_archives = max(1, int(max_open_archives))
        self.entries: list[tuple[Path, str, int, dict[str, Any]]] = []
        self._member_ranges = {}
        for shard in manifest["shards"]:
            path = self.root / shard["path"]
            for offset, (source_index, sample_metadata) in enumerate(
                zip(shard["source_indices"], shard["sample_metadata"])
            ):
                self.entries.append(
                    (
                        path,
                        shard["member_names"][offset],
                        int(source_index),
                        sample_metadata,
                    )
                )
            for name, span in zip(shard["member_names"], shard["member_ranges"]):
                self._member_ranges[path, name] = span
        self.entries.sort(key=lambda value: value[2])
        self._archives: OrderedDict[Path, Any] = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_archives"] = OrderedDict()
        return state

    def close(self) -> None:
        archives = getattr(self, "_archives", {})
        for archive in archives.values():
            archive.close()
        archives.clear()

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return len(self.entries)

    def _load_payload(self, index: int) -> dict[str, Any]:
        (path, member_name, source_index, metadata) = self.entries[index]
        archive = self._archives.get(path)
        if archive is None:
            if len(self._archives) >= self.max_open_archives:
                (_, stale) = self._archives.popitem(last=False)
                stale.close()
            archive = path.open("rb")
            self._archives[path] = archive
        else:
            self._archives.move_to_end(path)

        (offset, size) = self._member_ranges[path, member_name]
        payload = load_raw(
            archive,
            offset,
            size,
            expect_final_hidden=self._final_norm is None,
            layer_ids=self.required_layer_ids | {self.final_layer_id}
            if self.selective_read
            else None,
            max_length=self.max_length if self.selective_read else None,
        )
        if payload["sample_index"] != source_index:
            raise HiddenCacheError("Raw member source index mismatch")
        return payload

    def eligible_indices(
        self, *, block_size: int, complete_blocks: bool = False
    ) -> list[int]:
        """Return valid entry indices without opening a tar file."""
        return [
            index
            for (index, (_, _, _, metadata)) in enumerate(self.entries)
            if sample_metadata_has_valid_anchor(
                metadata,
                block_size=block_size,
                max_length=self.max_length,
                complete_blocks=complete_blocks,
            )
        ]

    def shard_key(self, index: int) -> Path:
        return self.entries[index][0]

    def training_metadata(self, index: int) -> dict[str, Any]:
        """Plan anchors from the manifest without opening hidden-state shards."""
        (_, _, source_index, metadata) = self.entries[index]
        length = min(self.max_length, int(metadata["sequence_length"]))
        mask = torch.zeros(length)
        for start, end in metadata["loss_mask_runs"]:
            mask[start : min(end, length)] = 1
        return dict(
            input_ids=torch.zeros(length, dtype=torch.long),
            attention_mask=torch.ones(length, dtype=torch.bool),
            loss_mask=mask,
            sample_index=source_index,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        payload = self._load_payload(index)
        input_ids = torch.as_tensor(payload.get("input_ids"))[: self.max_length]
        attention_mask = torch.as_tensor(payload.get("attention_mask"))[
            : self.max_length
        ]
        loss_mask = torch.as_tensor(payload.get("loss_mask"))[: self.max_length]
        length = input_ids.numel()
        if input_ids.ndim != 1 or length == 0:
            raise HiddenCacheError("cached input_ids must be a non-empty vector")
        if (
            attention_mask.shape != input_ids.shape
            or loss_mask.shape != input_ids.shape
        ):
            raise HiddenCacheError("cached token and mask shapes do not match")
        raw_hidden = payload.get("hidden_states")
        if not isinstance(raw_hidden, dict):
            raise HiddenCacheError("cached hidden_states must be a dictionary")
        hidden_states = {int(key): value for (key, value) in raw_hidden.items()}
        final_hidden = payload.get("final_hidden_state")
        if self._final_norm is None:
            if not isinstance(final_hidden, torch.Tensor):
                raise HiddenCacheError("cached final_hidden_state is missing")
            final_hidden = final_hidden[: self.max_length]
        selected = {}
        layer_ids = self.required_layer_ids | (
            {self.final_layer_id} if self._final_norm is not None else set()
        )
        for layer_id in sorted(layer_ids):
            value = hidden_states.get(layer_id)
            if not isinstance(value, torch.Tensor):
                raise HiddenCacheError(f"cached layer {layer_id} is missing")
            value = value[: self.max_length]
            if (
                value.ndim != 2
                or value.shape[0] != length
                or value.shape[1] != self.hidden_size
            ):
                raise HiddenCacheError(f"cached layer {layer_id} has invalid shape")
            if self.validate_finite and (not torch.isfinite(value).all()):
                raise HiddenCacheError(f"cached layer {layer_id} contains NaN or Inf")
            selected[layer_id] = value
        if self._final_norm is not None:
            final_hidden = self._final_norm(selected[self.final_layer_id])
        if final_hidden.shape != (length, self.hidden_size) or (
            (self.validate_finite or self._final_norm is not None)
            and (not torch.isfinite(final_hidden).all())
        ):
            raise HiddenCacheError(
                "normalized final hidden has invalid shape or values"
            )
        return {
            "sample_index": int(payload["sample_index"]),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "hidden_states": selected,
            "final_hidden_state": final_hidden,
            "final_norm_layer_id": self.final_layer_id,
        }


class RegenFullCollator:
    def __init__(self, pad_token_id: int, pad_to_length: int | None = None) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_length = pad_to_length

    @staticmethod
    def _pad_to(tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.shape[1] == length:
            return tensor
        shape = (tensor.shape[0], length - tensor.shape[1], *tensor.shape[2:])
        return torch.cat((tensor, tensor.new_zeros(shape)), dim=1)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise HiddenCacheError("cannot collate an empty regen_full batch")
        seq_len = max((item["input_ids"].numel() for item in features))
        if self.pad_to_length is not None:
            if seq_len > self.pad_to_length:
                raise HiddenCacheError("cached sequence exceeds fixed padding length")
            seq_len = self.pad_to_length

        def pad_vectors(key: str, padding_value: int) -> torch.Tensor:
            value = pad_sequence(
                [item[key] for item in features],
                batch_first=True,
                padding_value=padding_value,
            )
            return self._pad_to(value, seq_len)

        layer_ids = sorted(features[0]["hidden_states"])
        if any((sorted(item["hidden_states"]) != layer_ids for item in features)):
            raise HiddenCacheError("cached samples contain different hidden layers")
        final_layer_id = int(features[0]["final_norm_layer_id"])
        if any(
            (int(item["final_norm_layer_id"]) != final_layer_id for item in features)
        ):
            raise HiddenCacheError("cached samples disagree on final-norm layer ID")
        hidden_states = {}
        for layer_id in layer_ids:
            value = pad_sequence(
                [item["hidden_states"][layer_id] for item in features],
                batch_first=True,
                padding_value=0.0,
            )
            hidden_states[layer_id] = self._pad_to(value, seq_len)
        return {
            "sample_index": torch.tensor(
                [item["sample_index"] for item in features], dtype=torch.long
            ),
            "input_ids": pad_vectors("input_ids", self.pad_token_id),
            "attention_mask": pad_vectors("attention_mask", 0),
            "loss_mask": pad_vectors("loss_mask", 0),
            "hidden_states": hidden_states,
            "final_hidden_state": self._pad_to(
                pad_sequence(
                    [item["final_hidden_state"] for item in features],
                    batch_first=True,
                    padding_value=0.0,
                ),
                seq_len,
            ),
            "final_norm_layer_id": final_layer_id,
        }


def _identity(path):
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def validate_audit(root, manifest, audit_path, required_layers):
    if audit_path is None:
        raise ValueError("audited hidden loading requires an audit receipt")
    root = Path(root).resolve()
    audit = json.loads(Path(audit_path).read_text())
    if (
        audit.get("version") != 1
        or audit.get("complete") is not True
        or audit.get("root") != str(root)
        or audit.get("manifest_sha256") != sha256_file(root / "manifest.json")
        or audit.get("manifest_content_sha256")
        != hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        or audit.get("sample_count") != manifest["successful_samples"]
        or not set(required_layers).issubset(audit.get("layer_ids", []))
    ):
        raise ValueError("Hidden-cache audit does not cover this cache")
    shards = {shard["path"] for shard in manifest["shards"]}
    if shards != set(audit.get("shards", {})):
        raise ValueError("Hidden-cache audit shard set mismatch")
    for name in shards:
        observed = _identity(root / name)
        if any(observed[key] != audit["shards"][name][key] for key in observed):
            raise ValueError(f"Hidden cache changed after its audit: {name}")
    return audit


def _audit_shard(root, manifest, shard):
    # Spawned CPU-only workers never inherit a live CUDA context.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(1)

    subset = dict(manifest, shards=[shard])
    length = max(meta["sequence_length"] for meta in shard["sample_metadata"])
    dataset = RegenFullDataset(
        root,
        subset,
        required_layer_ids=set(manifest["stored_hidden_layer_ids"]),
        max_length=length,
        selective_read=False,
    )
    try:
        for index in range(len(dataset)):
            dataset[index]
    finally:
        dataset.close()
    return shard["path"], sha256_file(Path(root) / shard["path"]), len(dataset)


def audit_hidden_cache(root, audit_path, *, workers=1):
    """Read/check all samples and layers once, outside benchmark timing."""

    root, audit_path = Path(root).resolve(), Path(audit_path)
    manifest = json.loads((root / "manifest.json").read_text())
    layers = set(manifest["stored_hidden_layer_ids"])
    try:
        return validate_audit(root, manifest, audit_path, layers)
    except (OSError, ValueError, KeyError):
        pass
    begin = time.perf_counter()
    before = {
        shard["path"]: _identity(root / shard["path"]) for shard in manifest["shards"]
    }
    full_length = max(
        meta["sequence_length"]
        for shard in manifest["shards"]
        for meta in shard["sample_metadata"]
    )
    digests = {}
    checked = 0
    if workers > 1 and len(manifest["shards"]) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(
            max_workers=min(workers, len(manifest["shards"])),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            jobs = [
                pool.submit(_audit_shard, str(root), manifest, shard)
                for shard in manifest["shards"]
            ]
            for future in as_completed(jobs):
                name, digest, count = future.result()
                digests[name] = digest
                checked += count
                print(
                    f"hidden finite audit {checked}/{manifest['successful_samples']}",
                    flush=True,
                )
    else:
        dataset = RegenFullDataset(
            root,
            manifest,
            required_layer_ids=layers,
            max_length=full_length,
            selective_read=False,
        )
        try:
            for index in range(len(dataset)):
                dataset[index]
                if index % 32 == 0:
                    print(f"hidden finite audit {index + 1}/{len(dataset)}", flush=True)
            checked = len(dataset)
        finally:
            dataset.close()
    shards = {}
    for name, identity in before.items():
        digest = digests[name] if name in digests else sha256_file(root / name)
        if _identity(root / name) != identity:
            raise ValueError(f"Cache changed during finite audit: {name}")
        shards[name] = dict(identity, sha256=digest)
    value = dict(
        version=1,
        complete=True,
        root=str(root),
        manifest_sha256=sha256_file(root / "manifest.json"),
        manifest_content_sha256=hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        sample_count=checked,
        layer_ids=sorted(layers),
        shards=shards,
        audit_seconds=time.perf_counter() - begin,
        timing_scope="one-time audit, excluded from all performance intervals",
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, audit_path)
    return value
