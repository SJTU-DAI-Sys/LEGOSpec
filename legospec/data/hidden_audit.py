"""One-time complete finite-value audit for immutable offline hidden caches.

An audit includes full shard hashes. Reuse checks the manifest hash and shard
size/mtime identities at dataset construction, without re-reading tensor data.
The cache must remain unchanged while a loader is alive.
"""

from pathlib import Path
import hashlib
import json
import os
import time


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(part)
    return digest.hexdigest()


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
        or audit.get("manifest_sha256") != _hash(root / "manifest.json")
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
    from legospec.data.hidden_cache import RegenFullDataset

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
    return shard["path"], _hash(Path(root) / shard["path"]), len(dataset)


def audit_hidden_cache(root, audit_path, *, workers=1):
    """Read/check all samples and layers once, outside benchmark timing."""
    from legospec.data.hidden_cache import RegenFullDataset

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
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing

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
        digest = digests[name] if name in digests else _hash(root / name)
        if _identity(root / name) != identity:
            raise ValueError(f"Cache changed during finite audit: {name}")
        shards[name] = dict(identity, sha256=digest)
    value = dict(
        version=1,
        complete=True,
        root=str(root),
        manifest_sha256=_hash(root / "manifest.json"),
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
