"""Raw-tar durable commits, recovery and cache contract."""

from contextlib import nullcontext
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from legospec.data.capture_engine import metadata
from legospec.data.hidden_cache import (
    HiddenCacheError,
    RegenFullDataset,
    load_hidden_cache_manifest,
)
from legospec.data.hidden_tar_raw import RawSample
from legospec.data.hidden_tar_stream import StreamTar, StreamTarSink, finalize


class Metrics:
    def time(self, *args):
        return nullcontext()


def args(resume=False):
    return SimpleNamespace(
        resume=resume,
        tar_buffer_bytes=4096,
        tar_shard_bytes=1024,
        tar_commit_workers=1,
        tar_pending_commits=2,
    )


def sample(index):
    return dict(
        schema_version=2,
        sample_index=index,
        input_ids=torch.arange(3),
        attention_mask=torch.ones(3, dtype=torch.long),
        loss_mask=torch.ones(3),
        hidden_states={0: torch.full((3, 4), index, dtype=torch.bfloat16)},
        final_hidden_state=torch.full((3, 4), index + 10, dtype=torch.bfloat16),
    )


def test_stream_commit_resume_and_manifest_roundtrip(tmp_path):
    identity = {"source": "fixture"}
    sink = StreamTarSink(args(), tmp_path, identity, 0, Metrics())
    for index in range(3):
        value = sample(index)
        sink.add(index, RawSample(value), metadata(value))
    sink.finish()
    assert sink.done == {0, 1, 2}
    resumed = StreamTarSink(args(True), tmp_path, identity, 0, Metrics())
    assert resumed.done == sink.done
    value = sample(3)
    resumed.add(3, RawSample(value), metadata(value))
    resumed.finish()
    manifest = finalize(
        tmp_path,
        identity,
        dict(
            save_mode="regen_full",
            sample_index_version=1,
            target_model="fixture",
            num_hidden_layers=1,
            hidden_size=4,
            max_length=3,
            final_norm_layer_id=0,
            stored_hidden_layer_ids=[0],
        ),
        4,
        Metrics(),
    )
    for shard in manifest["shards"]:
        assert (
            shard["sha256"]
            == hashlib.sha256((tmp_path / shard["path"]).read_bytes()).hexdigest()
        )
    checked = load_hidden_cache_manifest(
        tmp_path,
        target_model="fixture",
        num_hidden_layers=1,
        hidden_size=4,
        required_layer_ids={0},
        max_length=3,
    )
    dataset = RegenFullDataset(tmp_path, checked, required_layer_ids={0}, max_length=3)
    try:
        for index in range(4):
            actual = dataset[index]
            torch.testing.assert_close(
                actual["hidden_states"][0], sample(index)["hidden_states"][0]
            )
            torch.testing.assert_close(
                actual["final_hidden_state"], sample(index)["final_hidden_state"]
            )
    finally:
        dataset.close()


def test_async_commit_errors_reach_finish(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("commit failure")

    monkeypatch.setattr(StreamTar, "publish", fail)
    sink = StreamTarSink(args(), tmp_path, {}, 0, Metrics())
    value = sample(0)
    sink.add(0, RawSample(value), metadata(value))
    with pytest.raises(OSError, match="commit failure"):
        sink.finish()
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("format", ["tar-v2", "tar-indexed-v2", "safetensors-v3"])
def test_removed_cache_formats_fail_at_boundary(tmp_path, format):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "cache_format": format})
    )
    with pytest.raises(HiddenCacheError, match="Only tar-raw-v1"):
        load_hidden_cache_manifest(
            tmp_path,
            target_model="fixture",
            num_hidden_layers=1,
            hidden_size=4,
            required_layer_ids={0},
            max_length=3,
        )
