"""Packed and tensor D2H paths preserve features across pinned-slot reuse."""

from contextlib import nullcontext
import io
from types import SimpleNamespace

import pytest
import torch

from legospec.data.capture_engine import Pipeline
from legospec.data.hidden_tar_raw import load_raw


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transfer pipeline")
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("include_final_context", [False, True])
def test_async_capture_preserves_reused_slot_contents(packed, include_final_context):
    class Metrics:
        def time(self, *args):
            return nullcontext()

        def gpu(self, *args):
            return nullcontext()

    class Sink:
        def __init__(self):
            self.samples = {}

        def add(self, index, value, metadata):
            encoded = value.header + b"".join(bytes(part) for part in value.parts)
            self.samples[index] = load_raw(io.BytesIO(encoded), 0, len(encoded))

        def finish(self):
            pass

    layer_ids = [0, 1] if include_final_context else [0]
    args = SimpleNamespace(
        pipeline_depth=2,
        writer_workers=2,
        max_batch_tokens=0,
        batch_size=2,
        max_length=3,
        capture_hidden_size=4,
        capture_layer_ids=layer_ids,
        separate_final=True,
        max_pending_bytes=1 << 30,
        packed_requests=packed,
        recompute_gemma4_final_norm=False,
        final_norm_layer_id=1,
        check_finite=True,
    )
    sink = Sink()
    pipeline = Pipeline(args, sink, Metrics())
    pointers = [slot.flat.data_ptr() for slot in pipeline.slots]
    for iteration in range(5):
        shape = (5, 4) if packed else (2, 3, 4)
        hidden = {
            layer: torch.full(
                shape, 10 * iteration + layer, device="cuda", dtype=torch.bfloat16
            )
            for layer in layer_ids
        }
        last = torch.full(
            shape, 10 * iteration + 7, device="cuda", dtype=torch.bfloat16
        )
        output = SimpleNamespace(
            hidden_states=hidden,
            final_hidden_state=last,
            packed=packed,
            attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], device="cuda"),
        )
        if packed:
            output.packed_aux = torch.stack([hidden[layer] for layer in layer_ids], 1)
            output.packed_last = last
            output.token_offsets = [0, 2, 5]
        records = [
            dict(
                source_index=2 * iteration + row,
                input_ids=torch.arange(length),
                attention_mask=torch.ones(length, dtype=torch.long),
                loss_mask=torch.ones(length),
            )
            for row, length in enumerate([2, 3])
        ]
        pipeline.submit(output, records)
    pipeline.finish()
    assert pointers == [slot.flat.data_ptr() for slot in pipeline.slots]
    assert len(sink.samples) == 10
    for index, sample in sink.samples.items():
        expected = 10 * (index // 2)
        assert torch.all(sample["hidden_states"][0] == expected)
        assert torch.all(sample["final_hidden_state"] == expected + 7)
        assert (1 in sample["hidden_states"]) == include_final_context
