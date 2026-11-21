from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from litespec.modeling.target.training_target_model import SGLangTrainingTargetModel


@pytest.mark.parametrize("return_logits", [False, True])
def test_sglang_prefill_trims_suffix_and_preserves_causal_positions(return_logits):
    ids = torch.tensor([[2, 3, 0, 0], [4, 0, 5, 0], [1, 4, 1, 2]])
    attention = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    loss = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]])
    backend = SGLangTrainingTargetModel(
        SimpleNamespace(model_config=SimpleNamespace(hidden_size=1, num_hidden_layers=3))
    )
    backend.capture_layer_ids = [0, 2]

    def fake_extend(requests, *, return_logits):
        tokens = [request.origin_input_ids for request in requests]
        assert tokens == [[2, 3], [4, 0, 5], [1, 4, 1, 2]]
        assert [request.extend_input_len for request in requests] == [2, 3, 4]
        causal = [torch.tensor(row).float().cumsum(0)[:, None] for row in tokens]
        return causal, [row + 10 for row in causal], (
            [torch.cat((row, -row), dim=-1) for row in causal] if return_logits else None
        )

    fake_sglang = {
        "sglang.srt.managers.schedule_batch": SimpleNamespace(
            Req=lambda **kwargs: SimpleNamespace(prefix_indices=(), **kwargs),
        ),
        "sglang.srt.sampling.sampling_params": SimpleNamespace(SamplingParams=SimpleNamespace),
    }
    with patch.dict("sys.modules", fake_sglang), patch.object(backend, "_extend", fake_extend):
        output = backend.generate_training_data(ids, attention, loss, return_logits)

    kept = torch.tensor([[True, True, False, False], [True, True, True, False], [True] * 4])
    expected = ids.float().cumsum(1).unsqueeze(-1)
    torch.testing.assert_close(output.hidden_states[0], expected.masked_fill(~kept[..., None], 0))
    torch.testing.assert_close(output.hidden_states[2], (expected + 10).masked_fill(~kept[..., None], 0))
    torch.testing.assert_close(output.input_ids, ids)
    torch.testing.assert_close(output.attention_mask, attention)
    torch.testing.assert_close(output.loss_mask, loss)
    if return_logits:
        logits = torch.cat((expected, -expected), dim=-1)
        torch.testing.assert_close(output.logits, logits.masked_fill(~kept[..., None], 0))
    else:
        assert output.logits is None
