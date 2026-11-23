from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from legospec.modeling.target.sglang import SGLangTrainingTargetModel


@pytest.mark.parametrize("return_logits", [False, True])
def test_sglang_prefill_trims_suffix_and_preserves_causal_positions(return_logits):
    ids = torch.tensor([[2, 3, 0, 0], [4, 0, 5, 0], [1, 4, 1, 2]])
    attention = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    loss = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]])

    def capture(records):
        tokens = [record["input_ids"].tolist() for record in records]
        assert tokens == [[2, 3], [4, 0, 5], [1, 4, 1, 2]]
        causal = torch.cat(
            [torch.tensor(row).float().cumsum(0)[:, None] for row in tokens]
        )
        return SimpleNamespace(
            hidden_states={0: causal, 2: causal + 10},
            final_hidden_state=causal + 20,
            token_offsets=[0, 2, 5, 9],
        )

    engine = SimpleNamespace(layer_ids=[0, 2], generate_capture_data=capture)
    backend = SGLangTrainingTargetModel(SimpleNamespace(), adapter=None, engine=engine)
    head = lambda hidden: torch.cat((hidden, -hidden), dim=-1)
    with patch.object(
        backend, "build_shared_components", return_value=SimpleNamespace(lm_head=head)
    ):
        output = backend.generate_training_data(ids, attention, loss, return_logits)

    kept = torch.tensor(
        [[True, True, False, False], [True, True, True, False], [True] * 4]
    )
    expected = ids.float().cumsum(1).unsqueeze(-1)
    torch.testing.assert_close(
        output.hidden_states[0], expected.masked_fill(~kept[..., None], 0)
    )
    torch.testing.assert_close(
        output.hidden_states[2], (expected + 10).masked_fill(~kept[..., None], 0)
    )
    torch.testing.assert_close(
        output.final_hidden_state, (expected + 20).masked_fill(~kept[..., None], 0)
    )
    torch.testing.assert_close(output.input_ids, ids)
    torch.testing.assert_close(output.attention_mask, attention)
    torch.testing.assert_close(output.loss_mask, loss)
    if return_logits:
        logits = torch.cat((expected + 20, -expected - 20), dim=-1)
        torch.testing.assert_close(
            output.logits, logits.masked_fill(~kept[..., None], 0)
        )
    else:
        assert output.logits is None
