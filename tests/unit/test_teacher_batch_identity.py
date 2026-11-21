from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from test_vocab_reuse import wrapper
from litespec.core.composable.batch import layout_from_anchors


@pytest.mark.parametrize('proposal', ['parallel', 'markov'])
@pytest.mark.parametrize('diagnostics', [False, True])
def test_unpacked_teacher_keeps_equal_positions_in_different_samples(proposal, diagnostics):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    actual = wrapper('weighted_sum', proposal)
    actual.optimized = True
    actual.objective_chunk_blocks = 0
    actual.objective_gradient_checkpointing = False
    reference = deepcopy(actual)
    ids = torch.randint(31, (2, 24))
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[1, 16:] = False
    mask = valid.float()
    anchors = torch.tensor([[1, 4], [1, 4]])
    layout = layout_from_anchors(ids, mask, anchors, torch.ones_like(anchors, dtype=torch.bool), 8)
    data = dict(input_ids=ids, loss_mask=mask, context_valid_mask=valid,
        target_hidden_states=torch.randn(2, 24, 32),
        target_last_hidden_states=torch.randn(2, 24, 16), layout=layout,
        collect_detailed_metrics=diagnostics)
    def identity(ids, hidden, mask, last, valid, layout, *args, **kwargs):
        return ids, hidden, mask, last, valid, layout, None
    rows = []
    hook = actual.lm_head.register_forward_pre_hook(lambda _, args: rows.append(args[0].numel() // 16))
    with patch('litespec.core.composable.packing.pack_training_batch', identity):
        result = actual(**data)
        hook.remove()
        def repeated(head, hidden, indices, **kwargs):
            values = hidden.gather(1, indices.flatten(1)[..., None].expand(-1, -1, hidden.size(-1)))
            return head(values).reshape(*indices.shape, -1)
        with patch('litespec.core.composable.training_model.project_teacher_rows', repeated):
            expected = reference(**data)
    # Positions 1..10 are unique within each sample; both samples must survive.
    assert rows[1] == 20
    torch.testing.assert_close(result.loss_terms.numerator, expected.loss_terms.numerator)
    result.loss_terms.numerator.backward()
    expected.loss_terms.numerator.backward()
    for (name, parameter), (other, ref) in zip(actual.named_parameters(), reference.named_parameters()):
        assert name == other
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, ref.grad)
