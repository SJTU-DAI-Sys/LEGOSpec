"""Logging diagnostics must preserve results without full-vocabulary stacks."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from litespec.core.block_metrics import distribution_acceptance_probability
from litespec.core.composable.vocab import VocabCache
from litespec.modeling.draft.composable.proposal.selector import SelectorProposal


class AcceptanceMetricMemoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_token_only_selector_matches_dense_greedy_including_ties_and_negative_deltas(self):
        proposal = SelectorProposal(SimpleNamespace(hidden_size=8, vocab_size=31,
            initializer_range=.02, composable_config={'proposal': {
                'rank': 4, 'top_k': 3, 'selector_mode': 'full_vocab_residual'}}))
        base = torch.randn(130, 7, 31)
        base[0] = 0  # Exact full-vocabulary argmax tie breaking must survive.
        original = base.clone()
        candidates = base.topk(3, -1).indices
        lattice = torch.randn(130, 7, 3, 3) * 8
        hidden = torch.randn(130, 7, 8)
        anchors = torch.randint(31, (130,))
        with torch.no_grad(), patch.object(proposal, 'parallel_lattice', return_value=(lattice, candidates)):
            expected, _, _ = proposal.sample_full_vocab(base, hidden, anchors, 0.)
            with patch('torch.softmax', side_effect=AssertionError('Greedy IDs require no softmax')):
                actual, logits, probabilities = proposal.sample_full_vocab(
                    base, hidden, anchors, 0., return_distributions=False)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(base, original))
        self.assertIsNone(logits)
        self.assertIsNone(probabilities)
        with self.assertRaises(ValueError):
            proposal.sample_full_vocab(base, hidden, anchors, 1., return_distributions=False)

    def test_row_bounded_acceptance_matches_dense_with_or_without_cached_probabilities(self):
        for dtype in (torch.float32, torch.bfloat16):
            p = torch.randn(2, 19, 7, 31, dtype=dtype)
            q = torch.randn_like(p)
            pp, qp = p.float().softmax(-1), q.float().softmax(-1)
            expected = torch.minimum(pp, qp).sum(-1)
            for cached in (False, True):
                with self.subTest(dtype=dtype, cached=cached):
                    sizes = []
                    softmax = torch.softmax
                    def observe(value, *args, **kwargs):
                        sizes.append(value.numel())
                        return softmax(value, *args, **kwargs)
                    with patch('torch.softmax', side_effect=observe):
                        actual = distribution_acceptance_probability(p, q, row_chunk_size=17,
                            target_probabilities=pp if cached else None,
                            draft_probabilities=qp if cached else None)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(all(size <= 17 * 31 for size in sizes))
                    if cached:
                        self.assertEqual(sizes, [])

    def test_metric_cache_lookup_never_allocates(self):
        value = torch.randn(3, 31, requires_grad=True)
        cache = VocabCache()
        self.assertIsNone(cache.peek_detached_probabilities(value))
        self.assertFalse(cache._entries or cache._frozen_entries)
        probabilities = cache.distribution(value).probabilities()
        got = cache.peek_detached_probabilities(value)
        self.assertEqual(got.data_ptr(), probabilities.data_ptr())
        self.assertFalse(got.requires_grad)

    def test_empty_acceptance_and_invalid_chunk(self):
        value = torch.empty(1, 0, 7, 31)
        self.assertEqual(distribution_acceptance_probability(value, value).shape, (1, 0, 7))
        with self.assertRaises(ValueError):
            distribution_acceptance_probability(value, value, row_chunk_size=0)

    def test_logged_step_preserves_training_loss_and_gradients(self):
        from torch import nn
        from litespec.core.composable.batch import prepare_layout
        from litespec.core.composable.training_model import ComposableTrainingModel
        from litespec.modeling.draft.composable import ComposableDraftConfig, ComposableDraftModel
        from litespec.modeling.target.target_utils import FlattenedLMHead
        recipe = dict(context=dict(type='dflash', target_layer_ids=[0, 1]),
            backbone=dict(type='layer_stack', layers=[dict(attention='gqa', access='dense')]),
            proposal=dict(type='residual_selector', rank=4, top_k=3, selector_mode='full_vocab_residual'),
            objective=dict(type='selector_losses', loss_decay_gamma=None,
                base=dict(type='soft_ce', weight=.2),
                final=dict(type='dpal_renyi', weight=1., rho=.3, reduction='complete_block_mean')),
            attention_configs=dict(gqa={}), access_configs={})
        config = ComposableDraftConfig(composable_config=recipe, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_target_layers=4, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, vocab_size=31, block_size=8, execution_config=dict(attention_backend='sdpa'),
            attention_dropout=0.)
        model = ComposableTrainingModel(ComposableDraftModel(config),
            FlattenedLMHead(16, 31, bias=False).requires_grad_(False),
            nn.Embedding(31, 16).requires_grad_(False), 4, num_anchors=4,
            objective_chunk_blocks=0, objective_gradient_checkpointing=False, training_mode='optimized')
        logged = deepcopy(model)
        ids = torch.randint(31, (2, 24)); mask = torch.ones_like(ids).float()
        values = dict(input_ids=ids, loss_mask=mask, context_valid_mask=mask.bool(),
            target_hidden_states=torch.randn(2, 24, 32), target_last_hidden_states=torch.randn(2, 24, 16),
            layout=prepare_layout(ids, mask, 8, 4, complete_blocks=True))
        a = model(**values, collect_detailed_metrics=False)
        b = logged(**values, collect_detailed_metrics=True)
        torch.testing.assert_close(a.loss_terms.numerator, b.loss_terms.numerator, rtol=0, atol=0)
        a.loss_terms.numerator.backward(); b.loss_terms.numerator.backward()
        for (name, p), (other_name, q) in zip(model.named_parameters(), logged.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertEqual(p.grad is None, q.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
