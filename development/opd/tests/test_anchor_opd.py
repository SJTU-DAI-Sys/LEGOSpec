"""Numerical and lifecycle checks for full-batch sampled-anchor OPD."""

import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from legospec.training.execution.sampled import verify_sampled_blocks
from legospec.training.execution.step import TrainingStep
from legospec.modeling.draft.composable import LegoSpecConfig, LegoSpecModel
from legospec.modeling.target.anchor_opd import Qwen3AnchorScorer


def target_model(device="cpu", dtype=torch.float32):
    torch.manual_seed(9)
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=23,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return (
        Qwen3ForCausalLM(config)
        .to(device=device, dtype=dtype)
        .requires_grad_(False)
        .eval()
    )


def fixture(proposal="parallel", device="cpu", dtype=torch.float32):
    target = target_model(device, dtype)
    prop = {"type": proposal, **({"rank": 4} if proposal == "rnn" else {})}
    config = LegoSpecConfig(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_target_layers=2,
        vocab_size=23,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        block_size=4,
        attention_dropout=0.0,
        composable_config={
            "context": {"type": "concat_projection", "target_layer_ids": [0]},
            "backbone": {
                "type": "layer_stack",
                "layers": [{"attention": "gqa", "access": "dense"}],
            },
            "attention_configs": {"gqa": {}},
            "proposal": prop,
            "objective": {"type": "e2e_tv", "loss_decay_gamma": None},
        },
    )
    draft = LegoSpecModel(config).to(device=device, dtype=dtype)
    wrapper = TrainingStep(
        draft,
        target.lm_head,
        target.model.embed_tokens,
        3,
        num_anchors=4,
        objective_chunk_blocks=0,
        opd_data_mode="sampled_anchor",
    ).to(device)
    scorer = Qwen3AnchorScorer(target)
    ids = torch.randint(23, (2, 12), device=device)
    mask = torch.ones_like(ids).float()
    mask[1, 10:] = 0
    return (
        wrapper,
        scorer,
        dict(
            input_ids=ids,
            loss_mask=mask,
            context_valid_mask=mask.bool(),
            target_hidden_states=None,
        ),
    )


class AnchorOPDTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_shared_prefix_matches_independent_causal_forward(self):
        model = target_model()
        scorer = Qwen3AnchorScorer(model)
        ids = torch.randint(23, (2, 12))
        valid = torch.ones_like(ids).bool()
        valid[1, 10:] = False
        anchors = torch.tensor([[1, 5, 7], [2, 6, 0]])
        keep = torch.tensor([[True, True, True], [True, True, False]])
        draft = torch.randint(23, (2, 3, 3))
        prefix = scorer.prefill_anchors(
            ids, valid, anchors, keep, capture_layer_ids=[0, 1]
        )
        actual = scorer.score_sampled_blocks(prefix, draft)
        for n, a in keep.nonzero().tolist():
            length = anchors[n, a].item() + 1
            sequence = torch.cat((ids[n, :length], draft[n, a, :-1]))[None]
            expected = model.model(
                sequence, output_hidden_states=True
            ).last_hidden_state[:, length - 1 :]
            torch.testing.assert_close(actual[n, a], expected[0], atol=2e-6, rtol=2e-5)
        full = model.model(ids[:, :8], output_hidden_states=True)
        torch.testing.assert_close(
            prefix.context_hidden[0], full.hidden_states[1], atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            model.model.norm(prefix.context_hidden[1]),
            full.last_hidden_state,
            atol=2e-6,
            rtol=2e-5,
        )
        self.assertEqual(prefix.key_values[0][0].shape[0], 2)
        prefix.release()
        self.assertEqual(prefix.key_values, [])

    def test_no_future_or_other_branch_leakage(self):
        scorer = Qwen3AnchorScorer(target_model())
        ids = torch.randint(23, (1, 12))
        valid = torch.ones_like(ids).bool()
        anchors = torch.tensor([[2, 7]])
        keep = torch.ones_like(anchors).bool()
        draft = torch.randint(23, (1, 2, 3))
        prefix = scorer.prefill_anchors(ids, valid, anchors, keep)
        before = scorer.score_sampled_blocks(prefix, draft)
        other = draft.clone()
        other[:, 0] = (other[:, 0] + 1) % 23
        after = scorer.score_sampled_blocks(prefix, other)
        torch.testing.assert_close(before[:, 1], after[:, 1])
        changed = ids.clone()
        changed[:, 3:] = (changed[:, 3:] + 1) % 23
        after = scorer.score_sampled_blocks(
            scorer.prefill_anchors(changed, valid, anchors, keep), draft
        )
        torch.testing.assert_close(before[:, 0], after[:, 0])

    def test_probabilistic_verifier_and_first_rejection(self):
        p = torch.tensor([[[[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]]]])
        ids = torch.zeros(1, 1, 3).long()
        logq = torch.tensor([[[0.8, 0.9, 0.5]]]).log()
        # First acceptance probability is 0.25: it can reject even though the
        # same token is proposed. Later decisions cannot revive a prefix.
        for uniforms, expected in [([0.1, 0.2, 0.3], 3), ([0.3, 0.1, 0.1], 0)]:
            r = verify_sampled_blocks(
                p.log(), ids, logq, torch.tensor([[uniforms]]), torch.ones(1, 1).bool()
            )
            self.assertEqual(r.item(), expected)
        with self.assertRaises(ValueError):
            verify_sampled_blocks(
                p.log(), ids, logq, torch.ones_like(logq), torch.ones(1, 1).bool()
            )

    def test_fast_forward_and_checkpoint_do_not_resample_target(self):
        wrapper, scorer, values = fixture("rnn")
        calls = []
        hook = wrapper.draft_model.backbone.register_forward_hook(
            lambda *args: calls.append(1)
        )
        output = wrapper(**values, opd_scorer=scorer)
        self.assertEqual(len(calls), 1)
        output.loss.backward()
        hook.remove()
        self.assertEqual((scorer.prefill_calls, scorer.branch_calls), (1, 1))

    def test_cached_and_live_context_loss_gradients_match(self):
        wrapper, scorer, values = fixture("rnn")
        with torch.no_grad():
            original = scorer.model.model(
                values["input_ids"],
                attention_mask=values["context_valid_mask"],
                output_hidden_states=True,
            )
        cached_values = dict(
            values,
            target_hidden_states=original.hidden_states[1],
            target_last_hidden_states=original.last_hidden_state,
        )
        results = []
        for current in (values, cached_values):
            wrapper.zero_grad(set_to_none=True)
            torch.manual_seed(101)
            out = wrapper(**current, opd_scorer=scorer)
            out.loss.backward()
            results.append(
                (
                    out.loss.detach(),
                    {
                        n: p.grad.clone()
                        for n, p in wrapper.named_parameters()
                        if p.grad is not None
                    },
                )
            )
        torch.testing.assert_close(results[0][0], results[1][0], atol=3e-6, rtol=3e-5)
        for name in results[0][1]:
            torch.testing.assert_close(
                results[0][1][name], results[1][1][name], atol=3e-6, rtol=3e-5
            )

    def test_empty_rank_and_no_complete_blocks(self):
        wrapper, scorer, values = fixture()
        values["loss_mask"][1] = 0
        output = wrapper(**values, opd_scorer=scorer)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        values["loss_mask"].zero_()
        wrapper.zero_grad(set_to_none=True)
        empty = wrapper(**values, opd_scorer=scorer)
        self.assertEqual(empty.loss.item(), 0)
        self.assertEqual(empty.loss_terms.denominator.item(), 0)
        empty.loss.backward()
        self.assertTrue(
            all(
                p.grad is None or not p.grad.count_nonzero()
                for p in wrapper.draft_model.parameters()
            )
        )

    def test_online_cli_defaults_and_explicit_chunk_rejection(self):
        from legospec.training.config import parse_args
        from contextlib import redirect_stderr
        import io

        argv = [
            "--target-model-path",
            "target",
            "--train-data-path",
            "data.jsonl",
            "--output-dir",
            "out",
            "--opd-data-mode",
            "sampled_anchor",
            "--init-from",
            "checkpoint",
            "--report-to",
            "none",
        ]
        args = parse_args(argv=argv)
        self.assertEqual(args.objective_chunk_blocks, 0)
        self.assertEqual(args.opd_target_chunk_anchors, 0)
        self.assertEqual(args.num_anchors, 512)
        # The existing SGLangAnchorScorer supports the same collocated TP=1 path.
        sglang_args = parse_args(argv=argv + ["--target-model-backend", "sglang"])
        self.assertEqual(sglang_args.target_model_backend, "sglang")
        for extra in (
            ["--objective-chunk-blocks", "32"],
            ["--opd-target-chunk-anchors", "128"],
            ["--tp-size", "2"],
        ):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv=argv + extra)


if __name__ == "__main__":
    unittest.main()
