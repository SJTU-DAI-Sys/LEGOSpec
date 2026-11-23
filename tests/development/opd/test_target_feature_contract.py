"""Keep pre-final-norm context separate from teacher features and logits."""

import legospec.training.target.features as target_features
import legospec.training.target.setup as target_setup
import legospec.training.target.streaming as target_streaming

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from legospec.modeling.target.hf import HFTrainingTargetModel
from legospec.modeling.target.anchor_opd import Qwen3AnchorScorer
from legospec.training.runtime import setup as setup


class TargetFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)
        c = Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
        c._attn_implementation = "eager"
        self.model = Qwen3ForCausalLM(c).eval()
        self.ids = torch.randint(32, (2, 7))
        self.mask = torch.ones_like(self.ids)
        self.backend = HFTrainingTargetModel(self.model)

    def reference(self):
        outputs = {}
        hook = self.model.model.layers[-1].register_forward_hook(
            lambda m, a, o: outputs.update(raw=o[0] if isinstance(o, tuple) else o)
        )
        try:
            with torch.no_grad():
                outputs["hf"] = self.model(
                    self.ids,
                    attention_mask=self.mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
        finally:
            hook.remove()
        return outputs

    def test_all_layers_and_teacher_match_hf(self):
        expected = self.reference()
        for logits in [False, True]:
            with self.subTest(logits=logits):
                got = self.backend.generate_training_data(
                    self.ids, self.mask, self.mask, return_logits=logits
                )
                self.assertEqual(len(got.hidden_states), 3)
                for i in [0, 1]:
                    self.assertTrue(
                        torch.equal(
                            got.hidden_states[i], expected["hf"].hidden_states[i + 1]
                        )
                    )
                self.assertTrue(torch.equal(got.hidden_states[2], expected["raw"]))
                self.assertTrue(
                    torch.equal(
                        got.final_hidden_state, expected["hf"].hidden_states[-1]
                    )
                )
                self.assertFalse(
                    torch.equal(got.hidden_states[2], got.final_hidden_state)
                )
                self.assertTrue(
                    torch.equal(
                        self.model.lm_head(got.final_hidden_state),
                        expected["hf"].logits,
                    )
                )
                if logits:
                    self.assertTrue(torch.equal(got.logits, expected["hf"].logits))
                else:
                    self.assertIsNone(got.logits)

    def test_selected_context_excludes_teacher_layer(self):
        self.backend.set_capture_layers([0])
        got = self.backend.generate_training_data(self.ids, self.mask, self.mask, False)
        draft = SimpleNamespace(
            target_layer_ids=[0], config=SimpleNamespace(num_target_layers=3)
        )
        with patch.object(target_features, "hidden_states_to_cuda", lambda x: x):
            context, teacher = target_features.prepare_target_features(
                {}, got, draft, True
            )
        self.assertTrue(torch.equal(context, got.hidden_states[0]))
        self.assertTrue(torch.equal(teacher, got.final_hidden_state))

    def test_context_selection_uses_absolute_layer_ids_and_draft_order(self):
        states = {i: torch.full((1, 2, 4), i, dtype=torch.float32) for i in [2, 7, 11]}
        got = target_features.concatenate_target_hidden(states, [11, 2])
        torch.testing.assert_close(got, torch.cat([states[11], states[2]], -1))

    def test_target_packet_uses_raw_context_and_normalized_teacher(self):
        self.backend.set_capture_layers([2])
        expected = self.backend.generate_training_data(
            self.ids, self.mask, self.mask, False
        )
        metadata = {
            "target_layer_ids": [2],
            "num_target_layers": 3,
            "include_target_last_hidden": True,
        }
        packet = target_streaming._target_packet(
            None,
            SimpleNamespace(is_target_leader=True),
            self.backend,
            metadata,
            (self.ids, self.mask, self.mask),
        )
        self.assertTrue(
            torch.equal(
                packet.target_context_hidden, expected.hidden_states[2].bfloat16()
            )
        )
        self.assertTrue(
            torch.equal(
                packet.target_last_hidden, expected.final_hidden_state.bfloat16()
            )
        )

    def test_hooks_are_removed_after_failure(self):
        norm = self.model.model.norm
        before = len(norm._forward_pre_hooks)
        with patch.object(
            self.model.model, "forward", side_effect=RuntimeError("fixture failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                self.backend.generate_training_data(
                    self.ids, self.mask, self.mask, False
                )
        self.assertEqual(len(norm._forward_pre_hooks), before)

    def test_nested_text_backbone_captures_raw_and_normalized_features(self):
        inner = self.model

        class NestedTarget(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = inner.model

            @property
            def base_model(self):
                return self.model

            def forward(self, **kwargs):
                return inner(**kwargs)

        backend = HFTrainingTargetModel(NestedTarget())
        backend.set_capture_layers([0, 2])
        expected = self.reference()
        for logits in [False, True]:
            got = backend.generate_training_data(self.ids, self.mask, self.mask, logits)
            torch.testing.assert_close(
                got.hidden_states[0], expected["hf"].hidden_states[1]
            )
            torch.testing.assert_close(got.hidden_states[2], expected["raw"])
            torch.testing.assert_close(
                got.final_hidden_state, expected["hf"].hidden_states[-1]
            )

    def test_online_hf_reuses_frozen_weights_and_preserves_softcap(self):
        self.model.config.final_logit_softcapping = 3.0
        tokenizer = SimpleNamespace(mask_token_id=None, get_vocab=lambda: {"x": 0})
        args = SimpleNamespace(
            target_model_path="fixture",
            trust_remote_code=False,
            mask_token_id=4,
            target_model_backend="hf",
        )
        draft = SimpleNamespace(
            uses_hidden_feedback=False,
            method_config_key="composable_config",
            config=SimpleNamespace(composable_config={}),
        )
        with (
            patch.object(
                target_setup.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            patch.object(
                target_setup,
                "load_text_model_config",
                return_value=self.model.config,
            ),
        ):
            _, components, _ = target_setup.resolve_tokenizer_and_components(
                args, [draft], target=self.backend
            )
        self.assertIs(components.embed_tokens, self.model.get_input_embeddings())
        self.assertIs(components.lm_head.weight, self.model.lm_head.weight)
        x = torch.randn(2, 3, 16, requires_grad=True)
        expected = torch.tanh(self.model.lm_head(x) / 3.0) * 3.0
        torch.testing.assert_close(components.lm_head(x), expected)
        components.lm_head(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))

    def test_sampled_anchor_context_is_raw_and_teacher_is_normalized(self):
        scorer = Qwen3AnchorScorer(self.model)
        anchors = torch.tensor([[1, 4], [2, 4]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        prefix = scorer.prefill_anchors(
            self.ids, self.mask.bool(), anchors, keep, capture_layer_ids=[0, 2]
        )
        with torch.no_grad():
            captured = {}
            handle = self.model.model.norm.register_forward_pre_hook(
                lambda m, a: captured.update(raw=a[0])
            )
            try:
                full = self.model.model(
                    self.ids[:, :5],
                    attention_mask=self.mask[:, :5],
                    output_hidden_states=True,
                )
            finally:
                handle.remove()
        torch.testing.assert_close(
            prefix.context_hidden[2], captured["raw"], rtol=2e-5, atol=2e-6
        )
        expected = full.last_hidden_state.gather(
            1, anchors[..., None].expand(-1, -1, 16)
        )
        torch.testing.assert_close(prefix.anchor_hidden, expected, rtol=2e-5, atol=2e-6)
        prefix.release()

    def test_cached_teacher_feature_is_not_inferred_from_context(self):
        raw, teacher = torch.randn(2, 7, 16), torch.randn(2, 7, 16)
        draft = SimpleNamespace(target_layer_ids=[2])
        data = {"hidden_states": {2: raw}, "final_hidden_state": teacher}
        with patch.object(torch.Tensor, "cuda", lambda tensor, **kwargs: tensor):
            context, got = target_features.prepare_target_features(
                data, None, draft, True
            )
            torch.testing.assert_close(context, raw)
            self.assertIs(got, teacher)
            del data["final_hidden_state"]
            with self.assertRaisesRegex(ValueError, "post-final-norm"):
                target_features.prepare_target_features(data, None, draft, True)
            _, absent = target_features.prepare_target_features(
                data, None, draft, False
            )
            self.assertIsNone(absent)


if __name__ == "__main__":
    unittest.main()
