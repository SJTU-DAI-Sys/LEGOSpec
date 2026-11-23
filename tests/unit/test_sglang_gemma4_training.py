"""The native capture adapter must preserve features and trainable input grads."""

import legospec.training.target.setup as target_features

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from legospec.modeling.target.sglang import SGLangTrainingTargetModel
from legospec.modeling.target.adapters.gemma import GemmaTargetAdapter
from legospec.modeling.target.training_target_model import get_training_target_model
from legospec.training.runtime import setup as setup


class CaptureFixture:
    def __init__(self):
        self.layer_ids = [1, 3]

        class NativeScaledEmbedding(nn.Embedding):
            def forward(self, ids):
                return super().forward(ids) * 2.0

        embedding = NativeScaledEmbedding(32, 4).requires_grad_(False)
        self.runner = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(embed_tokens=embedding),
                lm_head=embedding,
                suppress_token_ids=[31],
            )
        )

    @torch.inference_mode()
    def generate_capture_data(self, records):
        self.lengths = [r["input_ids"].numel() for r in records]
        offsets = [0]
        for n in self.lengths:
            offsets.append(offsets[-1] + n)
        values = torch.arange(offsets[-1] * 4, dtype=torch.float32).reshape(-1, 4)
        return SimpleNamespace(
            hidden_states={1: values, 3: values + 10},
            final_hidden_state=values + 20,
            token_offsets=offsets,
        )


def target():
    engine = CaptureFixture()
    adapter = GemmaTargetAdapter(
        SimpleNamespace(hidden_size=4, vocab_size=32, final_logit_softcapping=3.0)
    )
    target = SGLangTrainingTargetModel(engine.runner, adapter=adapter, engine=engine)
    target.capture_layer_ids = [1, 3]
    return target


@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("logits", [False, True])
def test_trim_padding_restore_layers_and_remove_inference_tensors(padding, logits):
    t = target()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    mask = torch.ones_like(ids)
    if padding:
        mask[1, 2:] = 0
    out = t.generate_training_data(ids, mask, mask, logits)
    assert t.engine.lengths == ([4, 2] if padding else [4, 4])
    for rows in (*out.hidden_states.values(), out.final_hidden_state):
        assert rows.shape == (2, 4, 4) and not torch.is_inference(rows)
        if padding:
            assert not rows[1, 2:].any()
    torch.testing.assert_close(
        out.final_hidden_state[mask.bool()], out.hidden_states[3][mask.bool()] + 10
    )
    if logits:
        torch.testing.assert_close(
            out.logits, t.build_shared_components(0).lm_head(out.final_hidden_state)
        )
    else:
        assert out.logits is None
    projection = nn.Linear(4, 3)
    projection(out.hidden_states[1]).sum().backward()
    assert projection.weight.grad is not None


def test_shared_components_scale_embeddings_softcap_logits_and_reuse_storage():
    t = target()
    c = t.build_shared_components(4)
    native = t.model_runner.model.language_model.embed_tokens
    ids = torch.tensor([[1, 2]])
    torch.testing.assert_close(c.embed_tokens(ids), native(ids))
    assert c.embed_tokens is native
    assert c.lm_head.weight is native.weight
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    expected = (hidden @ native.weight.T / 3.0).tanh() * 3.0
    expected[..., 31] = torch.finfo(expected.dtype).min
    torch.testing.assert_close(c.lm_head(hidden), expected)
    c.lm_head(hidden).softmax(-1).square().sum().backward()
    assert hidden.grad is not None and native.weight.grad is None
    assert torch.isfinite(hidden.grad).all()
    assert c is t.build_shared_components(4)
    with pytest.raises(ValueError, match="MASK"):
        t.build_shared_components(32)


def test_capture_policy_updates_native_model_and_engine():
    t = target()
    with patch.object(t.adapter, "install_capture") as install:
        t.set_capture_layers([0])
    install.assert_called_once_with(t.model_runner.model, [0])
    assert t.engine.layer_ids == t.capture_layer_ids == [0]


def test_tp2_with_replicated_native_embedding_and_tied_head():
    t = target()
    native = t.model_runner.model.language_model.embed_tokens
    assert not hasattr(native, "org_vocab_size")
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("legospec.distributed.get_tp_group", return_value=object()),
    ):
        c = t.build_shared_components(4)
    assert c.embed_tokens is native and c.lm_head.weight is native.weight
    hidden = torch.randn(3, 4, requires_grad=True)
    actual = c.lm_head(hidden)
    expected = (hidden @ native.weight.T / 3.0).tanh() * 3.0
    expected[..., 31] = torch.finfo(expected.dtype).min
    torch.testing.assert_close(actual, expected)
    actual.softmax(-1).square().sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("shard_embedding", [False, True])
def test_tp2_selects_embedding_and_head_partitioning_independently(shard_embedding):
    from legospec.modeling.target.components import (
        SGLangTPEmbeddingAdapter,
        SGLangTPLMHeadAdapter,
    )

    t = target()

    class Shard(nn.Module):
        org_vocab_size = 32
        embedding_dim = 4
        num_added_embeddings = 0

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(16, 4), requires_grad=False)

        def get_sharded_to_full_mapping(self):
            return None

    t.model_runner.model.lm_head = Shard()
    if shard_embedding:
        t.model_runner.model.language_model.embed_tokens = Shard()
    native_embedding = t.model_runner.model.language_model.embed_tokens
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_rank", return_value=0),
        patch("legospec.distributed.get_tp_group", return_value=object()),
    ):
        c = t.build_shared_components(4)
    assert isinstance(c.lm_head, SGLangTPLMHeadAdapter)
    if shard_embedding:
        assert isinstance(c.embed_tokens, SGLangTPEmbeddingAdapter)
        assert c.embed_tokens.embedding is native_embedding
    else:
        assert c.embed_tokens is native_embedding


def test_factory_routes_gemma_through_common_backend():
    sentinel = object()
    config = SimpleNamespace(model_type="gemma4_unified_text", num_hidden_layers=48)
    with (
        patch(
            "legospec.modeling.config_utils.load_text_model_config", return_value=config
        ),
        patch.object(
            SGLangTrainingTargetModel, "from_pretrained", return_value=sentinel
        ) as load,
    ):
        got = get_training_target_model(
            "fixture", backend="sglang", capture_layer_ids=[1, 47]
        )
    assert got is sentinel and load.call_args.kwargs["capture_layer_ids"] == [1, 47]
    assert isinstance(load.call_args.kwargs["adapter"], GemmaTargetAdapter)
    assert load.call_args.kwargs["attention_backend"] == "triton"


def test_native_capture_ids_are_supplied_before_loading_the_teacher():
    args = SimpleNamespace(
        target_model_path="fixture",
        trust_remote_code=False,
        target_model_backend="sglang",
        tp_size=1,
        disaggregate=False,
        batch_size=1,
        max_length=4096,
        require_target_last_hidden=True,
    )
    config = SimpleNamespace(model_type="gemma4_unified_text", num_hidden_layers=48)
    backend = SimpleNamespace(
        to_kwargs=lambda: dict(max_running_requests=None, max_total_tokens=None)
    )
    captured = []
    model = SimpleNamespace(set_capture_layers=lambda ids: captured.append(ids))
    with (
        patch.object(target_features, "load_text_model_config", return_value=config),
        patch.object(target_features, "required_hidden_layer_ids", return_value={1, 5}),
        patch.object(
            target_features.SGLangBackendArgs, "from_args", return_value=backend
        ),
        patch.object(
            target_features, "get_training_target_model", return_value=model
        ) as load,
    ):
        assert target_features.build_target_model(args, []) is model
    assert load.call_args.kwargs["capture_layer_ids"] == [1, 5, 47]
    assert load.call_args.kwargs["context_length"] == 4097
    assert captured == []
