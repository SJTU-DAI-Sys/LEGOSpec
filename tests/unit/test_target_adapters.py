"""Model semantics compose with execution backends without a model-specific trainer."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from legospec.modeling.target.adapters.base import TargetModelAdapter, get_model_adapter
from legospec.modeling.target.adapters.gemma import GemmaTargetAdapter
from legospec.modeling.target.adapters.qwen35 import Qwen35TargetAdapter


class Processor(nn.Module):
    def __init__(self):
        super().__init__()
        self.aux_hidden_states_provider = None

    def forward(self, ids, hidden, head, batch, aux_hidden_states=None):
        if aux_hidden_states is None:
            aux_hidden_states = self.aux_hidden_states_provider()
        return SimpleNamespace(
            aux_hidden_states=torch.cat(aux_hidden_states, -1),
            last_hidden_states=hidden,
        )


class FusedNorm(nn.Module):
    def forward(self, hidden, residual):
        hidden.add_(residual)
        residual.copy_(hidden)
        hidden.copy_(nn.functional.rms_norm(hidden, (hidden.shape[-1],)))
        return hidden, residual


class CaptureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CaptureBackbone()
        self.model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.model.norm = FusedNorm()
        self.logits_processor = Processor()

    def set_eagle3_layers_to_capture(self, ids):
        self.ids = ids

    def forward(self, hidden, residual):
        self.model.ids = self.ids
        normalized, aux = self.model(hidden, residual)
        return self.logits_processor(
            None, normalized, None, None, aux_hidden_states=aux
        )


class CaptureBackbone(nn.Module):
    def forward(self, hidden, residual):
        first = hidden.clone()
        normalized, _ = self.norm(hidden, residual)
        aux = [first] if 0 in self.ids else []
        return (normalized, aux) if aux else normalized


def test_sglang_capture_keeps_raw_final_before_inplace_norm_and_removes_hooks():
    adapter = TargetModelAdapter()
    model = CaptureModel()
    hidden, residual = torch.randn(4, 8), torch.randn(4, 8)
    raw = hidden + residual
    for ids in [[0, 1], [1], [0, 1]]:
        adapter.install_capture(model, ids)
        output = model(hidden.clone(), residual.clone())
        expected = torch.cat([hidden, raw], -1) if len(ids) == 2 else raw
        torch.testing.assert_close(output.aux_hidden_states, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            output.last_hidden_states, nn.functional.rms_norm(raw, (8,))
        )
        assert len(model.model.norm._forward_pre_hooks) == 1
    adapter.remove_capture()
    assert not model.model.norm._forward_pre_hooks
    assert not model.logits_processor._forward_pre_hooks


def test_qwen35_uses_native_capture_selector():
    model = CaptureModel()
    model.set_dflash_layers_to_capture = lambda ids: setattr(model, "capture_ids", ids)
    adapter = Qwen35TargetAdapter()
    adapter.install_capture(model, [0, 1])
    assert model.capture_ids == [0]
    assert len(adapter._capture_handles) == 2
    adapter.remove_capture()


def test_adapter_selection_and_backend_policy():
    default = get_model_adapter(SimpleNamespace(model_type="qwen3"))
    assert type(default) is TargetModelAdapter
    gemma = get_model_adapter(SimpleNamespace(model_type="gemma4_unified_text"))
    assert isinstance(gemma, GemmaTargetAdapter)
    kwargs = {"attention_backend": "fa3"}
    gemma.configure_backend("sglang", kwargs)
    assert kwargs["attention_backend"] == "triton"
    qwen = get_model_adapter(
        SimpleNamespace(model_type="qwen3", target_source_model_type="qwen3_5")
    )
    assert isinstance(qwen, Qwen35TargetAdapter)
    with pytest.raises(ValueError, match="SGLang"):
        qwen.configure_backend("hf", {})
    kwargs = {}
    qwen.configure_backend("sglang", kwargs)
    assert kwargs == {"attention_backend": "fa3"}
