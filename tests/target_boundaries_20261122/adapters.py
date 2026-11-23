from pathlib import Path
import textwrap,tarfile,ast
root=Path.cwd();target=root/'legospec/modeling/target'
(target/'adapters/base.py').write_text('''"""Default model semantics used by HF/SGLang and standalone components.

Adapters describe model structure and capture semantics; backends own execution,
TP communication, batching and resource lifetime. No training imports here.
"""
import torch
from torch import nn


class TargetModelAdapter:
    native_capture = False
    softcap_fp32 = False

    def __init__(self, config=None):
        self.config = getattr(config, "text_config", config)
        self._capture_handles = []

    @property
    def softcap(self):
        return getattr(self.config, "final_logit_softcapping", None)

    def configure_backend(self, backend, kwargs):
        pass

    def hf_backbone(self, model):
        backbone = getattr(model, "base_model", model)
        text = getattr(backbone, "language_model", backbone)
        return getattr(text, "model", text)

    def final_norm(self, backbone):
        for name in ("norm", "final_layernorm", "final_layer_norm", "ln_f"):
            norm = getattr(backbone, name, None)
            if isinstance(norm, nn.Module):
                return norm
        raise ValueError("Target adapter requires an identifiable final norm")

    def sglang_components(self, model):
        return model.get_input_embeddings(), model.lm_head

    def new_embedding(self):
        return nn.Embedding(
            self.config.vocab_size, self.config.hidden_size,
            padding_idx=getattr(self.config, "pad_token_id", None),
        )

    def configure_capture(self, model, layer_ids):
        model.set_eagle3_layers_to_capture(layer_ids)

    def install_capture(self, model, layer_ids):
        """Install instance-local hooks; replacing capture policy removes old hooks."""
        self.remove_capture()
        self.configure_capture(model, layer_ids)
        final_id = len(model.model.layers) - 1
        if final_id not in layer_ids:
            return
        from legospec.modeling.target.sglang_backend.utils import LogitsProcessorForEAGLE3
        captured = {}
        def capture_raw(module, args, kwargs):
            hidden = args[0] if args else kwargs["x"]
            residual = args[1] if len(args) > 1 else kwargs.get("residual")
            # Fused RMSNorm may update its inputs in-place. Capture the residual
            # stream before that update, preserving the native addition dtype.
            captured["raw"] = hidden.clone() if residual is None else hidden + residual
        self._capture_handles.append(
            self.final_norm(model.model).register_forward_pre_hook(capture_raw, with_kwargs=True)
        )
        for processor in model.modules():
            if isinstance(processor, LogitsProcessorForEAGLE3):
                def append_final(module, args, kwargs):
                    aux = kwargs.get("aux_hidden_states", args[4] if len(args) > 4 else None)
                    if aux is None and module.aux_hidden_states_provider is not None:
                        aux = module.aux_hidden_states_provider()
                    aux = [*(aux or []), captured.pop("raw")]
                    if len(args) > 4:
                        args = (*args[:4], aux, *args[5:])
                    else:
                        kwargs["aux_hidden_states"] = aux
                    return args, kwargs
                self._capture_handles.append(processor.register_forward_pre_hook(append_final, with_kwargs=True))

    def remove_capture(self):
        for handle in self._capture_handles:
            handle.remove()
        self._capture_handles.clear()


def get_model_adapter(config):
    from legospec.modeling.config_utils import is_qwen35_model_type
    config = getattr(config, "text_config", config)
    model_type = getattr(config, "target_source_model_type", getattr(config, "model_type", ""))
    if model_type == "gemma4_unified_text":
        from .gemma import GemmaTargetAdapter
        return GemmaTargetAdapter(config)
    if is_qwen35_model_type(model_type):
        from .qwen35 import Qwen35TargetAdapter
        return Qwen35TargetAdapter(config)
    return TargetModelAdapter(config)
''')
# Current Qwen communicator patch stays instance-local in its model adapter.
patch=(root/'tests/target_boundaries_20261122/communicator.txt').read_text()
qwen='''"""Qwen3.5/3.6 capture hooks for the pinned SGLang model implementation."""
from typing import List
from torch import nn
from .base import TargetModelAdapter


class Qwen35TargetAdapter(TargetModelAdapter):
    def configure_backend(self, backend, kwargs):
        if backend != "sglang":
            raise ValueError("Qwen3.5 targets require the SGLang backend in the pinned environment")
        kwargs["attention_backend"] = "fa3"

    def configure_capture(self, model, layer_ids):
        num_layers = len(model.model.layers)
        if hasattr(model, "set_dflash_layers_to_capture"):
            model.set_dflash_layers_to_capture([i for i in layer_ids if i < num_layers - 1])
        else:
            self._set_communicator_capture(model, layer_ids, num_layers)

'''+patch
(target/'adapters/qwen35.py').write_text(qwen)
(target/'adapters/gemma.py').write_text('''"""Gemma text semantics; execution and TP handling remain in the backends."""
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch import nn
from .base import TargetModelAdapter


class GemmaTokenEmbedding(nn.Embedding):
    def forward(self, input):
        embeddings = super().forward(input)
        return embeddings * embeddings.new_tensor(self.embedding_dim ** 0.5)


class GemmaTargetAdapter(TargetModelAdapter):
    native_capture = True
    softcap_fp32 = True

    def configure_backend(self, backend, kwargs):
        if backend == "sglang" and kwargs.get("attention_backend") not in (
            "triton", "fa4", "trtllm_mha", "ascend", "intel_xpu", "intel_amx", "aiter",
        ):
            kwargs["attention_backend"] = "triton"

    def sglang_components(self, model):
        # The native module already scales by sqrt(H); share it unchanged.
        return model.language_model.embed_tokens, model.lm_head

    def new_embedding(self):
        return GemmaTokenEmbedding(
            self.config.vocab_size, self.config.hidden_size,
            padding_idx=getattr(self.config, "pad_token_id", None),
        )

    def create_capture_engine(self, model_path, capture_layer_ids, *, torch_dtype=None, cache_dir=None, trust_remote_code=False, **kwargs):
        from legospec.distributed import get_tp_group
        from legospec.modeling.target.sglang import initialize_collocated_parallel
        from legospec.modeling.target.sglang_gemma4_capture import Gemma4Capture
        if torch_dtype not in (None, torch.bfloat16):
            raise ValueError("Gemma native capture uses bfloat16")
        context = int(kwargs.get("context_length") or 4097)
        args = SimpleNamespace(
            target_model_path=model_path,
            capture_layer_ids=list(capture_layer_ids),
            capture_hidden_size=self.config.hidden_size,
            capture_tp_size=dist.get_world_size(get_tp_group()),
            capture_tp_rank=dist.get_rank(get_tp_group()),
            max_length=context - 1,
            sglang_context_length=context,
            sglang_attention_backend=kwargs.get("attention_backend", "triton"),
            sglang_mem_fraction_static=kwargs.get("mem_fraction_static", 0.4),
            sglang_max_running_requests=kwargs.get("max_running_requests") or 1,
            sglang_max_total_tokens=kwargs.get("max_total_tokens") or context,
        )
        engine = Gemma4Capture(args, initialize_parallel=initialize_collocated_parallel)
        engine.runner.model.requires_grad_(False).eval()
        return engine
''')
p=target/'components.py';s=p.read_text();a=s.index('class _Gemma4TokenEmbedding');b=s.index('class TargetEmbeddingsAndHead',a);s=s[:a]+s[b:];a=s.index('        text_config =',s.index('class TargetEmbeddingsAndHead'));b=s.index('    @staticmethod',a)
s=s[:a]+'''        from legospec.modeling.target.adapters.base import get_model_adapter
        adapter = get_model_adapter(config)
        self.embed_tokens = adapter.new_embedding()
        self.lm_head = FlattenedLMHead(
            adapter.config.hidden_size, adapter.config.vocab_size, bias=False,
            softcap=adapter.softcap,
        )

'''+s[b:]
s+='''

def build_sglang_components(model, adapter, mask_token_id):
    """Share resident modules; vocabulary partitioning is independent of model TP."""
    from legospec.distributed import get_tp_group
    embedding, native_head = adapter.sglang_components(model)
    weight = native_head.weight
    vocab, hidden = adapter.config.vocab_size, adapter.config.hidden_size
    # SGLang's vocab-parallel modules expose org_vocab_size even at TP=1.
    # Replicated native embeddings retain their own forward scaling.
    if hasattr(embedding, "org_vocab_size"):
        embedding = SGLangTPEmbeddingAdapter(embedding, get_tp_group(), mask_token_id)
    if hasattr(native_head, "org_vocab_size"):
        head = SGLangTPLMHeadAdapter(native_head, get_tp_group(), softcap=adapter.softcap)
    else:
        with torch.device("meta"):
            head = FlattenedLMHead(hidden, vocab, bias=False, softcap=adapter.softcap, softcap_fp32=adapter.softcap_fp32)
        head.weight = weight if weight.shape[0] == vocab else nn.Parameter(weight[:vocab], requires_grad=False)
    set_projection_suppress_tokens(head, getattr(model, "suppress_token_ids", []))
    return SharedTargetEmbeddingsAndHead(embedding, head).requires_grad_(False).eval()
'''
p.write_text(s)
# Current cache consumers require real residual layers and explicit teacher state.
p=root/'legospec/data/hidden_cache.py';s=p.read_text().replace('            hidden_states.setdefault(self.final_layer_id, final_hidden)\n','');s=s.replace('''                        item.get(
                            "final_hidden_state", item["hidden_states"][final_layer_id]
                        )''','''                        item["final_hidden_state"]''');p.write_text(s)
