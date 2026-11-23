from legospec.modeling.target.custom_backend.gpt_oss import GptOssForCausalLM
from legospec.modeling.target.custom_backend.llama import LlamaForCausalLM
from legospec.modeling.target.custom_backend.llama4 import Llama4ForCausalLM
from legospec.modeling.target.custom_backend.phi3 import Phi3ForCausalLM
from legospec.modeling.target.custom_backend.qwen2 import Qwen2ForCausalLM
from legospec.modeling.target.custom_backend.qwen3 import Qwen3ForCausalLM
from legospec.modeling.target.custom_backend.qwen3_moe import Qwen3MoeForCausalLM

__all__ = [
    "GptOssForCausalLM",
    "LlamaForCausalLM",
    "Llama4ForCausalLM",
    "Phi3ForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
]
