from pathlib import Path
import ast, textwrap
root=Path.cwd(); target=root/'legospec/modeling/target'; train=root/'legospec/training/target'
old=(target/'training_target_model.py').read_text(); features=(train/'features.py').read_text(); gemma=(target/'sglang_gemma4_training.py').read_text()
def segment(source,name):
 tree=ast.parse(source)
 node=next(n for n in tree.body if getattr(n,'name',None)==name)
 start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
 return '\n'.join(source.splitlines()[start-1:node.end_lineno])+'\n'
def method(source,cls,name):
 node=next(n for n in ast.parse(source).body if getattr(n,'name',None)==cls)
 n=next(n for n in node.body if getattr(n,'name',None)==name)
 start=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])
 return '\n'.join(source.splitlines()[start-1:n.end_lineno])+'\n'
(target/'adapters').mkdir(exist_ok=True)
(target/'components.py').write_text((target/'target_utils.py').read_text()); (target/'target_utils.py').unlink()
# Preserve real backend implementations, split optional backend dependencies.
common='''"""Target feature contract and backend construction, independent of training."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import torch

'''+segment(old,'TargetTrainingOutput')+'\n'+segment(old,'TrainingTargetModel')
common=common.replace('hidden_states: Union[\n        torch.Tensor, Tuple[torch.Tensor, ...], Dict[int, torch.Tensor]\n    ]','hidden_states: dict[int, torch.Tensor]')
common=common.replace('optional in Stage 1','optional teacher projection')
common=common.replace('self.capture_layer_ids = None','self.capture_layer_ids = None\n        self._components = None')
common+='''
    @abstractmethod
    def build_shared_components(self, mask_token_id):
        """Frozen views of resident embedding/head, retaining input gradients."""

    @abstractmethod
    def build_anchor_scorer(self):
        """Create the backend's sampled-anchor service."""


def get_training_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype=None,
    device=None,
    cache_dir=None,
    capture_layer_ids=None,
    **kwargs,
) -> TrainingTargetModel:
    from legospec.modeling.config_utils import load_text_model_config
    from legospec.modeling.target.adapters.base import get_model_adapter

    config = load_text_model_config(
        pretrained_model_name_or_path, cache_dir=cache_dir,
        trust_remote_code=kwargs.get("trust_remote_code", False),
    )
    adapter = get_model_adapter(config)
    adapter.configure_backend(backend, kwargs)
    if backend == "sglang":
        from legospec.modeling.target.sglang import SGLangTrainingTargetModel
        backend_class = SGLangTrainingTargetModel
    elif backend == "hf":
        from legospec.modeling.target.hf import HFTrainingTargetModel
        backend_class = HFTrainingTargetModel
    else:
        raise ValueError(f"Invalid backend: {backend}")
    return backend_class.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype, device=device, cache_dir=cache_dir,
        capture_layer_ids=capture_layer_ids, adapter=adapter, **kwargs,
    )
'''
(target/'training_target_model.py').write_text(common)
imports='''from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from legospec.modeling.target.training_target_model import TrainingTargetModel, TargetTrainingOutput
from legospec.modeling.target.adapters.base import TargetModelAdapter
'''
hf=imports+'from transformers import AutoModelForCausalLM\n\n'+segment(old,'HFTrainingTargetModel')
hf=hf.replace('def __init__(self, model: nn.Module):','def __init__(self, model: nn.Module, *, adapter=None, capture_layer_ids=None):').replace('self.model = model','self.model = model\n        self.adapter = adapter or TargetModelAdapter(getattr(model, "config", None))\n        self.capture_layer_ids = capture_layer_ids',1)
hf=hf.replace('        **kwargs,\n    ) -> "HFTrainingTargetModel":','        capture_layer_ids=None,\n        adapter=None,\n        **kwargs,\n    ) -> "HFTrainingTargetModel":')
hf=hf.replace('return cls(target_model)','return cls(target_model, adapter=adapter, capture_layer_ids=capture_layer_ids)')
a=hf.index('        # Multimodal checkpoints');b=hf.index('        pre_final = {}',a)
hf=hf[:a]+'''        backbone = self.adapter.hf_backbone(self.model)
        forward_model = self.model if return_logits else backbone
        norm = self.adapter.final_norm(backbone)
'''+hf[b:]
a=hf.index('        capture_ids = list'); b=hf.index('\n        return TargetTrainingOutput',a)
hf=hf[:a]+'''        capture_ids = self.capture_layer_ids
        if capture_ids is None:
            capture_ids = range(len(all_hidden_states))
        hidden_states = {layer_id: all_hidden_states[layer_id] for layer_id in capture_ids}
'''+hf[b:]
hf+='''
    def build_shared_components(self, mask_token_id):
        from legospec.modeling.target.components import FlattenedLMHead, SharedTargetEmbeddingsAndHead
        if self._components is None:
            model = self.model.requires_grad_(False).eval()
            head = model.get_output_embeddings()
            if isinstance(head, nn.Linear):
                with torch.device("meta"):
                    shared_head = FlattenedLMHead(
                        head.in_features, head.out_features, bias=head.bias is not None,
                        softcap=self.adapter.softcap,
                    )
                shared_head.weight, shared_head.bias = head.weight, head.bias
                head = shared_head
            self._components = SharedTargetEmbeddingsAndHead(
                model.get_input_embeddings(), head,
            ).requires_grad_(False).eval()
        return self._components

    def build_anchor_scorer(self):
        from legospec.modeling.target.anchor_opd import Qwen3AnchorScorer
        return Qwen3AnchorScorer(self.model)
'''
(target/'hf.py').write_text(hf)
sg=imports+'import torch.distributed as dist\nfrom legospec.distributed import get_tp_group\n\n'+segment(old,'SGLangTrainingTargetModel')
sg=sg.replace('def __init__(self, model_runner):','def __init__(self, model_runner, *, adapter=None, engine=None):').replace('self.model_runner = model_runner','self.model_runner = model_runner\n        self.adapter = adapter or TargetModelAdapter()\n        self.engine = engine',1)
sg=sg.replace('        **kwargs,\n    ) -> "SGLangTrainingTargetModel":','        capture_layer_ids=None,\n        adapter=None,\n        **kwargs,\n    ) -> "SGLangTrainingTargetModel":')
a=sg.index('        from sglang.srt.configs.model_config')
sg=sg[:a]+'''        adapter = adapter or TargetModelAdapter()
        if adapter.native_capture:
            engine = adapter.create_capture_engine(
                pretrained_model_name_or_path, capture_layer_ids,
                torch_dtype=torch_dtype, cache_dir=cache_dir,
                trust_remote_code=trust_remote_code, **kwargs,
            )
            instance = cls(engine.runner, adapter=adapter, engine=engine)
            instance.capture_layer_ids = list(capture_layer_ids)
            return instance
'''+sg[a:]
sg=sg.replace('instance = cls(model_runner)','instance = cls(model_runner, adapter=adapter)').replace('instance.set_capture_layers([])','instance.set_capture_layers(capture_layer_ids)')
a=sg.index('    def set_capture_layers'); b=sg.index('    @staticmethod\n    def _unpack_runner_output',a)
sg=sg[:a]+'''    def set_capture_layers(self, layer_ids):
        if self.engine is not None:
            if list(layer_ids) != self.capture_layer_ids:
                raise ValueError("Capture layers must be supplied before model initialization")
            return
        if layer_ids is None:
            layer_ids = list(range(self.model_runner.model_config.num_hidden_layers))
        super().set_capture_layers(list(layer_ids))
        self.adapter.install_capture(self.model_runner.model, self.capture_layer_ids)

'''+sg[b:]
# All selected residual streams now live in aux; normalized teacher remains separate.
a=sg.index('    @staticmethod\n    def _split_aux');b=sg.index('    @torch.no_grad\n    def _extend',a)
sg=sg[:a]+sg[b:]
a=sg.index('        hidden_size = self.model_runner.model_config.hidden_size',sg.index('    def generate_training_data')); b=sg.index('        return TargetTrainingOutput',a)
sg=sg[:a]+'''        hidden_size = self.model_runner.model_config.hidden_size
        viewed = batched_aux.reshape(*batched_aux.shape[:2], len(self.capture_layer_ids), hidden_size)
        hidden_states = {layer: viewed[:, :, column] for column, layer in enumerate(self.capture_layer_ids)}
'''+sg[b:]
# return normalized teacher explicitly
pos=sg.index('        return TargetTrainingOutput',sg.index('    def generate_training_data'))
end=sg.index('        )',pos)
sg=sg[:end]+ '            final_hidden_state=batched_last,\n'+sg[end:]
# Native packed engine uses same public training model, no Gemma training subclass.
a=sg.index('        from sglang.srt.managers.schedule_batch import Req',sg.index('    def generate_training_data'))
sg=sg[:a]+'''        if self.engine is not None:
            return self._training_from_packed(input_ids, attention_mask, loss_mask, return_logits)
'''+sg[a:]
a=sg.index('        from types import SimpleNamespace',sg.index('    def generate_capture_data'))
sg=sg[:a]+'''        if self.engine is not None:
            return self.engine.generate_capture_data(records)
'''+sg[a:]
a=sg.index('        final_id =',sg.index('    def generate_capture_data'));b=sg.index('        offsets = [0]',a)
sg=sg[:a]+'''        viewed = aux.view(sum(lengths), len(self.capture_layer_ids), hidden_size)
        layers = {layer: viewed[:, column, :] for column, layer in enumerate(self.capture_layer_ids)}
'''+sg[b:]
sg=sg.replace('            hidden_states=layers,\n            packed=True,','            hidden_states=layers,\n            final_hidden_state=last,\n            packed=True,')
packed=method(gemma,'Gemma4SGLangTrainingTargetModel','generate_training_data').replace('def generate_training_data(', 'def _training_from_packed(')
sg+='\n'+packed
sg+='''
    def build_shared_components(self, mask_token_id):
        from legospec.modeling.target.components import build_sglang_components
        if not 0 <= mask_token_id < self.adapter.config.vocab_size:
            raise ValueError("MASK token is outside the target vocabulary")
        if self._components is None:
            self._components = build_sglang_components(
                self.model_runner.model, self.adapter, mask_token_id,
            )
        return self._components

    def build_anchor_scorer(self):
        from legospec.modeling.target.sglang_anchor_opd import SGLangAnchorScorer
        return SGLangAnchorScorer(self)
'''
(target/'sglang.py').write_text(sg)
# Training setup retains only training decisions.
setup='''"""Resolve training target requirements, capacity and tokenizer/mask policy."""
from typing import Optional
import torch
from transformers import AutoTokenizer
from legospec.args import SGLangBackendArgs
from legospec.distributed import print_on_rank0
from legospec.modeling.config_utils import load_text_model_config
from legospec.modeling.target.components import TargetEmbeddingsAndHead
from legospec.modeling.target.training_target_model import get_training_target_model

'''
for name in ['required_hidden_layer_ids','build_target_model','build_target_and_components','_select_mask_token_id','resolve_tokenizer_and_components']:
 setup+='\n'+segment(features,name)
a=setup.index('    source_model_type =');b=setup.index('    capture_layer_ids =',a);setup=setup[:a]+setup[b:]
a=setup.index('        if is_qwen35_model_type');b=setup.index('        if backend_kwargs["max_running_requests"]',a)
setup=setup[:a]+'''        backend_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
        backend_kwargs["context_length"] = backend_kwargs.get("context_length") or args.max_length + 1
'''+setup[b:]
setup=setup.replace('        backend=args.target_model_backend,','        backend=args.target_model_backend,\n        capture_layer_ids=sorted(capture_layer_ids),')
setup=setup.replace('    target.set_capture_layers(sorted(capture_layer_ids))\n','')
a=setup.index('    if not standalone_components and hasattr'); b=setup.index('    else:\n        components = TargetEmbeddingsAndHead',a)
setup=setup[:a]+'''    if not standalone_components:
        components = target.build_shared_components(mask_token_id)
'''+setup[b:]
(train/'setup.py').write_text(setup)
(train/'features.py').write_text('''"""Select draft context and normalized teacher features from target outputs."""
import torch


def hidden_states_to_cuda(hidden_states):
    if isinstance(hidden_states, dict):
        return {key: value.cuda(non_blocking=True) for key, value in hidden_states.items()}
    return hidden_states.cuda(non_blocking=True)


def concatenate_target_hidden(hidden_states: dict[int, torch.Tensor], target_layer_ids: list[int]):
    return torch.cat([hidden_states[layer] for layer in target_layer_ids], dim=-1)


def select_target_features(hidden_states, final_hidden_state, target_layer_ids, require_last):
    context = concatenate_target_hidden(hidden_states, target_layer_ids)
    if require_last and final_hidden_state is None:
        raise ValueError("Teacher supervision requires explicit post-final-norm hidden states")
    return context, final_hidden_state if require_last else None


def prepare_target_features(data, target_output, draft, require_last: bool):
    hidden_states = data["hidden_states"] if target_output is None else target_output.hidden_states
    final_hidden = data.get("final_hidden_state") if target_output is None else target_output.final_hidden_state
    context, teacher = select_target_features(
        hidden_states, final_hidden, draft.target_layer_ids, require_last,
    )
    return hidden_states_to_cuda(context), hidden_states_to_cuda(teacher) if teacher is not None else None
''')
# Move execution lifecycle helper out of model-specific file.
sgp=target/'sglang.py';sgp.write_text(sgp.read_text()+'\n\n'+segment(gemma,'initialize_collocated_parallel'))
# Save extracted patch for next stage.
(root/'tests/target_boundaries_20261122/communicator.txt').write_text(method(old,'SGLangTrainingTargetModel','_set_communicator_capture'))
(target/'sglang_gemma4_training.py').unlink()
# Update imports directly; no compatibility modules.
for directory in ['legospec','tests/unit','scripts']:
 for p in (root/directory).rglob('*.py'):
  s=p.read_text();s=s.replace('legospec.modeling.target.target_utils','legospec.modeling.target.components')
  s=s.replace('from legospec.modeling.target.training_target_model import HFTrainingTargetModel','from legospec.modeling.target.hf import HFTrainingTargetModel').replace('from legospec.modeling.target.training_target_model import SGLangTrainingTargetModel','from legospec.modeling.target.sglang import SGLangTrainingTargetModel')
  if p.name in ['setup.py','loader.py'] and 'training/target' not in str(p):
   s=s.replace('from legospec.training.target.features import','from legospec.training.target.setup import')
  p.write_text(s)
p=target/'__init__.py';s=p.read_text().replace('    HFTrainingTargetModel,\n','').replace('    SGLangTrainingTargetModel,\n','');s+='\nfrom .hf import HFTrainingTargetModel\nfrom .sglang import SGLangTrainingTargetModel\n';p.write_text(s)
p=root/'legospec/training/runtime/setup.py';s=p.read_text();a=s.index('        if session.args.target_model_backend == "sglang":',s.index('def make_online'));b=s.index('    return TrainingStep',a);s=s[:a]+'        session.anchor_scorer = session.target.build_anchor_scorer()\n'+s[b:];p.write_text(s)
p=train/'streaming.py';s=p.read_text();a=s.index('from legospec.training.target.features import');b=s.index('\n)',a)+2;s=s[:a]+'from legospec.training.target.features import select_target_features'+s[b:];a=s.index('    target_hidden_states = target_output.hidden_states',s.index('def _target_packet'));b=s.index('    return TargetFeaturePacket',a);s=s[:a]+'''    target_context_hidden, target_last_hidden = select_target_features(
        target_output.hidden_states, target_output.final_hidden_state,
        metadata["target_layer_ids"], metadata["include_target_last_hidden"],
    )
'''+s[b:];p.write_text(s)
p=root/'legospec/data/capture_engine.py';s=p.read_text().replace('                backend=args.target_model_backend,','                backend=args.target_model_backend,\n                capture_layer_ids=args.capture_layer_ids,').replace('            target.set_capture_layers(args.capture_layer_ids)\n','');p.write_text(s)
