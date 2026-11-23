from pathlib import Path
import ast
root=Path.cwd();target=root/'legospec/modeling/target'
p=target/'sglang_backend/capture.py';s=p.read_text();s=s.replace('        hidden_size,\n','        hidden_size,\n        adapter,\n',1).replace('        self.hidden_size = hidden_size','        self.hidden_size = hidden_size\n        self.adapter = adapter')
s=s.replace('            enable_return_aux_hidden_states=True,\n            speculative_algorithm="EAGLE3",\n            return_aux_hidden_state_layer_ids=self.layer_ids,','            enable_return_hidden_states=True,')
s=s.replace('        self.runner.init_attention_backends()','        self.runner.init_attention_backends()\n        self.adapter.install_capture(self.runner.model, self.layer_ids)')
s=s.replace('''            output = self.runner.forward(forward).logits_output''','''            teacher = {}
            def capture_teacher(module, inputs):
                teacher["hidden"] = inputs[1]
            # SGLang 0.5.21 returns auxiliary context in output.hidden_states.
            # The normalized teacher is the LM-head input, captured explicitly.
            handle = self.runner.model.logits_processor.register_forward_pre_hook(capture_teacher)
            try:
                output = self.runner.forward(forward).logits_output
                final = teacher["hidden"]
            finally:
                handle.remove()''')
s=s.replace('        aux, final = output.hidden_states, output.last_hidden_states','        aux = output.hidden_states');p.write_text(s)
p=target/'sglang_backend/parallel.py';s=p.read_text().replace('        use_xpu_communicator=False,','        use_xpu_communicator=False,\n        use_npu_communicator=False,');p.write_text(s)
# All supported targets now execute the current SGLang runner through one path.
p=target/'sglang.py';s=p.read_text();tree=ast.parse(s);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
def method(name):
 n=methods[name];start=min([n.lineno]+[d.lineno for d in n.decorator_list]);return '\n'.join(s.splitlines()[start-1:n.end_lineno])+'\n'
new='''"""SGLang native target execution shared by online training and offline capture."""
import torch
from legospec.modeling.target.training_target_model import TrainingTargetModel, TargetTrainingOutput


class SGLangTrainingTargetModel(TrainingTargetModel):
    def __init__(self, model_runner, *, adapter, engine):
        super().__init__()
        self.model_runner, self.adapter, self.engine = model_runner, adapter, engine
        self.capture_layer_ids = list(engine.layer_ids)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, capture_layer_ids,
                        adapter, torch_dtype=None, device=None, cache_dir=None,
                        trust_remote_code=False, **kwargs):
        from legospec.modeling.target.sglang_backend.capture import NativeCapture
        engine = NativeCapture(
            pretrained_model_name_or_path, capture_layer_ids=capture_layer_ids,
            hidden_size=adapter.config.hidden_size, adapter=adapter,
            torch_dtype=torch_dtype, cache_dir=cache_dir,
            trust_remote_code=trust_remote_code, **kwargs,
        )
        return cls(engine.runner, adapter=adapter, engine=engine)

    def set_capture_layers(self, layer_ids):
        self.capture_layer_ids = sorted(layer_ids)
        self.engine.layer_ids = self.capture_layer_ids
        self.adapter.install_capture(self.model_runner.model, self.capture_layer_ids)

    def generate_capture_data(self, records):
        return self.engine.generate_capture_data(records)

'''
new+=method('_training_from_packed').replace('def _training_from_packed(', 'def generate_training_data(')+'\n'+method('build_shared_components')+'\n'+method('build_anchor_scorer');p.write_text(new)
# Default decoder final residual is captured before norm; append it to backbone aux.
p=target/'adapters/base.py';s=p.read_text().replace('    native_capture = False\n','').replace('    supports_final_norm_recompute = False','    captures_final_layer = False\n    supports_final_norm_recompute = False')
a=s.index('        final_id = len(model.model.layers)');b=s.index('    def remove_capture',a)
s=s[:a]+'''        if self.captures_final_layer:
            return
        final_id = len(model.model.layers) - 1
        if final_id not in layer_ids:
            return
        captured = {}
        def capture_raw(module, args, kwargs):
            hidden = args[0] if args else kwargs["x"]
            residual = args[1] if len(args) > 1 else kwargs.get("residual")
            captured["raw"] = hidden.clone() if residual is None else hidden + residual
        self._capture_handles.append(
            self.final_norm(model.model).register_forward_pre_hook(capture_raw, with_kwargs=True)
        )
        def append_final(module, args, output):
            if isinstance(output, tuple):
                normalized, aux = output
                return normalized, [*aux, captured.pop("raw")]
            return output, [captured.pop("raw")]
        self._capture_handles.append(model.model.register_forward_hook(append_final))

'''+s[b:];p.write_text(s)
p=target/'adapters/gemma.py';s=p.read_text().replace('    native_capture = True','    captures_final_layer = True');p.write_text(s)
p=target/'adapters/qwen35.py';p.write_text('''"""Qwen3.5/3.6 native SGLang layer-output capture."""
from .base import TargetModelAdapter


class Qwen35TargetAdapter(TargetModelAdapter):
    captures_final_layer = True

    def configure_backend(self, backend, kwargs):
        if backend != "sglang":
            raise ValueError("Qwen3.5 targets require the SGLang backend")
        kwargs["attention_backend"] = "fa3"

    def configure_capture(self, model, layer_ids):
        model.set_dflash_layers_to_capture(layer_ids)
''')
# Pin the tested new stack; CUDA kernels are supplied by sglang-kernel now.
p=root/'pyproject.toml';s=p.read_text().replace('sglang==0.5.9','sglang==0.5.21').replace('torch==2.9.1','torch==2.13.0').replace('torchaudio==2.9.1','torchaudio==2.11.0').replace('torchvision==0.24.1','torchvision==0.28.0').replace('transformers==4.57.1','transformers==5.12.1').replace('tilelang==0.1.15','tilelang==0.1.12').replace('sgl-kernel==0.3.21','sglang-kernel==0.4.7').replace('Torch 2.9 Triton pin','PyTorch 2.13 Triton pin')
a=s.index('# The upstream CUDA 13 index');b=s.index('torch = [',a);s=s[:a]+s[b:];p.write_text(s)
