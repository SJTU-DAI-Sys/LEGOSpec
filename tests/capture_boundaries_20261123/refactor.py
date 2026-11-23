from pathlib import Path
import ast
root=Path.cwd();target=root/'legospec/modeling/target';backend=target/'sglang_backend'
# The native engine receives model/backend arguments, never a collector/trainer Namespace.
p=target/'sglang_gemma4_capture.py';s=p.read_text().replace('Packed text-only Gemma 4 Unified prefill using SGLang\'s native aux capture.','Packed prefill using SGLang native aux capture and a LEGOSpec TP mesh.')
s=s.replace('import torch\n','import torch\nimport torch.distributed as dist\nfrom legospec.distributed import get_tp_group\nfrom .parallel import initialize_collocated_parallel\n')
s=s.replace('SpawnRanks, get_device, get_schedule, publish','SpawnRanks, get_schedule, publish')
a=s.index('class Gemma4Capture:');b=s.index('        gpu_id = torch.cuda.current_device()',a)
s=s[:a]+'''class NativeCapture:
    def __init__(
        self, model_path, *, capture_layer_ids, hidden_size,
        torch_dtype=None, cache_dir=None, trust_remote_code=False, **backend_kwargs,
    ):
        if torch_dtype not in (None, torch.bfloat16):
            raise ValueError("Native SGLang capture uses bfloat16")
        self.layer_ids = list(capture_layer_ids)
        self.hidden_size = hidden_size
        options = dict(backend_kwargs)
        options["context_length"] = options.get("context_length") or 4097
        options["max_running_requests"] = options.get("max_running_requests") or 1
        options["max_total_tokens"] = options.get("max_total_tokens") or options["context_length"]
        options.setdefault("mem_fraction_static", 0.4)
        options.update(
            disable_custom_all_reduce=True,
            disable_cuda_graph=True,
            disable_prefill_cuda_graph=True,
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            enable_return_aux_hidden_states=True,
            speculative_algorithm="EAGLE3",
            return_aux_hidden_state_layer_ids=self.layer_ids,
        )
        server_args = ServerArgs(
            model_path=model_path, dtype="bfloat16",
            tp_size=dist.get_world_size(get_tp_group()),
            trust_remote_code=trust_remote_code, download_dir=cache_dir,
            **options,
        )
'''+s[b:]
s=s.replace('world_rank=getattr(args, "capture_tp_rank", 0), gpu_id=gpu_id','world_rank=dist.get_rank(get_tp_group()), gpu_id=gpu_id')
a=s.index('        if initialize_parallel is None:');b=s.index('        bootstrap.init_layer_runtime',a)
s=s[:a]+'''        # Both online training and offline collection own a LEGOSpec WORLD.
        initialize_collocated_parallel(server_args=server_args, local_rank=gpu_id)
'''+s[b:]
s=s.replace('mem_fraction_static=args.sglang_mem_fraction_static','mem_fraction_static=server_args.mem_fraction_static')
s=s.replace('        self.runner.alloc_memory_pool()','        self.runner.model.requires_grad_(False).eval()\n        self.runner.alloc_memory_pool()')
s=s.replace('Gemma 4 did not return','Native SGLang capture did not return')
(backend/'capture.py').write_text(s);p.unlink()
# Backend lifecycle is separate from model adapters, and only imported by native execution.
p=target/'sglang.py';s=p.read_text();a=s.index('\ndef initialize_collocated_parallel(');parallel=s[a:].replace('legospec_gemma4_teacher','legospec_target');s=s[:a]
(backend/'parallel.py').write_text('''"""TP-scoped native SGLang groups sharing a LEGOSpec distributed WORLD."""
import torch.distributed as dist

'''+parallel.lstrip())
s=s.replace('            engine = adapter.create_capture_engine(','            from legospec.modeling.target.sglang_backend.capture import NativeCapture\n\n            engine = NativeCapture(').replace('                capture_layer_ids,\n                torch_dtype=', '                capture_layer_ids=capture_layer_ids,\n                hidden_size=adapter.config.hidden_size,\n                torch_dtype=')
s=s.replace('''        from legospec.modeling.target.sglang_backend import (
            SGLangRunner,
            wrap_eagle3_logits_processors_in_module,
        )''','''        from legospec.modeling.target.sglang_backend.model_runner import SGLangRunner
        from legospec.modeling.target.sglang_backend.utils import wrap_eagle3_logits_processors_in_module''')
p.write_text(s)
# No eager runner import when importing the independent native capture implementation.
(backend/'__init__.py').unlink()
# Model-specific final norm loading/math belongs to the adapter, not cache IO.
p=target/'adapters/gemma.py';s=p.read_text();s=s[:s.index('    def create_capture_engine(')]
s=s.replace('from types import SimpleNamespace\n','').replace('import torch.distributed as dist\n','')
s=s.replace('    native_capture = True','    native_capture = True\n    supports_final_norm_recompute = True')
s+='''    def load_final_norm(self, model_path):
        from pathlib import Path
        from safetensors import safe_open

        with safe_open(Path(model_path) / "model.safetensors", framework="pt", device="cpu") as checkpoint:
            weight = checkpoint.get_tensor("model.language_model.norm.weight").to(dtype=torch.float32, copy=True)
        return GemmaFinalNorm(weight, self.config.rms_norm_eps)


class GemmaFinalNorm(nn.Module):
    """Reconstruct normalized teacher features from cached decoder residuals."""
    def __init__(self, weight, eps):
        super().__init__()
        self.register_buffer("weight", weight)
        self.eps = eps

    def forward(self, residual):
        return nn.functional.rms_norm(
            residual.float(), (self.weight.numel(),), self.weight, self.eps,
        ).to(residual.dtype)
'''
p.write_text(s)
p=target/'adapters/base.py';s=p.read_text().replace('    native_capture = False','    native_capture = False\n    supports_final_norm_recompute = False');s=s.replace('    def draft_config_values(self):','    def load_final_norm(self, model_path):\n        raise ValueError("This target adapter does not support cached final norm reconstruction")\n\n    def draft_config_values(self):');p.write_text(s)
# Offline caller uses the same loader, backend options and TP mesh as online training.
p=root/'legospec/data/capture_engine.py';s=p.read_text().replace('    from datetime import timedelta\n','').replace('    from transformers import AutoConfig, AutoTokenizer','    from transformers import AutoTokenizer\n    from legospec.modeling.config_utils import load_text_model_config\n    from legospec.modeling.target.adapters.base import get_model_adapter')
a=s.index('    config = AutoConfig.from_pretrained');b=s.index('    root = Path(args.output_path)',a)
s=s[:a]+'''    text_config = load_text_model_config(
        args.target_model_path, cache_dir=args.model_download_dir,
        trust_remote_code=args.trust_remote_code,
    )
    adapter = get_model_adapter(text_config)
    if args.recompute_gemma4_final_norm and not adapter.supports_final_norm_recompute:
        raise ValueError("The selected target adapter does not support final norm recomputation")
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    rank = dist.get_rank(get_dp_group())
    world = dist.get_world_size(get_dp_group())
'''+s[b:]
a=s.index('        if gemma4:',s.index('def run_generation'));b=s.index('        raw = load_dataset',a)
s=s[:a]+'''        kwargs = (
            SGLangBackendArgs.from_args(args).to_kwargs()
            if args.target_model_backend == "sglang" else {}
        )
        if args.target_model_backend == "sglang":
            kwargs["context_length"] = kwargs.get("context_length") or args.max_length + 1
            kwargs["max_running_requests"] = kwargs.get("max_running_requests") or args.batch_size
            kwargs["max_total_tokens"] = kwargs.get("max_total_tokens") or args.batch_size * args.max_length
        target = get_training_target_model(
            args.target_model_path,
            backend=args.target_model_backend,
            capture_layer_ids=args.capture_layer_ids,
            torch_dtype=torch.bfloat16,
            device="cuda" if args.target_model_backend == "hf" else None,
            cache_dir=args.model_download_dir,
            trust_remote_code=args.trust_remote_code,
            **kwargs,
        )
'''+s[b:]
s=s.replace('check_gemma4_capture_space','check_capture_space').replace('per Gemma 4 cache row','per cache row').replace('Gemma 4 pending cache','Pending capture cache').replace('Gemma 4 capture needs','Capture needs')
p.write_text(s)
p=root/'legospec/data/hidden_cache.py';s=p.read_text();a=s.index('def _load_gemma4_final_norm(');b=s.index('def _sha256(',a);s=s[:a]+s[b:]
a=s.index('            self._gemma4_final_norm = _load_gemma4_final_norm(');b=s.index('        self.max_length =',a)
s=s[:a]+'''            from legospec.modeling.config_utils import load_text_model_config
            from legospec.modeling.target.adapters.base import get_model_adapter

            config = load_text_model_config(manifest["target_model"])
            if config.hidden_size != self.hidden_size or config.num_hidden_layers != self.final_layer_id + 1:
                raise HiddenCacheError("Target checkpoint does not match cached feature dimensions")
            self._final_norm = get_model_adapter(config).load_final_norm(manifest["target_model"])
'''+s[b:]
s=s.replace('self._gemma4_final_norm','self._final_norm')
a=s.index('            (weight, eps) = self._final_norm');b=s.index('        if final_hidden.shape',a)
s=s[:a]+'''            final_hidden = self._final_norm(selected[self.final_layer_id])
'''+s[b:];p.write_text(s)
for p in (root/'tests/unit').glob('*.py'):
 s=p.read_text().replace('check_gemma4_capture_space','check_capture_space').replace('dataset._gemma4_final_norm','dataset._final_norm');p.write_text(s)
