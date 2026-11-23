from pathlib import Path
root=Path.cwd()
p=root/'legospec/modeling/target/adapters/base.py';s=p.read_text().replace('from torch import nn','from types import SimpleNamespace\nfrom torch import nn')
s=s.replace('    config = getattr(config, "text_config", config)\n    model_type =', '''    if isinstance(config, dict):
        outer_type = config.get("model_type", "")
        values = dict(config.get("text_config", config))
        values.setdefault("model_type", outer_type)
        config = SimpleNamespace(**values)
        if outer_type in {"gemma4_unified", "qwen3_5", "qwen3_5_moe"}:
            config.target_source_model_type = outer_type
    config = getattr(config, "text_config", config)
    model_type =''')
s=s.replace('if model_type == "gemma4_unified_text":','if model_type in {"gemma4_unified", "gemma4_unified_text"}:');p.write_text(s)
p=root/'legospec/data/hidden_cache.py';s=p.read_text().replace('from legospec.modeling.config_utils import load_text_model_config','from legospec.modeling.config_utils import get_pretrained_config_dict').replace('            config = load_text_model_config(manifest["target_model"])','            adapter = get_model_adapter(get_pretrained_config_dict(manifest["target_model"]))\n            config = adapter.config').replace('self._final_norm = get_model_adapter(config).load_final_norm(manifest["target_model"])','self._final_norm = adapter.load_final_norm(manifest["target_model"])');p.write_text(s)
