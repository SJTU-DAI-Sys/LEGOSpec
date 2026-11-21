---
base_model: google/gemma-4-12B-it
tags:
- speculative-decoding
- dflash2
- legospec
---

# LEGOSpec DFlash2 for Gemma 4 12B IT

This is the final LEGOSpec DFlash2 draft checkpoint for
`google/gemma-4-12B-it`, exported after 65,361 optimizer steps.
The repository contains only `config.json`, `model.safetensors`, and this
README. Model implementation and inference are provided separately by LEGOSpec.

## Architecture

- Native class: `litespec.modeling.draft.dflash2.DFlash2DraftModel`.
- 5 full-attention GQA layers; hidden size 3840; intermediate size 15360.
- 16 query heads, 8 KV heads, head dimension 256.
- Grouped convolution: kernel size 2, group size 16.
- Candidate selector: rank 256, top-k 16.
- Target decoder layers: [1, 5, 17, 29, 41] (zero-based decoder indices).
- Block size 8: one anchor plus 7 draft tokens; verification window 8.
- Same-position alignment.
- Original proportional RoPE: theta 1,000,000, partial rotary factor 0.25;
  no YaRN.
- BF16 weights; vocabulary size 262144.

`model_type: qwen3` identifies the Qwen3Config used by the custom draft
implementation. It does not identify the target model. The checkpoint uses
`dflash2_config`; do not replace it with another implementation's
`dflash_config`.

## Loading

Install a compatible LEGOSpec runtime exposing the native class above first.
The validated source is the original DFlash2 export accompanying this
checkpoint, not a Composable conversion. This weights repository does not
supply that runtime or remote Python code.

The validated environment used PyTorch 2.14.0+cu130 and Transformers
5.20.2. With the compatible LEGOSpec source available on
PYTHONPATH:

```python
import torch
from transformers import Qwen3Config
from litespec.modeling.draft.dflash2 import DFlash2DraftModel

checkpoint = "/path/to/downloaded/model-repository"
config = Qwen3Config.from_pretrained(checkpoint, local_files_only=True)
draft, loading_info = DFlash2DraftModel.from_pretrained(
    checkpoint,
    config=config,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    local_files_only=True,
    output_loading_info=True,
)
for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
    assert not loading_info.get(key), loading_info
draft = draft.eval()
```

This loads the draft only. End-to-end speculative decoding additionally
requires the target model, its tokenizer, target hidden-state capture, and
LEGOSpec's native Gemma verifier. Preserve Gemma's output-head softcapping
and suppressed-token handling. The tokenizer and target weights come from
`google/gemma-4-12B-it` and are not included here.

Use the explicit native class above: this release intentionally omits
`auto_map` because it contains no Python implementation. Loading it as a
standard Qwen3 causal language model or enabling `trust_remote_code` does
not provide the custom draft implementation.

## Provenance and validation

The checkpoint was trained on QZ and exported for inference. It is not a full
training-resume checkpoint. The original optimizer/scheduler training state
is not part of this release.

Packaging validation verified the weight SHA256 and successfully loaded the
release on CPU with no missing, unexpected, or mismatched parameters.
All original configuration values are preserved except removal of the
`auto_map` entry pointing to Python files not shipped in this repository.
No new generation benchmark was run as part of packaging.

Weight SHA256:
`adb68df5026e71d00f11fb650222677a514258b428cf0cc292d3cbc622f6017f`

Original accompanying runtime archive SHA256 (source identity):
`2bde39f158c870ccb7b68f5befd48ba7c37293150dfc663812a1fd40c6a7727d`
