# Using a LEGOSpec draft

A draft combines selected target hidden states, runs its backbone, and proposes tokens. The matching target provides the shared token embedding and output head.

## Load a checkpoint

```python
import torch
from legospec import LegoSpecConfig, LegoSpecModel

path = "/path/to/draft/current"
config = LegoSpecConfig.from_pretrained(path)
draft = LegoSpecModel.from_pretrained(
    path, config=config, dtype=torch.bfloat16
).cuda().eval()
```

Project-format checkpoints contain configuration and weights. Loading through these classes uses the installed LEGOSpec implementation directly.

## Generate with a Qwen3 target

This example is for a Qwen3 causal language model with decoder layers at `target.model.layers`, and a draft trained for that exact target. It accepts one nonempty, unpadded prompt per call. Other target architectures may need different feature capture or embedding/head semantics; this helper is not a universal wrapper for Gemma or multimodal targets.

After loading `draft` above:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

target_path = "/path/to/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(target_path)
target = AutoModelForCausalLM.from_pretrained(
    target_path, dtype=torch.bfloat16
).cuda().eval()

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain speculative decoding briefly."}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
input_ids = tokenizer(
    text, return_tensors="pt", add_special_tokens=False
).input_ids.cuda()
eos = target.generation_config.eos_token_id
stop_ids = [] if eos is None else eos if isinstance(eos, list) else [eos]

with torch.inference_mode():
    output_ids = draft.spec_generate(
        target,
        input_ids,
        max_new_tokens=128,
        stop_token_ids=stop_ids,
        temperature=0.0,
        top_p=1.0,
        seed=42,
    )

print(tokenizer.decode(
    output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
))
```

`spec_generate` returns the prompt followed by generated tokens. Set a positive temperature for stochastic sampling. The verifier applies probabilistic acceptance and residual sampling. `draft.get_last_decode_stats()` returns the most recent decode statistics.

## Work with features directly

Import `DraftInputs`, `SharedOps`, and `DraftState` from `legospec.modeling.draft`.

- `target_hidden_states`: selected raw decoder features, shaped `[N, S, T*D]` or `[N, S, T, D]`.
- `draft_embeddings`: target token embeddings for the anchor and draft slots, shaped `[N, A*Q, D]`.
- `anchor_positions` and `block_keep_mask`: absolute anchor positions and validity, shaped `[N, A]`.
- `mode="features"`: return draft features without a vocabulary projection.
- `mode="teacher_forcing"`: score supervised slots; selector recipes also need predecessor IDs.
- `mode="sample"`: propose tokens using the shared output head and required anchor IDs.

Here, N is batch size, S is history length, T is the number of selected target layers, A is the number of anchors, and Q is block size. A block contains an anchor plus Q−1 prediction slots. Context positions are absolute; cached calls supply newly confirmed history.

The hidden-feedback recipe uses `forward_hidden_feedback` and `extend_hidden_feedback` rather than the block-input interface.

## Configure a model

See the [recipe guide](../../../configs/README.md) and [model structure](../DESIGN.md). Context, Backbone, and Proposal form the forward path; Objective supplies training losses separately.
