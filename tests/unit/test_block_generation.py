"""All Proposal heads use the same verifier and generation accounting."""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from test_component_boundaries import tiny_model
from legospec.inference.generate import generate


@pytest.mark.parametrize(
    "proposal", ["parallel", "rnn", "sequential_rnn", "residual_selector"]
)
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_block_generation_reproducible_and_budgeted(proposal, temperature):
    torch.manual_seed(42)
    draft = tiny_model(proposal)
    draft.mask_token_id = 0
    target = Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=31,
            eos_token_id=None,
            pad_token_id=0,
            _attn_implementation="sdpa",
        )
    ).eval()
    prompt = torch.tensor([[1, 2, 3, 4]])
    first = generate(target, draft, prompt, temperature=temperature, max_new_tokens=11)
    assert first == generate(
        target, draft, prompt, temperature=temperature, max_new_tokens=11
    )
    assert len(first["generated_ids"]) == 11
    assert (
        first["terminal_tokens_outside_cycles"]
        + sum(c["committed_tokens"] for c in first["cycles"])
        == 11
    )
    assert all(0 <= c["accepted_draft_tokens"] <= 3 for c in first["cycles"])
    if temperature == 0:
        expected = target.generate(
            prompt, max_new_tokens=11, do_sample=False, pad_token_id=0
        )
        assert first["generated_ids"] == expected[0, 4:].tolist()
    assert generate(target, draft, prompt, max_new_tokens=0)["generated_ids"] == []
