from __future__ import annotations

"""Shared markov primitives; independent of draft algorithm implementations.
"""
from typing import Optional, Tuple

import torch
from torch import nn

from legospec.modeling.draft.proposal.sampling import sample


def _sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    return sample(logits, temperature, greedy_threshold=1e-5, promote_logits=False)


class AcceptRatePredictor(nn.Module):
    """Predict target/draft distribution acceptance probability per draft step."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


class VanillaMarkovHead(nn.Module):
    """Low-rank previous-token logit bias used by DSpark."""

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        if self.markov_rank <= 0:
            raise ValueError(f"markov_rank must be > 0, got {self.markov_rank}")
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def reset_draft_parameters(self):
        nn.init.zeros_(self.markov_w2.weight)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if base_logits.size(-2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids, hidden_states)

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                bsz,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()
        for step_idx in range(proposal_len):
            step_hidden = None if hidden_states is None else hidden_states[:, step_idx]
            step_logits = self.apply_step_logits(
                base_logits[:, step_idx],
                token_ids=prev_token_ids,
                hidden_states=step_hidden,
            )
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = _sample(
                step_logits.unsqueeze(1),
                temperature=temperature,
            ).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


class GatedMarkovHead(VanillaMarkovHead):
    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "gated"
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_gate(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("gated Markov head requires hidden_states")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return torch.sigmoid(self.gate_proj(gate_inputs))

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = self.compute_gate(token_ids, hidden_states).to(prev_embeddings.dtype)
        return self.project_bias(gate * prev_embeddings)


class RNNMarkovHead(VanillaMarkovHead):
    """Recurrent DSpark Markov head unrolled inside one draft block."""

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "rnn"
        self.state_size = markov_rank
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def teacher_forcing_latents(self, token_ids, hidden_states):
        """Keep every recurrent state, defer only the vocabulary projection."""
        state = hidden_states.new_zeros(*hidden_states.shape[:-2], self.markov_rank)
        outputs = []
        for step in range(hidden_states.size(-2)):
            previous = self.get_prev_embeddings(token_ids[..., step])
            z = torch.cat((state, previous, hidden_states[..., step, :]), dim=-1)
            gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
            gate = torch.sigmoid(gate_raw)
            state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
            outputs.append(torch.tanh(output_raw))
        return torch.stack(outputs, dim=-2)

    def _rnn_step(
        self,
        state: torch.Tensor,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        return new_state, self.project_bias(torch.tanh(output_raw))

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("rnn Markov head requires hidden_states")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = torch.zeros_like(prev_embeddings)
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("rnn Markov head requires hidden_states")
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits

        state = torch.zeros(
            *base_logits.shape[:-2],
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        output_logits = []
        for step_idx in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., step_idx])
            state, bias = self._rnn_step(
                state,
                prev_emb,
                hidden_states[..., step_idx, :],
            )
            output_logits.append(base_logits[..., step_idx, :] + bias)
        return torch.stack(output_logits, dim=-2)

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states is None:
            raise ValueError("rnn Markov head requires hidden_states")
        bsz, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                bsz,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits

        state = torch.zeros(
            bsz,
            self.markov_rank,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()
        for step_idx in range(proposal_len):
            prev_emb = self.get_prev_embeddings(prev_token_ids)
            state, bias = self._rnn_step(state, prev_emb, hidden_states[:, step_idx])
            step_logits = base_logits[:, step_idx] + bias
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = _sample(
                step_logits.unsqueeze(1),
                temperature=temperature,
            ).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


def build_markov_head(config, dspark_config: dict) -> Optional[nn.Module]:
    markov_rank = int(dspark_config.get("markov_rank", 0) or 0)
    if markov_rank < 0:
        raise ValueError(f"markov_rank must be >= 0, got {markov_rank}")
    if markov_rank == 0:
        return None

    markov_head_type = str(dspark_config.get("markov_head_type", "vanilla")).lower()
    if markov_head_type == "vanilla":
        return VanillaMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
        )
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    if markov_head_type == "rnn":
        return RNNMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    raise ValueError(f"Unsupported markov_head_type: {markov_head_type!r}")


"""DSpark additive logits; original head math reused from LEGOSpec."""

from legospec.modeling.draft.contracts import ProposalOutput
from legospec.modeling.draft.proposal.parallel import ParallelProposal


class MarkovProposal(ParallelProposal):
    head_class = VanillaMarkovHead

    def __init__(self, config):
        super().__init__(config)
        recipe = config.composable_config["proposal"]
        kwargs = dict(
            vocab_size=config.vocab_size, markov_rank=int(recipe.get("rank", 256))
        )
        if self.head_class is not VanillaMarkovHead:
            kwargs["hidden_size"] = config.hidden_size
        self.markov_head = self.head_class(**kwargs)

    def forward(self, features, context, shared_ops):
        if context.inputs.mode == "features":
            return super().forward(features, context, shared_ops)
        draft_hidden_states = self.prediction_hidden(features, context)
        base_logits = self.project_rows(
            shared_ops.lm_head, draft_hidden_states, context
        )
        if context.inputs.mode == "sample":
            import torch

            shape = draft_hidden_states.shape
            tokens, draft_logits = self.markov_head.sample_block_tokens(
                base_logits.flatten(0, 1),
                hidden_states=draft_hidden_states.flatten(0, 1),
                first_prev_token_ids=context.inputs.anchor_token_ids.flatten(),
                temperature=context.inputs.temperature,
            )
            draft_logits = draft_logits.reshape(*shape[:-1], -1)
            return ProposalOutput(
                draft_hidden_states,
                draft_logits,
                base_logits,
                tokens.reshape(shape[:-1]),
                torch.softmax(
                    draft_logits.float() / (context.inputs.temperature or 1.0), -1
                ),
                draft_state=features.draft_state,
            )
        if context.inputs.previous_token_ids is None:
            raise ValueError("Teacher forcing requires previous_token_ids")
        if context.inputs.projection_rows is not None:
            head = self.markov_head
            if hasattr(head, "teacher_forcing_latents"):
                latent = head.teacher_forcing_latents(
                    context.inputs.previous_token_ids, draft_hidden_states
                )
            else:
                latent = head.get_prev_embeddings(context.inputs.previous_token_ids)
            draft_logits = base_logits + self.project_rows(
                head.project_bias, latent, context
            )
        else:
            draft_logits = self.markov_head.apply_block_logits(
                base_logits,
                token_ids=context.inputs.previous_token_ids,
                hidden_states=draft_hidden_states,
            )
        return ProposalOutput(
            draft_hidden_states,
            draft_logits,
            base_logits,
            draft_state=features.draft_state,
        )


class RNNProposal(MarkovProposal):
    head_class = RNNMarkovHead
