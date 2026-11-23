from __future__ import annotations

"""Shared selector primitives; independent of draft algorithm implementations.
"""
from typing import Optional

import torch
from torch import nn


class CandidateSelector(nn.Module):
    """Low-rank transition scorer used to select a coherent token path."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        state_rank: int,
        top_k: int,
        initializer_range: float,
    ) -> None:
        super().__init__()
        if state_rank < 1:
            raise ValueError("DFlash2 selector_rank must be >= 1")
        if top_k < 1 or top_k > vocab_size:
            raise ValueError(
                "DFlash2 selector_top_k must be in [1, vocab_size], got "
                f"{top_k} for vocab_size={vocab_size}"
            )
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(
            torch.empty(int(vocab_size), int(state_rank))
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(int(vocab_size), int(state_rank))
        )
        self.hidden_projection = nn.Linear(
            int(hidden_size),
            int(state_rank),
            bias=False,
        )
        nn.init.normal_(self.predecessor_codebook, std=initializer_range)
        # The transition must start as a no-op so a fresh DFlash2 selector is
        # numerically identical to the unary DFlash proposal. The successor
        # factor learns first; the other bilinear factors receive signal once it
        # becomes non-zero.
        nn.init.zeros_(self.successor_codebook)

    def score_candidates(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Add low-rank predecessor transitions to a candidate set's logits."""

        predecessor = self.predecessor_codebook[predecessor_ids]
        successor = self.successor_codebook[candidate_ids]
        context = predecessor * self.hidden_projection(hidden_states)
        transition = torch.einsum("...r,...kr->...k", context, successor)
        return unary_logits + transition

    def build_lattice(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build SGLang's K-by-K transition lattice for every proposal slot."""

        predecessor_ids = torch.cat(
            [
                anchor_token_ids[:, None, None].expand(-1, 1, self.top_k),
                candidate_ids[:, :-1],
            ],
            dim=1,
        )
        predecessor = self.predecessor_codebook[predecessor_ids]
        successor = self.successor_codebook[candidate_ids]
        context = predecessor * self.hidden_projection(hidden_states)[:, :, None]
        return unary_logits[:, :, None] + torch.einsum(
            "blpr,blcr->blpc",
            context,
            successor,
        )

    def greedy_path(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Walk the candidate lattice for the local greedy inference helper."""

        path, _ = self.select_path(
            candidate_ids=candidate_ids,
            unary_logits=unary_logits,
            hidden_states=hidden_states,
            anchor_token_ids=anchor_token_ids,
            temperature=0.0,
        )
        return path

    def select_path(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Walk greedily or sample from each predecessor-conditioned row.

        For sampling, the returned rows are the sparse proposal distribution
        consumed by lossless speculative rejection sampling.
        """

        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")

        predecessor_ids = anchor_token_ids
        path = []
        probability_rows = []
        for position in range(candidate_ids.shape[1]):
            scores = self.score_candidates(
                candidate_ids=candidate_ids[:, position],
                unary_logits=unary_logits[:, position],
                hidden_states=hidden_states[:, position],
                predecessor_ids=predecessor_ids,
            )
            if temperature > 0:
                probabilities = torch.softmax(scores.float() / temperature, dim=-1)
                selected = torch.multinomial(probabilities, 1)
                probability_rows.append(probabilities)
            else:
                selected = scores.argmax(dim=-1, keepdim=True)
            predecessor_ids = candidate_ids[:, position].gather(1, selected)[:, 0]
            path.append(predecessor_ids)
        return (
            torch.stack(path, dim=1),
            torch.stack(probability_rows, dim=1) if probability_rows else None,
        )

    def reset_draft_parameters(self):
        nn.init.zeros_(self.successor_codebook)


"""Parallel top-K transition lattices with full-vocabulary or candidate-only support."""


from legospec.modeling.draft.contracts import ProposalOutput
from legospec.modeling.draft.proposal.parallel import ParallelProposal


class SelectorProposal(ParallelProposal):
    def __init__(self, config):
        super().__init__(config)
        recipe = config.composable_config["proposal"]
        self.candidate_selector = CandidateSelector(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            state_rank=int(recipe.get("rank", 256)),
            top_k=int(recipe.get("top_k", 16)),
            initializer_range=config.initializer_range,
        )
        self.stop_gradient = bool(recipe.get("stop_gradient", False))
        self.full_vocab_residual = (
            recipe.get("selector_mode", "candidate_only") == "full_vocab_residual"
        )

    def parallel_lattice(
        self, base_logits, hidden_states, anchor_token_ids, candidate_ids=None
    ):
        """Compute every slot's K-by-K delta table before any path selection.

        Row p of slot i is conditioned on candidate p of slot i-1. Slot zero
        has a known anchor instead, repeated across rows. No sampled token is
        needed to construct the table, and no vocabulary-sized predecessor
        expansion or predecessor-conditioned rescoring is used.
        """
        if base_logits.ndim < 3 or base_logits.size(-2) < 1:
            raise ValueError(
                "Selector lattice requires a non-empty complete block of slots"
            )
        if candidate_ids is None:
            candidate_ids = base_logits.topk(
                self.candidate_selector.top_k, dim=-1
            ).indices
        hidden = hidden_states.detach() if self.stop_gradient else hidden_states
        slots, k = candidate_ids.shape[-2:]
        # Zero unary extracts transition deltas without BF16 subtraction or
        # double addition. The shared scorer projects all slots in one call.
        delta = self.candidate_selector.build_lattice(
            candidate_ids=candidate_ids.reshape(-1, slots, k),
            unary_logits=base_logits.new_zeros(candidate_ids.shape).reshape(
                -1, slots, k
            ),
            hidden_states=hidden.reshape(-1, slots, hidden.size(-1)),
            anchor_token_ids=anchor_token_ids.reshape(-1),
        )
        return delta.reshape(*candidate_ids.shape, k), candidate_ids

    @staticmethod
    def predecessor_rows(candidate_ids, predecessor_ids):
        """Locate true predecessors in the *previous* slot's candidate set."""
        k = candidate_ids.size(-1)
        known_anchor = predecessor_ids[..., :1, None].expand(
            *candidate_ids.shape[:-2], 1, k
        )
        previous_candidates = torch.cat(
            (known_anchor, candidate_ids[..., :-1, :]), dim=-2
        )
        matches = previous_candidates == predecessor_ids[..., None]
        return matches.long().argmax(-1), matches.any(-1)

    def candidate_corrections(
        self, base_logits, hidden_states, predecessor_ids, candidate_ids=None
    ):
        """Gather precomputed rows; missing predecessors use zero correction.

        This scoring API consumes complete blocks, matching parallel inference.
        A current gold label outside top-K remains supervised. A predecessor
        outside the previous top-K removes only that slot's Selector correction.
        """
        lattice, candidate_ids = self.parallel_lattice(
            base_logits,
            hidden_states,
            predecessor_ids[..., 0],
            candidate_ids,
        )
        rows, covered = self.predecessor_rows(candidate_ids, predecessor_ids)
        k = candidate_ids.size(-1)
        delta = lattice.gather(
            -2, rows[..., None, None].expand(*rows.shape, 1, k)
        ).squeeze(-2)
        delta = torch.where(covered[..., None], delta, torch.zeros_like(delta))
        return delta, candidate_ids

    def full_vocab_logits(
        self, base_logits, hidden_states, predecessor_ids, candidate_ids=None
    ):
        delta, candidate_ids = self.candidate_corrections(
            base_logits, hidden_states, predecessor_ids, candidate_ids
        )
        return base_logits.float().scatter_add(
            -1, candidate_ids, delta.float()
        ), candidate_ids

    def sample_candidates_parallel(
        self, base_logits, hidden_states, anchor_token_ids, temperature
    ):
        """Candidate-only distribution using a parallel score table."""
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        lattice, candidate_ids = self.parallel_lattice(
            base_logits, hidden_states, anchor_token_ids
        )
        scores = base_logits.gather(-1, candidate_ids)[..., None, :] + lattice
        row = torch.zeros_like(anchor_token_ids)
        tokens, probabilities = [], []
        k = candidate_ids.size(-1)
        for position in range(base_logits.size(-2)):
            selected_scores = (
                scores[:, position]
                .gather(1, row[:, None, None].expand(-1, 1, k))
                .squeeze(1)
            )
            probability = torch.softmax(
                selected_scores.float() / (temperature or 1.0), -1
            )
            row = (
                selected_scores.argmax(-1)
                if temperature == 0
                else torch.multinomial(probability, 1).squeeze(-1)
            )
            tokens.append(candidate_ids[:, position].gather(1, row[:, None]).squeeze(1))
            probabilities.append(probability)
        return torch.stack(tokens, -1), torch.stack(probabilities, -2), candidate_ids

    def sample_full_vocab(
        self,
        base_logits,
        hidden_states,
        anchor_token_ids,
        temperature,
        *,
        return_distributions=True,
    ):
        """Sample actual full-vocabulary conditionals along the generated path."""
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not return_distributions and temperature != 0:
            raise ValueError(
                "Token-only full-vocabulary sampling requires greedy temperature=0"
            )
        # All transition scores are ready before the first random draw. The
        # remaining loop only gathers rows, normalizes, and selects tokens.
        lattice, candidate_ids = self.parallel_lattice(
            base_logits, hidden_states, anchor_token_ids
        )
        previous = anchor_token_ids
        tokens, logits, probabilities = [], [], []
        k = candidate_ids.size(-1)
        for position in range(base_logits.size(-2)):
            if position == 0:
                delta = lattice[:, position, 0]
            else:
                matches = candidate_ids[:, position - 1] == previous[:, None]
                row_index = matches.long().argmax(-1)
                delta = (
                    lattice[:, position]
                    .gather(1, row_index[:, None, None].expand(-1, 1, k))
                    .squeeze(1)
                )
                delta = torch.where(
                    matches.any(-1)[:, None], delta, torch.zeros_like(delta)
                )
            if not return_distributions:
                # Logging consumes only greedy IDs. Bound the temporary score
                # row and never retain vocabulary logits/probabilities per slot.
                token = torch.cat(
                    [
                        base_logits[start : start + 64, position]
                        .float()
                        .scatter_add(
                            -1,
                            candidate_ids[start : start + 64, position],
                            delta[start : start + 64].float(),
                        )
                        .argmax(-1)
                        for start in range(0, max(1, base_logits.size(0)), 64)
                    ]
                )
                tokens.append(token)
                previous = token
                continue
            row = (
                base_logits[:, position]
                .float()
                .scatter_add(-1, candidate_ids[:, position], delta.float())
            )
            probability = torch.softmax(row / (temperature or 1.0), -1)
            token = (
                row.argmax(-1)
                if temperature == 0
                else torch.multinomial(probability, 1).squeeze(-1)
            )
            tokens.append(token)
            logits.append(row)
            probabilities.append(probability)
            previous = token
        return (
            torch.stack(tokens, -1),
            torch.stack(logits, -2) if return_distributions else None,
            torch.stack(probabilities, -2) if return_distributions else None,
        )

    def forward(self, features, context, shared_ops):
        if context.inputs.mode == "features":
            return super().forward(features, context, shared_ops)
        draft_hidden_states = self.prediction_hidden(features, context)
        base_logits = self.project_rows(
            shared_ops.lm_head, draft_hidden_states, context
        )
        output = ProposalOutput(
            draft_hidden_states,
            base_logits=base_logits,
            draft_state=features.draft_state,
        )
        if self.full_vocab_residual:
            if context.inputs.mode == "sample":
                if context.inputs.anchor_token_ids is None:
                    raise ValueError("Selector sampling requires anchor_token_ids")
                shape = draft_hidden_states.shape[:-1]
                tokens, logits, probabilities = self.sample_full_vocab(
                    base_logits.flatten(0, 1),
                    draft_hidden_states.flatten(0, 1),
                    context.inputs.anchor_token_ids.flatten(),
                    context.inputs.temperature,
                )
                output.draft_token_ids = tokens.reshape(shape)
                output.draft_logits = logits.reshape(*shape, -1)
                output.draft_probabilities = probabilities.reshape(*shape, -1)
            else:
                if context.inputs.previous_token_ids is None:
                    raise ValueError("Selector scoring requires previous_token_ids")
                output.draft_logits, output.candidate_ids = self.full_vocab_logits(
                    base_logits,
                    draft_hidden_states,
                    context.inputs.previous_token_ids,
                )
            # candidate_logits deliberately stays None: generic verifier and
            # accuracy code must consume the full-vocabulary draft_logits.
            return output
        candidate_logits, candidate_ids = base_logits.topk(
            self.candidate_selector.top_k, dim=-1
        )
        if context.inputs.mode == "sample":
            tokens, draft_probabilities, _ = self.sample_candidates_parallel(
                base_logits=base_logits.flatten(0, 1),
                hidden_states=draft_hidden_states.flatten(0, 1),
                anchor_token_ids=context.inputs.anchor_token_ids.flatten(),
                temperature=context.inputs.temperature,
            )
            output.draft_token_ids = tokens.reshape(draft_hidden_states.shape[:-1])
            draft_probabilities = draft_probabilities.reshape_as(candidate_logits)
            output.draft_probabilities = base_logits.new_zeros(
                base_logits.shape, dtype=torch.float32
            ).scatter(-1, candidate_ids, draft_probabilities)
        else:
            if context.inputs.previous_token_ids is None:
                raise ValueError("Selector scoring requires previous_token_ids")
            delta, candidate_ids = self.candidate_corrections(
                base_logits,
                draft_hidden_states,
                context.inputs.previous_token_ids,
                candidate_ids,
            )
            unary = (
                candidate_logits.detach() if self.stop_gradient else candidate_logits
            )
            output.candidate_logits = unary.float() + delta.float()
        if context.inputs.mode == "sample":
            output.draft_logits = output.draft_probabilities.log() * (
                context.inputs.temperature or 1.0
            )
        output.candidate_ids = candidate_ids
        return output
