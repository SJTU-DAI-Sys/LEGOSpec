"""Additive token accuracy and speculative acceptance diagnostics."""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from legospec.modeling.draft.objective.outputs import RatioTerms


@torch.no_grad()
def final_prediction_ids(prediction):
    """Top-1 of the actual proposal, including sparse Selector reranking."""
    if prediction.candidate_logits is not None:
        if prediction.candidate_ids is None:
            raise ValueError("Final sparse proposal scores require candidate_ids")
        indices = prediction.candidate_logits.argmax(-1, keepdim=True)
        return prediction.candidate_ids.gather(-1, indices).squeeze(-1)
    if prediction.draft_logits is None:
        raise ValueError("Final proposal logits are required for token accuracy")
    return prediction.draft_logits.argmax(-1)


@torch.no_grad()
def final_token_accuracy_terms(prediction, labels, prediction_mask):
    """Count every supervised token, including gold tokens outside the top-K."""
    supervised = prediction_mask > 0.5
    correct = (final_prediction_ids(prediction) == labels) & supervised
    return correct.sum().float(), supervised.sum().float()


@torch.no_grad()
def acceptance_metric_terms(
    proposal,
    prediction,
    target_logits,
    labels,
    previous_token_ids,
    prediction_mask,
    *,
    vocab_cache=None,
):
    """Return additive walk/expected numerators and the valid-block count.

    Walks use free-running greedy predecessors, starting at each ground-truth
    anchor. Expected lengths use teacher-forced conditional proposal rows, just
    like DFlash2's training diagnostic; they are not a rollout evaluation.
    Both lengths include the anchor and stop at the first unsupervised slot.
    """
    supervised = prediction_mask > 0.5
    hidden = prediction.draft_hidden_states
    anchor_ids = previous_token_ids[..., 0].reshape(-1)
    shape = labels.shape
    probabilities = {}
    if vocab_cache is not None:
        probabilities["target_probabilities"] = vocab_cache.peek_detached_probabilities(
            target_logits
        )
        if prediction.candidate_logits is None:
            probabilities["draft_probabilities"] = (
                vocab_cache.peek_detached_probabilities(prediction.draft_logits)
            )
    if getattr(proposal, "full_vocab_residual", False):
        greedy_ids, _, _ = proposal.sample_full_vocab(
            prediction.base_logits.flatten(0, 1),
            hidden.flatten(0, 1),
            anchor_ids,
            0.0,
            return_distributions=False,
        )
        greedy_ids = greedy_ids.reshape(shape)
        acceptance = distribution_acceptance_probability(
            target_logits, prediction.draft_logits, **probabilities
        )
    elif hasattr(proposal, "candidate_selector"):
        selector = proposal.candidate_selector
        unary = prediction.base_logits.gather(-1, prediction.candidate_ids)
        greedy_ids = selector.greedy_path(
            candidate_ids=prediction.candidate_ids.flatten(0, 1),
            unary_logits=unary.flatten(0, 1),
            hidden_states=hidden.flatten(0, 1),
            anchor_token_ids=anchor_ids,
        ).reshape(shape)
        acceptance = sparse_distribution_acceptance_probability(
            target_logits,
            prediction.candidate_logits,
            prediction.candidate_ids,
            **probabilities,
        )
    else:
        if hasattr(proposal, "markov_head"):
            greedy_ids, _ = proposal.markov_head.sample_block_tokens(
                prediction.base_logits.flatten(0, 1),
                hidden_states=hidden.flatten(0, 1),
                first_prev_token_ids=anchor_ids,
                temperature=0.0,
            )
            greedy_ids = greedy_ids.reshape(shape)
        elif hasattr(proposal, "sequential_head"):
            greedy_ids, _ = proposal.sequential_head.sample_block_tokens(
                hidden_states=hidden.flatten(0, 1),
                first_prev_token_ids=anchor_ids,
                temperature=0.0,
            )
            greedy_ids = greedy_ids.reshape(shape)
        else:
            greedy_ids = final_prediction_ids(prediction)
        acceptance = distribution_acceptance_probability(
            target_logits, prediction.draft_logits, **probabilities
        )
    walk = block_accepted_length_terms(greedy_ids, labels, supervised)
    expected = expected_accept_len_terms(acceptance, supervised)
    return walk.numerator, expected.numerator, walk.denominator


class PrefixCounts(NamedTuple):
    reached: torch.Tensor
    accepted: torch.Tensor
    selector_covered: torch.Tensor
    selector_unary_correct: torch.Tensor
    eligible: torch.Tensor


def _prefix_events(hit: torch.Tensor, supervised: torch.Tensor):
    accepted = (hit & supervised).long().cumprod(dim=-1).bool()
    previous = torch.cat(
        (torch.ones_like(accepted[..., :1]), accepted[..., :-1]), dim=-1
    )
    return previous & supervised, accepted


def hard_label_prefix_counts(
    unary_hit: torch.Tensor,
    covered: torch.Tensor,
    selector_hit: Optional[torch.Tensor],
    supervised: torch.Tensor,
) -> PrefixCounts:
    if selector_hit is None:
        selector_reached = selector_accepted = torch.zeros_like(supervised)
    else:
        selector_reached, selector_accepted = _prefix_events(selector_hit, supervised)
    unary_reached, unary_accepted = _prefix_events(unary_hit, supervised)
    oracle_reached, oracle_accepted = _prefix_events(covered, supervised)

    def count(events):
        return F.pad(events.float().sum(dim=(0, 1)), (1, 0))

    return PrefixCounts(
        reached=torch.stack(
            [count(unary_reached), count(oracle_reached), count(selector_reached)]
        ),
        accepted=torch.stack(
            [count(unary_accepted), count(oracle_accepted), count(selector_accepted)]
        ),
        selector_covered=count(selector_reached & covered),
        selector_unary_correct=count(selector_reached & unary_hit),
        eligible=count(supervised.long().cumprod(dim=-1).bool()),
    )


def block_accepted_length_terms(
    predicted_ids: torch.Tensor,
    target_ids: torch.Tensor,
    supervised: torch.Tensor,
) -> RatioTerms:
    valid_blocks = supervised.any(dim=-1)
    accepted = (
        ((predicted_ids == target_ids) & supervised).long().cumprod(dim=-1).sum(dim=-1)
    )
    lengths = accepted.float() + 1.0
    return RatioTerms((lengths * valid_blocks).sum(), valid_blocks.sum())


def distribution_acceptance_probability(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    *,
    target_probabilities: Optional[torch.Tensor] = None,
    draft_probabilities: Optional[torch.Tensor] = None,
    row_chunk_size: int = 128,
) -> torch.Tensor:
    """Return speculative acceptance probability for two dense distributions."""

    if target_logits.shape != draft_logits.shape:
        raise ValueError("target_logits and draft_logits must have identical shapes")
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")
    vocab = target_logits.size(-1)
    target = target_logits.detach().reshape(-1, vocab)
    draft = draft_logits.detach().reshape(-1, vocab)
    target_cached = (
        None
        if target_probabilities is None
        else target_probabilities.reshape(-1, vocab)
    )
    draft_cached = (
        None if draft_probabilities is None else draft_probabilities.reshape(-1, vocab)
    )
    terms = []
    for start in range(0, target.size(0), row_chunk_size):
        stop = start + row_chunk_size
        p = (
            torch.softmax(target[start:stop].float(), dim=-1)
            if target_cached is None
            else target_cached[start:stop]
        )
        q = (
            torch.softmax(draft[start:stop].float(), dim=-1)
            if draft_cached is None
            else draft_cached[start:stop]
        )
        terms.append(torch.minimum(p, q).sum(dim=-1))
    if not terms:
        return target_logits.new_empty(target_logits.shape[:-1], dtype=torch.float32)
    return torch.cat(terms).reshape(target_logits.shape[:-1])


def sparse_distribution_acceptance_probability(
    target_logits: torch.Tensor,
    draft_candidate_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    target_probabilities: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return acceptance probability for a normalized sparse draft proposal."""

    if draft_candidate_logits.shape != candidate_ids.shape:
        raise ValueError("draft_candidate_logits and candidate_ids must match")
    if target_logits.shape[:-1] != candidate_ids.shape[:-1]:
        raise ValueError("target and candidate batch dimensions must match")
    target_probs = (
        torch.softmax(target_logits.detach().float(), dim=-1)
        if target_probabilities is None
        else target_probabilities
    )
    target_candidate_probs = target_probs.gather(-1, candidate_ids)
    draft_probs = torch.softmax(draft_candidate_logits.detach().float(), dim=-1)
    return torch.minimum(target_candidate_probs, draft_probs).sum(dim=-1)


def expected_accept_len_terms(
    acceptance_probability: torch.Tensor,
    supervised: torch.Tensor,
    block_valid: Optional[torch.Tensor] = None,
) -> RatioTerms:
    """Return additive terms for ``1 + sum_k prod_{i<=k} acceptance_i``."""

    if acceptance_probability.shape != supervised.shape:
        raise ValueError("acceptance_probability and supervised must match")
    valid_blocks = supervised.any(dim=-1) if block_valid is None else block_valid
    alive = acceptance_probability.detach().float() * supervised.float()
    expected = 1.0 + alive.cumprod(dim=-1).sum(dim=-1)
    return RatioTerms(
        (expected * valid_blocks.float()).sum(), valid_blocks.float().sum()
    )


def fsdp_global_weighted_mean(
    numerator: torch.Tensor, denominator: torch.Tensor, group=None
):
    global_denominator = denominator.detach().float().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM, group=group)
        world_size = dist.get_world_size(group)
    return numerator * (world_size / global_denominator.clamp_min(1e-6))
