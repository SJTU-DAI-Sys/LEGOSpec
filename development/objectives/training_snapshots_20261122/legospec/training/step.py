"""One FSDP forward owns all trainable fusion, backbone, heads and losses."""

from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from legospec.training.outputs import RatioTerms, TrainingOutput
from legospec.modeling.draft.composable.contracts import (
    BackboneOutput,
    DraftInputs,
    SharedOps,
)

from legospec.training.batch import prepare_sample_epoch_layout
from legospec.training.component_reduction import combine_component_terms
from legospec.training.metrics import (
    acceptance_metric_terms,
    final_token_accuracy_terms,
)
from legospec.modeling.draft.composable.objective.base import Supervision
from legospec.modeling.draft.composable.objective.composition import LossBranches
from legospec.modeling.draft.composable.objective.vocab import (
    VocabCache,
    project_teacher_rows,
)


class TrainingStep(nn.Module):
    def __init__(
        self,
        draft_model,
        target_lm_head,
        target_embed_tokens,
        mask_token_id,
        num_anchors=32,
        objective_chunk_blocks=512,
        objective_gradient_checkpointing=None,
        loss_process_group=None,
        opd_data_mode="teacher_forced",
        complete_blocks=False,
        supervision_mode="sampled",
        training_mode="legacy",
        **kwargs,
    ):
        super().__init__()
        removed = {
            "selector_loss_alpha",
            "selector_base_ce_alpha",
            "selector_warmup_ratio",
            "selector_ramp_ratio",
        } & kwargs.keys()
        if removed:
            raise ValueError(
                "Removed Selector loss arguments; configure objective.base/final.weight instead"
            )
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.mask_token_id = 0 if mask_token_id is None else int(mask_token_id)
        self.num_anchors = int(num_anchors)
        self.objective_chunk_blocks = int(objective_chunk_blocks)
        self.objective_gradient_checkpointing = bool(
            self.objective_chunk_blocks > 0
            if objective_gradient_checkpointing is None
            else objective_gradient_checkpointing
        )
        self.loss_process_group = loss_process_group
        self.opd_data_mode = opd_data_mode
        self.complete_blocks = bool(complete_blocks)
        self.supervision_mode = supervision_mode
        if training_mode not in ("legacy", "optimized"):
            raise ValueError("training_mode must be legacy or optimized")
        self.optimized = training_mode == "optimized"
        if supervision_mode not in ("sampled", "all"):
            raise ValueError("supervision_mode must be sampled or all")
        if (
            getattr(draft_model, "uses_hidden_feedback", False)
            and opd_data_mode != "teacher_forced"
        ):
            raise ValueError(
                "EAGLE3 supports fixed-data teacher_forced TTT, not sampled_anchor OPD"
            )
        recipe = draft_model.config.composable_config
        self.loss_branches = LossBranches(recipe)
        draft_model.config.composable_config = self.loss_branches.recipe
        self.indexer_recipe = getattr(
            draft_model.config, "indexer_config", recipe.get("indexer")
        )
        self.selector_enabled = self.loss_branches.is_selector
        self.component_normalization = any(
            getattr(o, "needs_component_normalization", False)
            for o in self.loss_branches.objectives.values()
        )
        if self.component_normalization and self.indexer_recipe is not None:
            raise ValueError("OPD-Survival with an indexer is not implemented")

    def forward(
        self,
        input_ids,
        target_hidden_states,
        loss_mask,
        target_last_hidden_states=None,
        layout=None,
        window_denominators=None,
        epoch_progress=0.0,
        collect_detailed_metrics=False,
        context_valid_mask=None,
        opd_accepted_lengths=None,
        opd_global_denominators=None,
        opd_reference_samples=None,
        metadata_only=False,
        opd_scorer=None,
        sampled_opd_batch=None,
        execution_plan=None,
        **removed_loss_arguments,
    ):
        if self.opd_data_mode == "sampled_anchor":
            from legospec.training.anchor_opd import forward_anchor_opd

            if removed_loss_arguments:
                raise ValueError(
                    "Unexpected sampled OPD arguments: "
                    + ", ".join(removed_loss_arguments)
                )
            return forward_anchor_opd(
                self,
                input_ids,
                target_hidden_states,
                loss_mask,
                context_valid_mask=context_valid_mask,
                target_last_hidden_states=target_last_hidden_states,
                opd_scorer=opd_scorer,
                sampled_opd_batch=sampled_opd_batch,
                layout=layout,
                opd_global_denominators=opd_global_denominators,
                opd_reference_samples=opd_reference_samples,
                metadata_only=metadata_only,
                collect_detailed_metrics=collect_detailed_metrics,
            )
        model = self.draft_model
        hidden_feedback = getattr(model, "uses_hidden_feedback", False)
        if metadata_only and not self.component_normalization:
            raise ValueError("Metadata forward requires a component objective")
        if removed_loss_arguments:
            raise ValueError(
                "Removed loss arguments: "
                + ", ".join(removed_loss_arguments)
                + "; configure objective.base/final.weight instead"
            )
        # Teacher-free objectives do not acquire a teacher requirement just for
        # logging. Distribution acceptance diagnostics require available logits.
        acceptance_metrics = (
            collect_detailed_metrics and target_last_hidden_states is not None
        )
        if layout is None:
            effective_mask = loss_mask
            if hidden_feedback and context_valid_mask is not None:
                effective_mask = effective_mask * context_valid_mask
            layout = prepare_sample_epoch_layout(
                input_ids,
                effective_mask,
                torch.ones_like(input_ids, dtype=torch.bool)
                if context_valid_mask is None else context_valid_mask,
                block_size=model.block_size,
                num_anchors=self.num_anchors,
                seed=torch.initial_seed(),
                epoch=0,
                supervision_mode=self.supervision_mode if hidden_feedback else "sampled",
                minimum_anchor=1 if hidden_feedback else 0,
            ).to(input_ids.device)
        original_shape = layout.anchor_positions.shape
        source_layout = layout
        reference_batch_size = original_shape[0]
        sample_anchor_counts = layout.prediction_mask.bool().any(-1).sum(-1)
        packed_plan = None
        if self.optimized:
            from legospec.training.batch import pack_training_batch

            (
                input_ids,
                target_hidden_states,
                loss_mask,
                target_last_hidden_states,
                context_valid_mask,
                layout,
                packed_plan,
            ) = pack_training_batch(
                input_ids,
                target_hidden_states,
                loss_mask,
                target_last_hidden_states,
                context_valid_mask,
                layout,
                model.block_size,
                query_len=model.ttt_length if hidden_feedback else model.block_size,
                plan=execution_plan,
            )
            if opd_accepted_lengths is not None:
                opd_accepted_lengths = {
                    k: v.flatten(0, 1)
                    .index_select(0, packed_plan.kept_block_indices)
                    .unsqueeze(0)
                    for k, v in opd_accepted_lengths.items()
                }
        batch_size, num_anchors = layout.anchor_positions.shape
        supervision_counts = (
            {"supervision/anchors": layout.block_keep_mask.sum().float().detach()}
            if hidden_feedback
            else {}
        )
        if hidden_feedback:
            valid = (
                torch.ones_like(input_ids, dtype=torch.bool)
                if context_valid_mask is None
                else context_valid_mask.bool()
            )
            # TTT uses right padding, matching the target-cache collator.
            if (valid[:, 1:] & ~valid[:, :-1]).any():
                raise ValueError(
                    "EAGLE3 training expects contiguous right-padded sequences"
                )
            if ((layout.anchor_positions < 1) & layout.block_keep_mask).any():
                raise ValueError("EAGLE3 anchors require a predecessor hidden")
            if (layout.block_keep_mask & ~layout.prediction_mask.bool().all(-1)).any():
                raise ValueError("EAGLE3 supervision requires complete blocks")
            inputs = None
            hidden = model.forward_hidden_feedback(
                target_hidden_states,
                input_ids,
                valid,
                layout.anchor_positions,
                SharedOps(self.embed_tokens, self.lm_head),
                packed_plan=packed_plan,
            )
            from legospec.modeling.draft.composable.contracts import ProposalOutput

            prediction = ProposalOutput(hidden, backbone_hidden_states=hidden)
            index_aux = None
        else:
            noise_ids = input_ids.new_full(
                (batch_size, num_anchors, model.block_size), self.mask_token_id
            )
            anchor_indices = layout.anchor_positions
            if packed_plan is not None:
                anchor_indices = (
                    anchor_indices
                    + packed_plan.context_offsets[packed_plan.block_sample_ids][None]
                )
            noise_ids[..., 0] = input_ids.gather(1, anchor_indices).masked_fill(
                ~layout.block_keep_mask, self.mask_token_id
            )
            inputs = DraftInputs(
                target_hidden_states=target_hidden_states,
                draft_embeddings=self.embed_tokens(noise_ids.flatten(1)),
                anchor_positions=layout.anchor_positions,
                block_keep_mask=layout.block_keep_mask,
                mode="features",
                context_valid_mask=context_valid_mask,
                prediction_mask=layout.prediction_mask,
                collect_indexer_teacher=self.indexer_recipe is not None,
                packed_plan=packed_plan,
                context_position_ids=None
                if packed_plan is None
                else packed_plan.context_position_ids[None],
                document_ids=None
                if packed_plan is None
                else torch.repeat_interleave(
                    torch.arange(reference_batch_size, device=input_ids.device),
                    packed_plan.lengths,
                )[None],
                anchor_document_ids=None
                if packed_plan is None
                else packed_plan.block_sample_ids[None],
            )
            prediction = model(
                inputs, shared_ops=SharedOps(self.embed_tokens, self.lm_head)
            )
            index_aux = prediction.indexer_aux
        if self.indexer_recipe is not None and self.indexer_recipe["stage"] == "warmup":
            if index_aux is None:
                raise ValueError("Warmup must return indexer auxiliary terms")
            numerator, denominator = index_aux.numerator, index_aux.denominator
            if window_denominators is not None:
                reference, _, global_index = window_denominators.to(numerator).unbind()
                numerator = reference * numerator / global_index.clamp_min(1)
                denominator = numerator.new_tensor(reference_batch_size)
            return TrainingOutput(
                numerator / denominator.clamp_min(1),
                RatioTerms(numerator, denominator),
                {
                    "objective/indexer_kl": RatioTerms(
                        index_aux.numerator.detach(), index_aux.denominator.detach()
                    )
                },
            )
        # Every checkpoint chunk owns whole blocks, preserving prefix objectives.
        # The public chunk limit was per sample in the padded execution. Keep
        # its total block capacity when samples share the packed anchor axis.
        effective_chunk_blocks = self.objective_chunk_blocks * (
            reference_batch_size if packed_plan is not None else 1
        )
        if packed_plan is not None:
            from legospec.training.supervision import align_objective_blocks

            block_hidden = prediction.backbone_hidden_states.reshape(
                batch_size,
                num_anchors,
                model.ttt_length if hidden_feedback else model.block_size,
                -1,
            )
            (block_hidden, anchors, keep, labels, previous, mask, indices, owners) = (
                align_objective_blocks(
                    block_hidden,
                    layout.anchor_positions,
                    layout.block_keep_mask,
                    layout.labels,
                    layout.previous_token_ids,
                    layout.prediction_mask,
                    layout.label_indices,
                    packed_plan.block_sample_ids[None],
                    head=self.lm_head,
                    chunk_size=effective_chunk_blocks,
                )
            )
            layout = replace(
                layout,
                anchor_positions=anchors,
                block_keep_mask=keep,
                labels=labels,
                previous_token_ids=previous,
                prediction_mask=mask,
                label_indices=indices,
            )
            prediction.backbone_hidden_states = block_hidden.flatten(1, 2)
            packed_owners = owners
            num_anchors = anchors.size(1)
            if opd_accepted_lengths is not None:
                opd_accepted_lengths = {
                    name: torch.nn.functional.pad(
                        value, (0, num_anchors - value.size(1))
                    )
                    for name, value in opd_accepted_lengths.items()
                }
        chunk_size = effective_chunk_blocks or num_anchors
        zero = prediction.draft_hidden_states.float().sum() * 0.0
        names = list(self.loss_branches.objectives)
        numerators = {name: zero for name in names}
        denominators = {name: zero.detach().clone() for name in names}
        metric_numerators = {}
        metric_denominators = {}
        # Sequence-balanced objectives need counts from the whole layout, not
        # the current checkpoint chunk. These are anchor counts, not tokens.
        correct_num = zero.detach().clone()
        accuracy_den = zero.detach().clone()
        walk_num = zero.detach().clone()
        expected_num = zero.detach().clone()
        block_den = zero.detach().clone()
        component_terms = {name: {} for name in names}
        component_sum_metrics = {}
        accepted_chunks = {
            name: []
            for name in names
            if getattr(
                self.loss_branches.objectives[name],
                "needs_component_normalization",
                False,
            )
        }
        for start in range(0, num_anchors, chunk_size):
            end = min(start + chunk_size, num_anchors)
            block_hidden = prediction.backbone_hidden_states.reshape(
                batch_size,
                num_anchors,
                model.ttt_length if hidden_feedback else model.block_size,
                -1,
            )[:, start:end]
            labels = layout.labels[:, start:end]
            previous = layout.previous_token_ids[:, start:end]
            mask = layout.prediction_mask[:, start:end]
            teacher_indices = None
            if target_last_hidden_states is not None and (
                self.loss_branches.needs_teacher or acceptance_metrics
            ):
                teacher_indices = (layout.label_indices[:, start:end] - 1).clamp_min(0)

            # Bind all layout slices: checkpoint replay must not read a later chunk.
            replay = (
                None
                if opd_accepted_lengths is None
                else {name: r[:, start:end] for name, r in opd_accepted_lengths.items()}
            )

            owners = None if packed_plan is None else packed_owners[:, start:end]
            projection_rows = None
            if self.optimized:
                from legospec.training.supervision import SupervisionRows

                row_mask, source_indices = mask, teacher_indices
                if (
                    execution_plan is not None
                    and execution_plan.supervision_mask is not None
                    and execution_plan.supervision_mask.device.type == "cpu"
                ):
                    width = execution_plan.supervision_mask.size(1)
                    padding = max(0, num_anchors - width)
                    row_mask = torch.nn.functional.pad(
                        execution_plan.supervision_mask, (0, 0, 0, padding)
                    )[:, start:end]
                    source_indices = torch.nn.functional.pad(
                        execution_plan.teacher_indices, (0, 0, 0, padding)
                    )[:, start:end]
                if source_indices is not None:
                    # Source identities address hidden.flatten(0, 1). Equal
                    # positions in different batch samples are different rows.
                    source_indices = source_indices + torch.arange(
                        target_last_hidden_states.size(0), device=source_indices.device
                    )[:, None, None] * target_last_hidden_states.size(1)
                projection_mask = row_mask.bool()
                if self.selector_enabled or acceptance_metrics:
                    # Preserve predecessor candidates and recurrent diagnostics.
                    projection_mask = (
                        projection_mask.flip([-1]).long().cumsum(-1).flip([-1]) > 0
                    )
                projection_rows = SupervisionRows.from_weights(
                    row_mask, projection_mask, teacher_source_index=source_indices
                ).to(mask.device)

            def make_chunk(
                labels,
                previous,
                mask,
                teacher_indices,
                anchors,
                keep,
                replay,
                first_chunk,
                owners,
                projection_rows,
            ):
                def run_chunk(hidden):
                    if hidden_feedback:
                        output = model.score_hidden_feedback(
                            hidden,
                            SharedOps(self.embed_tokens, self.lm_head),
                            projection_rows=projection_rows,
                        )
                    else:
                        chunk_inputs = replace(
                            inputs,
                            anchor_positions=anchors,
                            block_keep_mask=keep,
                            previous_token_ids=previous,
                            mode="teacher_forcing",
                            projection_rows=projection_rows,
                        )
                        output = model.proposal(
                            BackboneOutput(hidden.flatten(1, 2)),
                            SimpleNamespace(inputs=chunk_inputs),
                            SharedOps(self.embed_tokens, self.lm_head),
                        )
                    teacher_logits = None
                    vocab_cache = VocabCache()
                    if not metadata_only and (
                        self.loss_branches.needs_teacher or acceptance_metrics
                    ):
                        if teacher_indices is None:
                            raise ValueError(
                                "Active loss/metrics require target_last_hidden_states"
                            )
                        with torch.no_grad():
                            teacher_logits = project_teacher_rows(
                                self.lm_head,
                                target_last_hidden_states,
                                teacher_indices,
                                **(
                                    dict(
                                        active=mask.bool(),
                                        cache=vocab_cache,
                                        row_plan=projection_rows,
                                    )
                                    if self.optimized
                                    else {}
                                ),
                            )
                    supervision = Supervision(
                        labels=labels,
                        prediction_mask=mask,
                        target_logits=teacher_logits,
                        epoch_progress=epoch_progress,
                        sample_anchor_counts=sample_anchor_counts,
                        block_sample_ids=owners,
                        vocab_cache=vocab_cache,
                    )
                    if self.component_normalization:
                        components, accepted, sums, ratios = {}, {}, {}, {}
                        for name in names:
                            objective = self.loss_branches.objectives[name]
                            branch_prediction = (
                                replace(
                                    output,
                                    draft_logits=output.base_logits,
                                    candidate_logits=None,
                                    candidate_ids=None,
                                )
                                if name == "base"
                                else output
                            )
                            r = None if replay is None else replay.get(name)
                            branch_supervision = replace(
                                supervision, opd_accepted_lengths=r
                            )
                            if metadata_only:
                                if getattr(
                                    objective, "needs_component_normalization", False
                                ):
                                    r = objective.accepted_lengths(
                                        branch_prediction, branch_supervision
                                    )
                                    counts = objective.component_denominators(r, mask)
                                    accepted[name] = r
                                else:
                                    counts = {
                                        "token": objective.native_denominator(
                                            SimpleNamespace(
                                                prediction_mask=mask,
                                                sample_prediction_counts=mask.flatten(
                                                    1
                                                ).sum(-1),
                                            )
                                        )
                                    }
                                components[name] = {
                                    key: (hidden.float().sum() * 0.0, den)
                                    for key, den in counts.items()
                                }
                            else:
                                terms = objective(branch_prediction, branch_supervision)
                                native = terms.component_terms or {
                                    "token": terms.loss_terms
                                }
                                components[name] = {
                                    key: (term.numerator, term.denominator)
                                    for key, term in native.items()
                                }
                                sums.update(
                                    {
                                        name + "/" + key: value.detach()
                                        for key, value in terms.sum_metrics.items()
                                    }
                                )
                                ratios.update(
                                    {
                                        (name, key): term.detached()
                                        for key, term in terms.ratio_metrics.items()
                                        if key in objective.metric_names
                                    }
                                )
                            if objective.sample_reduction:
                                # The same full-layout sample count is repeated
                                # in each chunk; accumulate it only once.
                                if metadata_only:
                                    count = objective.native_denominator(source_layout)
                                    components[name]["token"] = (
                                        hidden.float().sum() * 0.0,
                                        count if first_chunk else count * 0.0,
                                    )
                                elif not first_chunk:
                                    components[name] = {
                                        key: (num, den * 0.0)
                                        for key, (num, den) in components[name].items()
                                    }
                        with torch.no_grad():
                            correct, count = final_token_accuracy_terms(
                                output, labels, mask
                            )
                            if acceptance_metrics and not metadata_only:
                                walk, expected, blocks = acceptance_metric_terms(
                                    getattr(model, "proposal", None),
                                    output,
                                    teacher_logits,
                                    labels,
                                    previous,
                                    mask,
                                    vocab_cache=supervision.vocab_cache,
                                )
                            else:
                                walk = expected = blocks = mask.new_zeros(())
                        return {
                            "components": components,
                            "accepted": accepted,
                            "sums": sums,
                            "ratios": ratios,
                            "accuracy": (correct, count),
                            "acceptance": (walk, expected, blocks),
                        }
                    values = []
                    for name in names:
                        terms = self.loss_branches.branch_output(
                            name, output, supervision
                        )
                        values.extend(
                            [terms.loss_terms.numerator, terms.loss_terms.denominator]
                        )
                        # Names are static objective metadata, so checkpoint
                        # replay has the same structure for every chunk.
                        for key in self.loss_branches.objectives[name].metric_names:
                            metric = terms.ratio_metrics[key]
                            values.extend(
                                [metric.numerator.detach(), metric.denominator.detach()]
                            )
                    with torch.no_grad():
                        correct, count = final_token_accuracy_terms(
                            output, labels, mask
                        )
                        if acceptance_metrics:
                            walk, expected, blocks = acceptance_metric_terms(
                                getattr(model, "proposal", None),
                                output,
                                teacher_logits,
                                labels,
                                previous,
                                mask,
                                vocab_cache=supervision.vocab_cache,
                            )
                        else:
                            walk = expected = blocks = mask.new_zeros(())
                    return (*values, correct, count, walk, expected, blocks)

                return run_chunk

            operation = make_chunk(
                labels,
                previous,
                mask,
                teacher_indices,
                layout.anchor_positions[:, start:end],
                layout.block_keep_mask[:, start:end],
                replay,
                start == 0,
                owners,
                projection_rows,
            )
            values = (
                checkpoint(operation, block_hidden, use_reentrant=False)
                if self.objective_gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
                else operation(block_hidden)
            )
            if self.component_normalization:
                for name, items in values["components"].items():
                    for key, (num, den) in items.items():
                        old = component_terms[name].get(key)
                        component_terms[name][key] = RatioTerms(
                            num if old is None else old.numerator + num,
                            den if old is None else old.denominator + den,
                        )
                for name, r in values["accepted"].items():
                    accepted_chunks[name].append(r.detach())
                for key, value in values["sums"].items():
                    component_sum_metrics[key] = (
                        component_sum_metrics.get(key, zero.detach()) + value
                    )
                for key, term in values["ratios"].items():
                    metric_numerators[key] = (
                        metric_numerators.get(key, zero.detach()) + term.numerator
                    )
                    metric_denominators[key] = (
                        metric_denominators.get(key, zero.detach()) + term.denominator
                    )
                correct, count = values["accuracy"]
                walk, expected, blocks = values["acceptance"]
                correct_num, accuracy_den = correct_num + correct, accuracy_den + count
                walk_num, expected_num, block_den = (
                    walk_num + walk,
                    expected_num + expected,
                    block_den + blocks,
                )
                continue
            offset = 0
            for name in names:
                numerators[name] = numerators[name] + values[offset]
                if (
                    not self.loss_branches.objectives[name].sample_reduction
                    or start == 0
                ):
                    denominators[name] = denominators[name] + values[offset + 1]
                offset += 2
                for key in self.loss_branches.objectives[name].metric_names:
                    metric_numerators[name, key] = (
                        metric_numerators.get((name, key), zero.detach())
                        + values[offset]
                    )
                    metric_denominators[name, key] = (
                        metric_denominators.get((name, key), zero.detach())
                        + values[offset + 1]
                    )
                    offset += 2
            correct_num = correct_num + values[-5]
            accuracy_den = accuracy_den + values[-4]
            walk_num = walk_num + values[-3]
            expected_num = expected_num + values[-2]
            block_den = block_den + values[-1]

        if self.component_normalization:
            if metadata_only:
                accepted = {
                    name: torch.cat(chunks, dim=1)
                    for name, chunks in accepted_chunks.items()
                }
                if packed_plan is not None:
                    from legospec.training.batch import unpack_block_values

                    accepted = {
                        k: unpack_block_values(v, packed_plan, original_shape)
                        for k, v in accepted.items()
                    }
                return {
                    "accepted_lengths": accepted,
                    "component_denominators": {
                        name: {
                            key: term.denominator.detach()
                            for key, term in items.items()
                        }
                        for name, items in component_terms.items()
                    },
                    "sample_count": reference_batch_size,
                }
            if self.indexer_recipe is not None:
                raise ValueError("OPD-Survival with an indexer is not implemented")
            numerator, denominator, component_metrics = combine_component_terms(
                component_terms,
                self.loss_branches.objectives,
                self.loss_branches.weights,
                reference_batch_size,
                self.loss_process_group,
                opd_global_denominators,
                opd_reference_samples,
            )
            metrics = {
                "hard_label/token_accuracy": RatioTerms(
                    correct_num.detach(), accuracy_den.detach()
                ),
                **component_metrics,
            }
            metrics.update(
                {
                    name + "/" + key: RatioTerms(
                        value.detach(), metric_denominators[name, key].detach()
                    )
                    for (name, key), value in metric_numerators.items()
                }
            )
            if acceptance_metrics:
                metrics["hard_label/block_accepted_length"] = RatioTerms(
                    walk_num.detach(), block_den.detach()
                )
                metrics["expected_accept_len"] = RatioTerms(
                    expected_num.detach(), block_den.detach()
                )
            return TrainingOutput(
                numerator / denominator.clamp_min(1e-12),
                RatioTerms(numerator, denominator.detach()),
                metrics,
                sum_metrics={**component_sum_metrics, **supervision_counts},
            )
        if self.selector_enabled:
            # Order is reference sample count, final denominator, base denominator.
            if window_denominators is None:
                reference = zero.new_tensor(reference_batch_size)
                global_denominators = denominators
            else:
                reference, final_global, base_global = window_denominators.to(
                    zero
                ).unbind()
                global_denominators = {"final": final_global, "base": base_global}
            numerator = reference * sum(
                self.loss_branches.weights[name]
                * numerators[name]
                / global_denominators[name].clamp_min(1e-12)
                for name in names
            )
            denominator = zero.new_tensor(reference_batch_size)
        else:
            numerator, denominator = numerators["final"], denominators["final"]
        if self.indexer_recipe is not None:
            if index_aux is None:
                raise ValueError("DSA training requires auxiliary statistics")
            if window_denominators is None:
                reference = zero.new_tensor(reference_batch_size)
                main_global, index_global = denominators["final"], index_aux.denominator
            else:
                reference, main_global, index_global = window_denominators.to(
                    zero
                ).unbind()
            numerator = reference * (
                numerators["final"] / main_global.clamp_min(1e-12)
                + self.indexer_recipe.get("loss_weight", 1.0)
                * index_aux.numerator
                / index_global.clamp_min(1)
            )
            denominator = zero.new_tensor(reference_batch_size)
        metrics = {
            "hard_label/token_accuracy": RatioTerms(
                correct_num.detach(), accuracy_den.detach()
            )
        }
        for name in names:
            metrics["objective/" + name] = RatioTerms(
                numerators[name].detach(), denominators[name].detach()
            )
            objective = self.loss_branches.objectives[name]
            for key in objective.metric_names:
                metrics[name + "/" + key] = RatioTerms(
                    metric_numerators[name, key].detach(),
                    metric_denominators[name, key].detach(),
                )
        if index_aux is not None:
            metrics["objective/indexer_kl"] = RatioTerms(
                index_aux.numerator.detach(), index_aux.denominator.detach()
            )
        if acceptance_metrics:
            metrics["hard_label/block_accepted_length"] = RatioTerms(
                walk_num.detach(), block_den.detach()
            )
            metrics["expected_accept_len"] = RatioTerms(
                expected_num.detach(), block_den.detach()
            )
        return TrainingOutput(
            numerator / denominator.clamp_min(1e-12),
            RatioTerms(numerator, denominator.detach()),
            metrics,
            sum_metrics=supervision_counts,
        )
