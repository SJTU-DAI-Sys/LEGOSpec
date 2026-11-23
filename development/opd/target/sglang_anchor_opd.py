"""SGLang sampled-anchor teacher: one document prefill, shared paged prefix KV.

Each active anchor is an independent causal EXTEND request into the same source
KV slots. Only its sampled suffix gets new slots; there is no prefix KV copying
or anchor chunking. This service owns the runner's pools until prefix.release().
"""

from array import array
from dataclasses import dataclass, field

import torch


@dataclass
class SGLangAnchorPrefix:
    owner: object
    valid: torch.Tensor
    anchors: torch.Tensor
    keep: torch.Tensor
    anchor_hidden: torch.Tensor
    full_hidden: torch.Tensor
    token_indices: list
    source_ids: list
    context_hidden: dict = field(default_factory=dict)
    released: bool = False

    def release(self):
        if self.released:
            return
        self.owner._clear_pools()
        self.owner._active_prefix = None
        self.context_hidden.clear()
        self.token_indices.clear()
        self.source_ids.clear()
        self.full_hidden = None
        self.released = True


class SGLangAnchorScorer:
    uses_live_anchor_hidden = True
    backend = "sglang"

    def __init__(self, target):
        self.target = target
        self.runner = target.model_runner
        self.model = self.runner.model.requires_grad_(False).eval()
        if self.model.config.model_type != "qwen3":
            raise ValueError("SGLang sampled-anchor OPD currently supports dense Qwen3")
        if int(self.runner.server_args.tp_size) != 1:
            raise ValueError("SGLang sampled-anchor OPD requires tp_size=1")
        from sglang.srt.runtime_context import get_schedule

        if get_schedule().page_size != 1:
            raise ValueError(
                "Arbitrary anchor prefix sharing requires SGLang page_size=1"
            )
        self.prefill_calls = self.branch_calls = 0
        self._active_prefix = None
        self._captured = {}

    def _clear_pools(self):
        self.runner.req_to_token_pool.clear()
        self.runner.token_to_kv_pool_allocator.clear()
        self._captured.clear()

    def _request(self, rid, ids, prefix_indices=None):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", ids),
            vocab_size=self.runner.model_config.vocab_size,
            sampling_params=SamplingParams(temperature=0.0, max_new_tokens=1),
        )
        req.init_next_round_input()
        if prefix_indices is not None:
            req.prefix_indices = prefix_indices
        req.set_extend_range(len(req.prefix_indices), len(ids))
        return req

    @torch.no_grad()
    def _forward(self, reqs, capture_layer_ids=()):
        from sglang.srt.managers.overlap_utils import resolve_forward_inputs
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.radix_cache import RadixCache
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        if not reqs:
            raise ValueError("Internal SGLang extend cannot contain zero requests")
        required = sum(req.extend_range.length for req in reqs)
        pool, allocator = (
            self.runner.req_to_token_pool,
            self.runner.token_to_kv_pool_allocator,
        )
        if len(reqs) > pool.available_size() or required > allocator.available_size():
            raise RuntimeError(
                "SGLang OPD pool too small for all anchors together: "
                f"need requests={len(reqs)}, suffix_tokens={required}; "
                f"available requests={pool.available_size()}, tokens={allocator.available_size()}"
            )
        if max(map(lambda req: len(req.get_fill_ids()), reqs)) > pool.max_context_len:
            raise ValueError(
                "Anchor branch exceeds the configured SGLang context length"
            )
        layer_ids = sorted(capture_layer_ids)
        self.target.adapter.remove_capture()
        if layer_ids:
            self.target.adapter.install_capture(self.model, layer_ids)
        else:
            self.model.capture_aux_hidden_states = False
            self.model.model.layers_to_capture = []
        tree = RadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
            )
        )
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree,
            model_config=self.runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        resolve_forward_inputs(batch, self.target.engine.future_map)
        forward = ForwardBatch.init_new(
            batch,
            self.runner,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_hidden_states_before_norm=False,
        )
        teacher = {}

        def capture_teacher(module, args):
            teacher["hidden"] = args[1]

        handle = self.model.logits_processor.register_forward_pre_hook(capture_teacher)
        try:
            output = self.runner.forward(forward).logits_output
            hidden = teacher["hidden"]
            if hidden.shape[0] != required:
                raise RuntimeError("SGLang must return every EXTEND token hidden state")
            captured = {}
            if layer_ids:
                aux = output.hidden_states.reshape(
                    required, len(layer_ids), hidden.shape[-1]
                )
                captured = {layer: aux[:, i] for i, layer in enumerate(layer_ids)}
            return hidden, captured, batch
        finally:
            handle.remove()
            self.target.adapter.remove_capture()
            self.model.capture_aux_hidden_states = False
            self.model.model.layers_to_capture = []

    @torch.no_grad()
    def prefill_anchors(self, input_ids, valid, anchors, keep, *, capture_layer_ids=()):
        if self._active_prefix is not None:
            raise RuntimeError(
                "Release the previous SGLang anchor prefix before prefilling again"
            )
        if input_ids.ndim != 2 or input_ids.shape != valid.shape:
            raise ValueError("Source IDs and validity must have matching [N,S] axes")
        if anchors.shape != keep.shape or anchors.shape[0] != input_ids.shape[0]:
            raise ValueError("Anchor axes must match source batch")
        if input_ids.shape[1] == 0 or input_ids.shape[0] == 0:
            raise ValueError("SGLang source batch must be nonempty")
        valid = valid.bool()
        lengths = valid.long().sum(-1)
        expected = (
            torch.arange(valid.shape[1], device=valid.device)[None] < lengths[:, None]
        )
        if not torch.equal(expected, valid):
            raise ValueError(
                "SGLang anchor OPD requires right-padded source attention masks"
            )
        if ((anchors[keep] < 0) | (anchors[keep] >= input_ids.shape[1])).any():
            raise ValueError("Active anchor outside source")
        safe = anchors.masked_fill(~keep, 0)
        if not valid.gather(1, safe).masked_fill(~keep, True).all():
            raise ValueError("Active anchor must refer to a valid source token")
        ids_cpu = input_ids.tolist()
        lens = [max(1, n) for n in lengths.tolist()]
        source = [row[:length] for row, length in zip(ids_cpu, lens)]
        reqs = [self._request(f"prefix-{i}", ids) for i, ids in enumerate(source)]
        try:
            hidden, captured, batch = self._forward(reqs, capture_layer_ids)
            n, s = input_ids.shape
            full = hidden.new_zeros(n, s, hidden.shape[-1])
            contexts = {
                i: value.new_zeros(n, s, value.shape[-1])
                for i, value in captured.items()
            }
            offset = 0
            for index, length in enumerate(lens):
                full[index, :length] = hidden[offset : offset + length]
                for layer_id in contexts:
                    contexts[layer_id][index, :length] = captured[layer_id][
                        offset : offset + length
                    ]
                offset += length
            # SGLang's write_req_to_token_pool Triton kernel reads prefix
            # pointers as int64, although the request table itself is int32.
            indices = [
                self.runner.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, :length
                ].to(torch.int64)
                for req, length in zip(reqs, lens)
            ]
            first = full.gather(1, safe[..., None].expand(-1, -1, full.shape[-1]))
            prefix = SGLangAnchorPrefix(
                self, valid, safe, keep, first, full, indices, source, contexts
            )
            self._active_prefix = prefix
            self.prefill_calls += 1
            return prefix
        except BaseException:
            self._clear_pools()
            raise

    @torch.no_grad()
    def score_sampled_blocks(self, prefix, draft_ids, *, cached_anchor_hidden=None):
        if prefix is not self._active_prefix or prefix.released:
            raise ValueError("Expired or foreign SGLang prefix")
        n, a, k = draft_ids.shape
        if prefix.anchors.shape != (n, a) or k < 1:
            raise ValueError("Sampled branch shape must align with the prefix anchors")
        if cached_anchor_hidden is not None:
            raise ValueError("SGLang verification uses its live anchor hidden state")
        first = prefix.anchor_hidden
        result = first.new_zeros(n, a, k, first.shape[-1])
        result[..., 0, :] = first
        active = prefix.keep.nonzero().tolist()
        if k > 1 and active:
            sampled = draft_ids[..., :-1].tolist()
            anchors = prefix.anchors.tolist()
            reqs = []
            for row, col in active:
                length = anchors[row][col] + 1
                reqs.append(
                    self._request(
                        f"branch-{row}-{col}",
                        prefix.source_ids[row][:length] + sampled[row][col],
                        prefix.token_indices[row][:length],
                    )
                )
            try:
                hidden, _, batch = self._forward(reqs)
                suffix = hidden.reshape(len(active), k - 1, -1)
                rows, cols = prefix.keep.nonzero(as_tuple=True)
                result[rows, cols, 1:] = suffix
            except BaseException:
                prefix.release()
                raise
        self.branch_calls += 1
        return result
