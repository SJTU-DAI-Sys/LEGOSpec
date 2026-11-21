"""Native Gemma 4 SGLang prefill inside a collocated TP/DP draft trainer."""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from .training_target_model import TrainingTargetModel, TargetTrainingOutput
from .target_utils import FlattenedLMHead, SharedTargetEmbeddingsAndHead


def initialize_collocated_parallel(*, server_args, local_rank):
    """Give each teacher a TP-scoped SGLang world; preserve the trainer WORLD.

    All trainer ranks create every subgroup in the same collective order.
    SGLang owns these new groups, never the FSDP/optimizer process groups.
    """
    from litespec.distributed import get_tp_group, get_disaggregated_topology
    from sglang.srt.distributed import parallel_state
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.runtime_context import get_parallel

    if not dist.is_initialized():
        raise RuntimeError('Initialize training distributed groups before the SGLang teacher')
    if get_disaggregated_topology() is not None:
        raise ValueError('Gemma 4 native teacher currently requires collocated training')
    tp_size = dist.get_world_size(get_tp_group())
    world_size = dist.get_world_size()
    if world_size % tp_size or tp_size != server_args.tp_size:
        raise ValueError('SGLang and trainer TP widths disagree')
    ranks = dist.get_process_group_ranks(get_tp_group())
    expected = list(range((dist.get_rank()//tp_size)*tp_size, (dist.get_rank()//tp_size+1)*tp_size))
    if ranks != expected:
        raise ValueError('Gemma 4 native teacher requires the trainer contiguous TP mesh')
    if parallel_state._WORLD is not None:
        raise RuntimeError('A SGLang world already exists in this process')
    options = dict(
        local_rank=local_rank, torch_distributed_backend=dist.get_backend(),
        use_pynccl=False, use_mscclpp=False, use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False, use_hpu_communicator=False,
        use_xpu_communicator=False)
    coordinator = parallel_state.GroupCoordinator(
        group_ranks=[list(range(start, start+tp_size)) for start in range(0, world_size, tp_size)],
        group_name='legospec_gemma4_teacher_tp', **options)
    singleton = (coordinator if tp_size == 1 else parallel_state.GroupCoordinator(
        group_ranks=[[rank] for rank in range(world_size)],
        group_name='legospec_gemma4_teacher_singleton', **options))
    for name in ('_WORLD', '_TP', '_ATTN_TP', '_MOE_TP', '_SHARED_EXPERTS_TP'):
        setattr(parallel_state, name, coordinator)
    for name in ('_PP', '_SELF_PP', '_ATTN_CP', '_MOE_EP', '_MOE_DP'):
        setattr(parallel_state, name, singleton)
    get_parallel().override_permanently(
        world_group=coordinator, tp_group=coordinator, pp_group=singleton,
        attn_tp_group=coordinator, attn_cp_group=singleton,
        moe_ep_group=singleton, moe_dp_group=singleton,
        moe_tp_group=coordinator, shared_experts_tp_group=coordinator,
    )
    initialize_dp_attention(server_args)


class Gemma4SGLangTrainingTargetModel(TrainingTargetModel):
    def __init__(self, engine, text_config, capture_layer_ids):
        super().__init__()
        self.engine = engine
        self.model_runner = engine.runner
        self.text_config = text_config
        self._configured_capture_ids = tuple(capture_layer_ids)
        self.capture_layer_ids = list(capture_layer_ids)
        self._components = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, capture_layer_ids,
                        torch_dtype=None, device=None, cache_dir=None,
                        trust_remote_code=False, **kwargs):
        from transformers import AutoConfig
        from .sglang_gemma4_capture import Gemma4Capture
        from litespec.distributed import get_tp_group

        if torch_dtype not in (None, torch.bfloat16):
            raise ValueError('Gemma 4 native training uses bfloat16')
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
            cache_dir=cache_dir, trust_remote_code=trust_remote_code).text_config
        context = int(kwargs.get('context_length') or 4097)
        args = SimpleNamespace(target_model_path=pretrained_model_name_or_path,
            capture_layer_ids=list(capture_layer_ids), capture_hidden_size=config.hidden_size,
            capture_tp_size=dist.get_world_size(get_tp_group()),
            capture_tp_rank=dist.get_rank(get_tp_group()),
            max_length=context-1, sglang_context_length=context,
            sglang_attention_backend=kwargs.get('attention_backend', 'triton'),
            sglang_mem_fraction_static=kwargs.get('mem_fraction_static', .4),
            sglang_max_running_requests=kwargs.get('max_running_requests') or 1,
            sglang_max_total_tokens=kwargs.get('max_total_tokens') or context)
        engine = Gemma4Capture(args, initialize_parallel=initialize_collocated_parallel)
        engine.runner.model.requires_grad_(False).eval()
        return cls(engine, config, capture_layer_ids)

    def set_capture_layers(self, layer_ids):
        if tuple(layer_ids) != self._configured_capture_ids:
            raise ValueError('Gemma 4 capture layers must be supplied before model initialization')
        super().set_capture_layers(list(layer_ids))

    def build_shared_components(self, mask_token_id):
        if not 0 <= mask_token_id < self.text_config.vocab_size:
            raise ValueError('MASK token is outside the Gemma vocabulary')
        if self._components is None:
            from litespec.distributed import get_tp_group
            from .target_utils import SGLangTPEmbeddingAdapter, SGLangTPLMHeadAdapter
            from .target_utils import set_projection_suppress_tokens
            model = self.model_runner.model
            embedding = model.language_model.embed_tokens
            weight = model.lm_head.weight
            vocab, hidden = self.text_config.vocab_size, self.text_config.hidden_size
            tp_size = dist.get_world_size(get_tp_group()) if dist.is_initialized() else 1
            if weight.ndim != 2 or weight.shape[1] != hidden:
                raise ValueError('Unexpected Gemma output embedding shape')
            # Transformer TP does not imply vocabulary TP. Gemma's native
            # scaled nn.Embedding (and its tied head) can be fully replicated.
            # Select the two views independently from their actual storage.
            if embedding.weight.shape[0] < vocab:
                if tp_size == 1:
                    raise ValueError('A vocabulary-sharded embedding requires TP>1')
                embedding = SGLangTPEmbeddingAdapter(embedding, get_tp_group(), mask_token_id)
            if weight.shape[0] < vocab:
                if tp_size == 1:
                    raise ValueError('A vocabulary-sharded LM head requires TP>1')
                head = SGLangTPLMHeadAdapter(model.lm_head, get_tp_group(),
                    softcap=self.text_config.final_logit_softcapping)
            else:
                with torch.device('meta'):
                    head = FlattenedLMHead(hidden, vocab, bias=False,
                        softcap=self.text_config.final_logit_softcapping, softcap_fp32=True)
                # Padded vocabulary rows are never exposed to the draft.
                head.weight = (weight if weight.shape[0] == vocab else
                               nn.Parameter(weight[:vocab], requires_grad=False))
            set_projection_suppress_tokens(head, getattr(model, 'suppress_token_ids', []))
            self._components = SharedTargetEmbeddingsAndHead(
                # Native Gemma4TextScaledWordEmbedding already applies sqrt(H).
                # Sharing the module preserves the scale exactly once.
                embedding, head).requires_grad_(False).eval()
        return self._components

    @torch.no_grad()
    def generate_training_data(self, input_ids, attention_mask, loss_mask, return_logits=True):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError('Gemma prefill expects matching batch-first token IDs and masks')
        width = input_ids.shape[1]
        records = []
        for index, (ids, mask) in enumerate(zip(input_ids.tolist(), attention_mask.tolist())):
            length = width
            while length > 1 and not mask[length-1]:
                length -= 1
            records.append(dict(source_index=index, input_ids=torch.tensor(ids[:length])))
        captured = self.engine.generate_capture_data(records)
        self.engine.last_token_logits = getattr(captured, 'last_token_logits', None)

        def restore(rows):
            # The capture engine uses inference_mode. A fresh tensor outside it
            # is required before trainable projections save teacher features.
            result = rows.new_zeros((len(records), width, rows.shape[-1]))
            for index, (start, stop) in enumerate(zip(captured.token_offsets, captured.token_offsets[1:])):
                result[index, :stop-start].copy_(rows[start:stop])
            return result

        hidden = {layer: restore(captured.hidden_states[layer]) for layer in self.capture_layer_ids}
        final = restore(captured.final_hidden_state)
        logits = self.build_shared_components(0).lm_head(final) if return_logits else None
        return TargetTrainingOutput(hidden_states=hidden, logits=logits, input_ids=input_ids,
            attention_mask=attention_mask, loss_mask=loss_mask, final_hidden_state=final)
