import gc
import glob
import json
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch.distributed.nn import functional as dist_nn
from legospec.modeling.config_utils import load_target_model_config


def set_projection_suppress_tokens(head, token_ids):
    """Retain the native model's input-only marker policy in training logits."""
    ids = sorted(set(token_ids))
    if any(not 0 <= token < head.out_features for token in ids):
        raise ValueError("Suppressed token is outside the output vocabulary")
    head.register_buffer(
        "_logits_suppress_indices",
        torch.tensor(ids, dtype=torch.long, device=head.weight.device),
        persistent=False,
    )


class _GlooAllGather(torch.autograd.Function):
    """Avoid the upstream Gloo all-gather backward's global-rank scatter bug."""

    @staticmethod
    def forward(ctx, group, value):
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        outputs = [torch.empty_like(value) for _ in range(dist.get_world_size(group))]
        dist.all_gather(outputs, value, group=group)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # This fallback is for CPU protocol tests; NCCL retains reduce-scatter.
        gradients = torch.stack(grad_outputs)
        dist.all_reduce(gradients, group=ctx.group)
        return None, gradients[ctx.rank].contiguous()


class SGLangTPEmbeddingAdapter(nn.Module):
    """Reuse SGLang's vocab-parallel embedding for TP-sharded draft batches."""

    def __init__(self, embedding: nn.Module, tp_group, mask_token_id: int):
        super().__init__()
        self.embedding = embedding
        self.tp_group = tp_group
        self.tp_rank = dist.get_rank(tp_group)
        self.tp_size = dist.get_world_size(tp_group)
        self.mask_token_id = int(mask_token_id)
        self.vocab_size = int(embedding.org_vocab_size)
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError(
                "DLite vocab_row MASK mode requires an existing SGLang "
                f"embedding row, but mask_token_id={self.mask_token_id} and "
                f"target vocab size={self.vocab_size}."
            )
        self.embedding_dim = int(embedding.embedding_dim)
        self.num_embeddings = self.vocab_size
        self._trace_pending = True

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        trace = self._trace_pending
        if trace:
            print(f"[rank {dist.get_rank()}] TP embedding: enter", flush=True)
        # MASK is a real in-vocabulary token and uses its target
        # embedding row exactly.  Replacing it with the vocabulary mean creates
        # a different teacher/student input distribution.
        invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
        if bool(invalid.any()):
            raise ValueError("Embedding input contains an out-of-range token ID.")
        # Packed ranks have different token counts. Communicate flat, padded
        # IDs, but run the same number of vocab-parallel lookups everywhere.
        flat = input_ids.reshape(-1)
        count = torch.tensor([flat.numel()], device=flat.device, dtype=torch.long)
        counts = [torch.empty_like(count) for _ in range(self.tp_size)]
        dist.all_gather(counts, count, group=self.tp_group)
        width = max(1, max(int(item.item()) for item in counts))
        flat = F.pad(flat, (0, width - flat.numel()), value=self.mask_token_id)
        gathered = [torch.empty_like(flat) for _ in range(self.tp_size)]
        dist.all_gather(gathered, flat, group=self.tp_group)
        selected = None
        for owner, owner_ids in enumerate(gathered):
            owner_embeddings = self.embedding(owner_ids)
            if owner == self.tp_rank:
                selected = owner_embeddings[: input_ids.numel()].reshape(
                    *input_ids.shape, self.embedding_dim
                )
        assert selected is not None
        if trace:
            print(f"[rank {dist.get_rank()}] TP embedding: lookup done", flush=True)
        if trace:
            print(f"[rank {dist.get_rank()}] TP embedding: exit", flush=True)
            self._trace_pending = False
        return selected


class SGLangTPLMHeadAdapter(nn.Module):
    """Reuse SGLang's LM-head shard for different samples on each TP rank."""

    def __init__(self, lm_head: nn.Module, tp_group, *, softcap: float | None = None):
        super().__init__()
        self.sharded_lm_head = lm_head
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.in_features = int(lm_head.embedding_dim)
        self.out_features = int(lm_head.org_vocab_size)
        self._trace_pending = True
        self.softcap = softcap
        mapping_getter = getattr(lm_head, "get_sharded_to_full_mapping", None)
        if callable(mapping_getter):
            mapping = mapping_getter()
        elif getattr(lm_head, "num_added_embeddings", None) == 0:
            # Current SGLang removed the mapping helper for a contiguous base
            # vocabulary. Its padding is a suffix of the gathered vocabulary.
            mapping = None
        else:
            raise ValueError(
                "LM head must expose vocabulary mapping or an unextended base vocabulary"
            )
        self.register_buffer(
            "sharded_to_full",
            (
                None
                if mapping is None
                else torch.tensor(
                    mapping, dtype=torch.long, device=lm_head.weight.device
                )
            ),
            persistent=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        trace = self._trace_pending
        if trace:
            print(f"[rank {dist.get_rank()}] TP LM head: enter", flush=True)
        if hidden_states.size(-1) != self.in_features:
            raise ValueError(
                f"Expected hidden size {self.in_features}, got {hidden_states.size(-1)}."
            )
        output_shape = (*hidden_states.shape[:-1], self.out_features)
        flat_hidden = hidden_states.reshape(-1, self.in_features).contiguous()
        local_rows = flat_hidden.size(0)

        # Different TP ranks own different draft samples, so masking can leave a
        # different number of active proposal rows on each rank.  Autograd-aware
        # all_gather/all_to_all require equal tensor shapes; pad only for the
        # collectives and crop back to the local row count before returning.
        local_count = torch.tensor(
            [local_rows], dtype=torch.int64, device=flat_hidden.device
        )
        gathered_counts = [torch.empty_like(local_count) for _ in range(self.tp_size)]
        dist.all_gather(gathered_counts, local_count, group=self.tp_group)
        max_rows = max(int(count.item()) for count in gathered_counts)
        # A protocol row preserves the collective/autograd path even when
        # every local supervision selection is empty.
        max_rows = max(1, max_rows)
        if local_rows < max_rows:
            flat_hidden = F.pad(flat_hidden, (0, 0, 0, max_rows - local_rows))

        hidden_by_owner = (
            _GlooAllGather.apply(self.tp_group, flat_hidden)
            if dist.get_backend(self.tp_group) == "gloo"
            else dist_nn.all_gather(flat_hidden, group=self.tp_group)
        )
        local_logits = [
            F.linear(owner_hidden, self.sharded_lm_head.weight).contiguous()
            for owner_hidden in hidden_by_owner
        ]
        # Equal padded row counts permit the tensor collective. In addition to
        # avoiding list communication, this avoids PyTorch's Gloo list fallback
        # treating subgroup-local scatter sources as global ranks.
        send = torch.cat(local_logits, dim=0)
        received = dist_nn.all_to_all_single(
            torch.empty_like(send), send, group=self.tp_group
        )
        vocab_shards = received.reshape(self.tp_size, max_rows, -1).unbind(0)
        if trace:
            print(f"[rank {dist.get_rank()}] TP LM head: shards gathered", flush=True)
        gathered_logits = torch.cat(vocab_shards, dim=-1)
        if self.sharded_to_full is not None:
            gathered_logits = gathered_logits.index_select(
                -1, self.sharded_to_full.to(gathered_logits.device)
            )
        gathered_logits = gathered_logits[:local_rows, : self.out_features]
        if trace:
            print(f"[rank {dist.get_rank()}] TP LM head: exit", flush=True)
            self._trace_pending = False
        result = gathered_logits.reshape(output_shape)
        if self.softcap is not None:
            # Native SGLang casts BF16 projections before applying softcap.
            result = torch.tanh(result.float() / self.softcap) * self.softcap
        indices = getattr(self, "_logits_suppress_indices", None)
        if indices is not None and indices.numel():
            result.index_fill_(-1, indices, torch.finfo(result.dtype).min)
        return result

    @property
    def weight(self):
        return self.sharded_lm_head.weight


class SharedTargetEmbeddingsAndHead(nn.Module):
    """Non-owning training view over target-resident embedding/head modules."""

    def __init__(self, embed_tokens: nn.Module, lm_head: nn.Module):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head


class FlattenedLMHead(nn.Linear):
    """Project all token axes with one GEMM, including sliced proposal blocks.

    Keep Linear's parameters/state keys and gradients while matching the online
    TP adapter's contiguous two-dimensional projection input.
    """

    def __init__(
        self, *args, softcap: float | None = None, softcap_fp32: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.softcap = softcap
        self.softcap_fp32 = softcap_fp32

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim == 0 or input.size(-1) != self.in_features:
            raise ValueError(
                f"Expected final hidden dimension {self.in_features}, "
                f"got shape {tuple(input.shape)}."
            )
        output_shape = (*input.shape[:-1], self.out_features)
        flat_hidden = input.reshape(-1, self.in_features).contiguous()
        logits = F.linear(flat_hidden, self.weight, self.bias).reshape(output_shape)
        if self.softcap is not None:
            if self.softcap_fp32:
                logits = logits.float()
            if torch.is_grad_enabled():
                logits = torch.tanh(logits / self.softcap) * self.softcap
            else:
                logits.div_(self.softcap).tanh_().mul_(self.softcap)
        indices = getattr(self, "_logits_suppress_indices", None)
        if indices is not None and indices.numel():
            logits.index_fill_(-1, indices, torch.finfo(logits.dtype).min)
        return logits


class TargetEmbeddingsAndHead(nn.Module):
    """
    Efficiently loads only the embedding layer and lm_head from a pretrained model.
    Handles safetensors slicing and Weight Tying correctly.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        from legospec.modeling.target.adapters.base import get_model_adapter

        adapter = get_model_adapter(config)
        self.embed_tokens = adapter.new_embedding()
        self.lm_head = FlattenedLMHead(
            adapter.config.hidden_size,
            adapter.config.vocab_size,
            bias=False,
            softcap=adapter.softcap,
        )

    @staticmethod
    def _infer_embed_key(model_path: str) -> str:
        """Find the text embedding key without instantiating the target model."""
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))
        if index_files:
            with open(index_files[0], "r", encoding="utf-8") as handle:
                keys = list(json.load(handle).get("weight_map", {}))
        else:
            keys = []
            for filename in glob.glob(os.path.join(model_path, "*.safetensors")):
                with safe_open(filename, framework="pt", device="cpu") as handle:
                    keys.extend(handle.keys())

        candidates = [key for key in keys if key.endswith("embed_tokens.weight")]
        preferred = (
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        )
        for key in preferred:
            if key in candidates:
                return key
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return "model.embed_tokens.weight"
        raise ValueError(
            "Could not choose a text embedding from checkpoint keys: "
            + ", ".join(sorted(candidates))
            + ". Pass --embedding-key explicitly."
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embed_key: Optional[str] = None,
        lm_head_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ) -> "TargetEmbeddingsAndHead":

        # 1. Load Config
        config = load_target_model_config(
            model_path, cache_dir=cache_dir, trust_remote_code=trust_remote_code
        )
        instance = cls(config)

        if lm_head_key is None:
            lm_head_key = "lm_head.weight"

        # 2. Resolve Model Path
        local_model_path = model_path
        if not os.path.exists(local_model_path):
            try:
                local_model_path = snapshot_download(
                    repo_id=model_path,
                    cache_dir=cache_dir,
                    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"],
                )
            except Exception as e:
                print(f"Warning: Snapshot download failed or path check failed: {e}")

        if embed_key is None:
            embed_key = cls._infer_embed_key(local_model_path)

        # 3. Handle Weight Tying
        tie_weights = getattr(config, "tie_word_embeddings", False)

        # 4. Load Weights
        instance._load_weights(local_model_path, embed_key, lm_head_key, tie_weights)

        # 5. Move to Device & Freeze
        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)

        return instance

    def _load_weights(
        self, model_path: str, embed_key: str, lm_head_key: str, tie_weights: bool
    ):
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))
        weight_map = {}
        files_to_load = {}

        if index_files:
            with open(index_files[0], "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})

            if embed_key in weight_map:
                files_to_load[embed_key] = weight_map[embed_key]
            else:
                raise ValueError(
                    f"Embedding key '{embed_key}' not found in weight map."
                )

            if not tie_weights:
                if lm_head_key in weight_map:
                    files_to_load[lm_head_key] = weight_map[lm_head_key]
                else:
                    print(
                        f"Warning: {lm_head_key} not found. Ensure model doesn't use tied weights manually."
                    )
        else:
            safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
            bins = glob.glob(os.path.join(model_path, "*.bin"))
            target_file = safetensors[0] if safetensors else (bins[0] if bins else None)

            if not target_file:
                raise FileNotFoundError("No checkpoint found.")

            files_to_load[embed_key] = os.path.basename(target_file)
            if not tie_weights:
                files_to_load[lm_head_key] = os.path.basename(target_file)

        loaded_keys = set()

        file_to_keys_map = {}
        for key, filename in files_to_load.items():
            full_path = os.path.join(model_path, filename)
            if full_path not in file_to_keys_map:
                file_to_keys_map[full_path] = []
            file_to_keys_map[full_path].append(key)

        for file_path, keys in file_to_keys_map.items():
            self._load_file_content(file_path, keys, embed_key, lm_head_key)
            loaded_keys.update(keys)

        if tie_weights:
            print(
                "Weight tying detected: Sharing weights between Embeddings and LM Head."
            )
            self.lm_head.weight = self.embed_tokens.weight

        if embed_key not in loaded_keys:
            raise RuntimeError("Failed to load embeddings.")
        if not tie_weights and lm_head_key not in loaded_keys:
            raise RuntimeError(
                "LM head weights were not found and the target does not tie its "
                "embedding weights; refusing to use a random target head."
            )

    def _load_file_content(
        self,
        file_path: str,
        keys_to_extract: list,
        target_embed_key: str,
        target_head_key: str,
    ):
        """Helper to load specific keys from a file"""
        print(f"Loading {keys_to_extract} from {os.path.basename(file_path)}...")

        state_dict_part = {}

        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                for k in keys_to_extract:
                    if k in f.keys():
                        state_dict_part[k] = f.get_tensor(k)
        else:
            print(
                f"Warning: Loading .bin file {os.path.basename(file_path)} into RAM. Convert to safetensors for efficiency."
            )
            full_state = torch.load(file_path, map_location="cpu")
            for k in keys_to_extract:
                if k in full_state:
                    state_dict_part[k] = full_state[k]
            del full_state
            gc.collect()

        for k, tensor in state_dict_part.items():
            if k == target_embed_key:
                self.embed_tokens.weight.data.copy_(tensor)
                print(" -> Loaded Embeddings")
            elif k == target_head_key:
                if tensor.shape == self.lm_head.weight.data.shape:
                    self.lm_head.weight.data.copy_(tensor)
                    print(" -> Loaded LM Head")
                else:
                    raise RuntimeError(
                        f"Shape mismatch for {k}. Expected {self.lm_head.weight.shape}, got {tensor.shape}"
                    )


def build_sglang_components(model, adapter, mask_token_id):
    """Share resident modules; vocabulary partitioning is independent of model TP."""
    from legospec.distributed import get_tp_group

    embedding, native_head = adapter.sglang_components(model)
    weight = native_head.weight
    vocab, hidden = adapter.config.vocab_size, adapter.config.hidden_size
    # SGLang's vocab-parallel modules expose org_vocab_size even at TP=1.
    # Replicated native embeddings retain their own forward scaling.
    if hasattr(embedding, "org_vocab_size"):
        embedding = SGLangTPEmbeddingAdapter(embedding, get_tp_group(), mask_token_id)
    if hasattr(native_head, "org_vocab_size"):
        head = SGLangTPLMHeadAdapter(
            native_head, get_tp_group(), softcap=adapter.softcap
        )
    else:
        with torch.device("meta"):
            head = FlattenedLMHead(
                hidden,
                vocab,
                bias=False,
                softcap=adapter.softcap,
                softcap_fp32=adapter.softcap_fp32,
            )
        head.weight = (
            weight
            if weight.shape[0] == vocab
            else nn.Parameter(weight[:vocab], requires_grad=False)
        )
    set_projection_suppress_tokens(head, getattr(model, "suppress_token_ids", []))
    return SharedTargetEmbeddingsAndHead(embedding, head).requires_grad_(False).eval()
