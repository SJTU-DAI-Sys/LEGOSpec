"""Packed prefill using SGLang native aux capture and a LEGOSpec TP mesh."""

import os
from array import array
from types import SimpleNamespace

import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import bootstrap
from sglang.srt.managers.overlap_utils import FutureMap, resolve_forward_inputs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import SpawnRanks, get_schedule, publish
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from legospec.distributed import get_tp_group

from .parallel import initialize_target_parallel


class NativeCapture:
    def __init__(
        self,
        model_path,
        *,
        capture_layer_ids,
        hidden_size,
        adapter,
        torch_dtype=None,
        cache_dir=None,
        trust_remote_code=False,
        **backend_kwargs,
    ):
        if torch_dtype not in (None, torch.bfloat16):
            raise ValueError("Native SGLang capture uses bfloat16")
        self.layer_ids = list(capture_layer_ids)
        self.hidden_size = hidden_size
        self.adapter = adapter
        options = dict(backend_kwargs)
        options["context_length"] = options.get("context_length") or 4097
        options["max_running_requests"] = options.get("max_running_requests") or 1
        options["max_total_tokens"] = (
            options.get("max_total_tokens") or options["context_length"]
        )
        options.setdefault("mem_fraction_static", 0.4)
        options.update(
            disable_custom_all_reduce=True,
            disable_cuda_graph=True,
            disable_prefill_cuda_graph=True,
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            enable_return_hidden_states=True,
        )
        server_args = ServerArgs(
            model_path=model_path,
            dtype="bfloat16",
            tp_size=dist.get_world_size(get_tp_group()),
            trust_remote_code=trust_remote_code,
            download_dir=cache_dir,
            **options,
        )
        gpu_id = torch.cuda.current_device()
        port = int(os.environ["MASTER_PORT"])
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(world_rank=dist.get_rank(get_tp_group()), gpu_id=gpu_id),
        )
        model_config = ModelConfig.from_server_args(server_args)
        # Both online training and offline collection own a LEGOSpec WORLD.
        initialize_target_parallel(server_args=server_args, local_rank=gpu_id)
        bootstrap.init_layer_runtime(model_config=model_config)
        self.runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            nccl_port=port,
            server_args=server_args,
        )
        self.runner.model.requires_grad_(False).eval()
        self.runner.alloc_memory_pool()
        self.runner.init_attention_backends()
        self.adapter.install_capture(self.runner.model, self.layer_ids)
        self.runner.init_cuda_graphs(capture_decode_cuda_graph=False)
        self.future_map = FutureMap(
            device=self.runner.device,
            spec_algo=SpeculativeAlgorithm.NONE,
            req_to_token_pool=self.runner.req_to_token_pool,
            needs_cpu_seq_lens=False,
        )
        self.tree_cache = RadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self.runner.req_to_token_pool,
                token_to_kv_pool_allocator=self.runner.token_to_kv_pool_allocator,
                page_size=get_schedule().page_size,
            )
        )

    @torch.inference_mode()
    def generate_capture_data(self, records):
        requests = []
        offsets = [0]
        sampling = SamplingParams(temperature=0, max_new_tokens=1)
        for record in records:
            ids = array("q", record["input_ids"].reshape(-1).tolist())
            request = Req(
                rid=str(record["source_index"]),
                origin_input_text="",
                origin_input_ids=ids,
                sampling_params=sampling,
                return_hidden_states=True,
                vocab_size=self.runner.model_config.vocab_size,
            )
            request.init_next_round_input()
            request.set_extend_range(0, len(ids))
            requests.append(request)
            offsets.append(offsets[-1] + len(ids))

        batch = ScheduleBatch.init_new(
            reqs=requests,
            req_to_token_pool=self.runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.runner.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        try:
            batch.prepare_for_extend()
            resolve_forward_inputs(batch, self.future_map)
            forward = ForwardBatch.init_new(
                batch,
                self.runner,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                return_hidden_states_before_norm=False,
            )
            teacher = {}

            def capture_teacher(module, inputs):
                teacher["hidden"] = inputs[1]

            # SGLang 0.5.21 returns auxiliary context in output.hidden_states.
            # The normalized teacher is the LM-head input, captured explicitly.
            handle = self.runner.model.logits_processor.register_forward_pre_hook(
                capture_teacher
            )
            try:
                output = self.runner.forward(forward).logits_output
                final = teacher["hidden"]
            finally:
                handle.remove()
        finally:
            self.runner.req_to_token_pool.clear()
            self.runner.token_to_kv_pool_allocator.clear()

        token_count = offsets[-1]
        aux = output.hidden_states
        if aux is None or aux.shape != (
            token_count,
            len(self.layer_ids) * self.hidden_size,
        ):
            raise ValueError(
                "Native SGLang capture did not return all requested decoder layers"
            )
        if final is None or final.shape != (token_count, self.hidden_size):
            raise ValueError(
                "Native SGLang capture did not return the final normalized state"
            )
        viewed = aux.view(token_count, len(self.layer_ids), self.hidden_size)
        return SimpleNamespace(
            hidden_states={
                layer: viewed[:, i] for i, layer in enumerate(self.layer_ids)
            },
            final_hidden_state=final,
            packed=True,
            token_offsets=offsets,
            packed_aux=aux,
            packed_last=final,
            last_token_logits=output.next_token_logits,
        )
