"""Packed text-only Gemma 4 Unified prefill using SGLang's native aux capture."""

from array import array
import os
from types import SimpleNamespace

import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import bootstrap
from sglang.srt.managers.overlap_utils import FutureMap, resolve_forward_inputs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import SpawnRanks, get_device, get_schedule, publish
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class Gemma4Capture:
    def __init__(self, args, *, initialize_parallel=None):
        self.layer_ids = args.capture_layer_ids
        self.hidden_size = args.capture_hidden_size
        server_args = ServerArgs(
            model_path=args.target_model_path,
            dtype="bfloat16",
            tp_size=getattr(args, 'capture_tp_size', 1),
            disable_custom_all_reduce=True,
            attention_backend=args.sglang_attention_backend,
            mem_fraction_static=args.sglang_mem_fraction_static,
            context_length=args.sglang_context_length or args.max_length + 1,
            max_running_requests=args.sglang_max_running_requests,
            max_total_tokens=args.sglang_max_total_tokens,
            disable_cuda_graph=True,
            disable_prefill_cuda_graph=True,
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            enable_return_aux_hidden_states=True,
            speculative_algorithm="EAGLE3",
            return_aux_hidden_state_layer_ids=self.layer_ids,
        )
        gpu_id = torch.cuda.current_device()
        port = int(os.environ["MASTER_PORT"])
        publish(server_args, role="scheduler", ranks=SpawnRanks(
            world_rank=getattr(args, 'capture_tp_rank', 0), gpu_id=gpu_id))
        model_config = ModelConfig.from_server_args(server_args)
        if initialize_parallel is None:
            bootstrap.init_parallel_runtime(
                server_args=server_args, device=get_device().device, dist_port=port
            )
        else:
            # Collocated training already owns the global DP/FSDP WORLD.
            initialize_parallel(server_args=server_args, local_rank=gpu_id)
        bootstrap.init_layer_runtime(model_config=model_config)
        self.runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=args.sglang_mem_fraction_static,
            gpu_id=gpu_id,
            nccl_port=port,
            server_args=server_args,
        )
        self.runner.alloc_memory_pool()
        self.runner.init_attention_backends()
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
            request.extend_end = len(ids)
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
                batch, self.runner,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                return_hidden_states_before_norm=False,
            )
            output = self.runner.forward(forward).logits_output
        finally:
            self.runner.req_to_token_pool.clear()
            self.runner.token_to_kv_pool_allocator.clear()

        token_count = offsets[-1]
        aux, final = output.hidden_states, output.last_hidden_states
        if aux is None or aux.shape != (token_count, len(self.layer_ids) * self.hidden_size):
            raise ValueError("Gemma 4 did not return all requested decoder layers")
        if final is None or final.shape != (token_count, self.hidden_size):
            raise ValueError("Gemma 4 did not return the final normalized state")
        viewed = aux.view(token_count, len(self.layer_ids), self.hidden_size)
        return SimpleNamespace(
            hidden_states={layer: viewed[:, i] for i, layer in enumerate(self.layer_ids)},
            final_hidden_state=final,
            packed=True,
            token_offsets=offsets,
            packed_aux=aux,
            packed_last=final,
            last_token_logits=output.next_token_logits,
        )
