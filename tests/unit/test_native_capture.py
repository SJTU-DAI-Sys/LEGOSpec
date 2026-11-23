"""Exercise the native packed engine and its online/offline entrypoints."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from legospec.modeling.target.adapters.gemma import GemmaTargetAdapter
from legospec.modeling.target.sglang import SGLangTrainingTargetModel
from legospec.modeling.target.training_target_model import get_training_target_model


@pytest.fixture
def native_module():
    class ServerArgs(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(page_size=1, **kwargs)

    class Request(SimpleNamespace):
        def set_extend_range(self, start, end):
            self.extend_range = (start, end)

        def init_next_round_input(self):
            self.fill_ids = self.origin_input_ids

    class Batch(SimpleNamespace):
        def prepare_for_extend(self):
            self.prepared = True

    class Processor(nn.Module):
        def forward(self, ids, hidden):
            return hidden

    class Runner:
        def __init__(self, *, model_config, server_args, **kwargs):
            self.model = nn.Linear(4, 4)
            self.model.logits_processor = Processor()
            self.model.set_eagle3_layers_to_capture = lambda ids: setattr(
                self.model, "capture_ids", ids
            )
            self.model_config, self.server_args = model_config, server_args
            self.device = "cpu"
            self.req_to_token_pool = SimpleNamespace(clear=Mock())
            self.token_to_kv_pool_allocator = SimpleNamespace(clear=Mock())
            self.fail = False
            self.alloc_memory_pool = Mock()
            self.init_attention_backends = Mock()
            self.init_cuda_graphs = Mock()

        def forward(self, batch):
            assert batch.prepared
            if self.fail:
                raise RuntimeError("prefill failed")
            # Separate causal sums ensure adjacent requests never share context.
            values = torch.cat(
                [torch.tensor(list(r.origin_input_ids)).cumsum(0) for r in batch.reqs]
            )
            values = values.float()[:, None].expand(-1, 4)
            aux = torch.cat(
                [values + 10 * layer for layer in self.model.capture_ids],
                -1,
            )
            self.model.logits_processor(None, values + 100)
            return SimpleNamespace(
                logits_output=SimpleNamespace(
                    hidden_states=aux,
                    next_token_logits=torch.ones(len(batch.reqs), 32),
                )
            )

    publish = Mock()
    shim = {
        "sglang.srt.configs.model_config": SimpleNamespace(
            ModelConfig=SimpleNamespace(
                from_server_args=lambda args: SimpleNamespace(vocab_size=32)
            )
        ),
        "sglang.srt.distributed": SimpleNamespace(
            bootstrap=SimpleNamespace(init_layer_runtime=Mock())
        ),
        "sglang.srt.managers.overlap_utils": SimpleNamespace(
            FutureMap=lambda **kwargs: SimpleNamespace(**kwargs),
            resolve_forward_inputs=lambda batch, future: None,
        ),
        "sglang.srt.managers.schedule_batch": SimpleNamespace(
            Req=Request,
            ScheduleBatch=SimpleNamespace(init_new=lambda **kwargs: Batch(**kwargs)),
        ),
        "sglang.srt.mem_cache.cache_init_params": SimpleNamespace(
            CacheInitParams=SimpleNamespace
        ),
        "sglang.srt.mem_cache.radix_cache": SimpleNamespace(
            RadixCache=lambda params: params
        ),
        "sglang.srt.model_executor.forward_batch_info": SimpleNamespace(
            CaptureHiddenMode=SimpleNamespace(FULL=1),
            ForwardBatch=SimpleNamespace(
                init_new=lambda batch, runner, **kwargs: batch
            ),
        ),
        "sglang.srt.model_executor.model_runner": SimpleNamespace(ModelRunner=Runner),
        "sglang.srt.runtime_context": SimpleNamespace(
            SpawnRanks=SimpleNamespace,
            get_schedule=lambda: SimpleNamespace(page_size=1),
            publish=publish,
        ),
        "sglang.srt.sampling.sampling_params": SimpleNamespace(
            SamplingParams=SimpleNamespace
        ),
        "sglang.srt.server_args": SimpleNamespace(ServerArgs=ServerArgs),
        "sglang.srt.speculative.spec_info": SimpleNamespace(
            SpeculativeAlgorithm=SimpleNamespace(NONE=0)
        ),
    }
    path = (
        Path(__file__).parents[2] / "legospec/modeling/target/sglang_backend/capture.py"
    )
    name = "legospec.modeling.target.sglang_backend.capture_fixture"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", shim):
        spec.loader.exec_module(module)
    module.publish = publish
    with (
        patch.object(module, "initialize_target_parallel") as initialize,
        patch.object(module, "get_tp_group", return_value="target-tp"),
        patch.object(module.dist, "get_world_size", return_value=2),
        patch.object(module.dist, "get_rank", return_value=1),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.dict("os.environ", {"MASTER_PORT": "29599"}),
    ):
        module.initialize = initialize
        yield module


def make_engine(module):
    return module.NativeCapture(
        "any-native-model",
        capture_layer_ids=[1, 3],
        hidden_size=4,
        adapter=GemmaTargetAdapter(),
        torch_dtype=torch.bfloat16,
        cache_dir="weights",
        trust_remote_code=True,
        attention_backend="triton",
        context_length=64,
        max_running_requests=2,
        max_total_tokens=128,
        mem_fraction_static=0.3,
    )


def test_native_engine_preserves_initialization_and_request_boundaries(native_module):
    engine = make_engine(native_module)
    args = engine.runner.server_args
    assert (
        args.tp_size == 2 and args.download_dir == "weights" and args.trust_remote_code
    )
    assert engine.runner.model.capture_ids == [1, 3]
    assert args.context_length == 64 and args.max_total_tokens == 128
    assert args.disable_cuda_graph and args.enable_return_hidden_states
    native_module.initialize.assert_called_once_with(server_args=args, local_rank=0)
    assert native_module.publish.call_args.kwargs["ranks"].world_rank == 1
    assert all(not p.requires_grad for p in engine.runner.model.parameters())
    records = [
        dict(source_index=2, input_ids=torch.tensor([1, 2])),
        dict(source_index=9, input_ids=torch.tensor([4, 5, 6])),
    ]
    output = engine.generate_capture_data(records)
    assert output.token_offsets == [0, 2, 5] and output.packed
    expected = torch.tensor([1, 3, 4, 9, 15]).float()[:, None].expand(-1, 4)
    torch.testing.assert_close(output.hidden_states[1], expected + 10)
    torch.testing.assert_close(output.hidden_states[3], expected + 30)
    torch.testing.assert_close(output.final_hidden_state, expected + 100)
    assert output.packed_aux.data_ptr() == output.hidden_states[1].data_ptr()
    assert output.packed_last is output.final_hidden_state
    engine.runner.req_to_token_pool.clear.assert_called_once()
    engine.runner.token_to_kv_pool_allocator.clear.assert_called_once()


def test_native_engine_releases_pools_on_forward_failure(native_module):
    engine = make_engine(native_module)
    engine.runner.fail = True
    with pytest.raises(RuntimeError, match="prefill failed"):
        engine.generate_capture_data(
            [dict(source_index=0, input_ids=torch.tensor([1, 2]))]
        )
    engine.runner.req_to_token_pool.clear.assert_called_once()
    engine.runner.token_to_kv_pool_allocator.clear.assert_called_once()


def test_same_factory_serves_packed_collection_and_padded_training(native_module):
    config = SimpleNamespace(
        model_type="gemma4_unified_text", num_hidden_layers=4, hidden_size=4
    )
    with (
        patch(
            "legospec.modeling.config_utils.load_text_model_config", return_value=config
        ),
        patch.dict(
            "sys.modules",
            {"legospec.modeling.target.sglang_backend.capture": native_module},
        ),
    ):
        target = get_training_target_model(
            "fixture", backend="sglang", capture_layer_ids=[3, 1], context_length=64
        )
    assert isinstance(target, SGLangTrainingTargetModel)
    assert isinstance(target.adapter, GemmaTargetAdapter)
    ids = torch.tensor([[1, 2, 0], [4, 5, 6]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    online = target.generate_training_data(ids, mask, mask, return_logits=False)
    offline = target.generate_capture_data(
        [dict(source_index=i, input_ids=ids[i, : int(mask[i].sum())]) for i in range(2)]
    )
    for layer in [1, 3]:
        torch.testing.assert_close(
            online.hidden_states[layer][mask.bool()], offline.hidden_states[layer]
        )
        assert not torch.is_inference(online.hidden_states[layer])
    torch.testing.assert_close(
        online.final_hidden_state[mask.bool()], offline.final_hidden_state
    )
    projection = nn.Linear(4, 2)
    projection(online.hidden_states[1]).sum().backward()
    assert projection.weight.grad is not None


@pytest.mark.parametrize("model_type", ["gemma4_unified_text", "qwen3"])
def test_offline_collector_uses_shared_loader_and_distributed_setup(
    tmp_path, model_type
):
    import datasets
    from transformers import AutoTokenizer

    import legospec.args as arguments
    import legospec.distributed as distributed
    import legospec.modeling.target as target_api
    from legospec.data import capture_engine
    from legospec.data.capture_engine import run_generation

    data = tmp_path / "prompts.json"
    data.write_text("[]")
    config = SimpleNamespace(model_type=model_type, num_hidden_layers=4, hidden_size=4)
    args = SimpleNamespace(
        target_model_path="fixture",
        model_download_dir="model-cache",
        trust_remote_code=True,
        target_model_backend="sglang",
        recompute_gemma4_final_norm=False,
        tp_size=2,
        dist_timeout=30,
        output_path=str(tmp_path / "output"),
        data_path=str(data),
        target_layer_ids=[1],
        max_length=63,
        batch_size=2,
        resume=False,
    )
    tokenizer = SimpleNamespace(pad_token_id=0)
    options = SimpleNamespace(
        to_kwargs=lambda: dict(
            context_length=None, max_running_requests=None, max_total_tokens=None
        )
    )
    with (
        patch(
            "legospec.modeling.config_utils.load_text_model_config", return_value=config
        ),
        patch.object(distributed, "init_distributed") as initialize,
        patch.object(distributed, "destroy_distributed") as destroy,
        patch.object(distributed, "get_dp_group", return_value="dp"),
        patch.object(capture_engine.dist, "get_rank", return_value=0),
        patch.object(capture_engine.dist, "get_world_size", return_value=1),
        patch.object(capture_engine.dist, "new_group", return_value="cpu"),
        patch.object(capture_engine.dist, "barrier"),
        patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer),
        patch.object(arguments.SGLangBackendArgs, "from_args", return_value=options),
        patch.object(
            target_api, "get_training_target_model", return_value=object()
        ) as load,
        patch.object(
            datasets,
            "load_dataset",
            side_effect=RuntimeError("stop after construction"),
        ),
    ):
        with pytest.raises(RuntimeError, match="stop after construction"):
            run_generation(args, None)
    initialize.assert_called_once_with(timeout=30, tp_size=2)
    destroy.assert_called_once()
    load.assert_called_once_with(
        "fixture",
        backend="sglang",
        capture_layer_ids=[1, 3],
        torch_dtype=torch.bfloat16,
        device=None,
        cache_dir="model-cache",
        trust_remote_code=True,
        context_length=64,
        max_running_requests=2,
        max_total_tokens=126,
    )
