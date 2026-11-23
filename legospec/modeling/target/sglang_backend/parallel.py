"""TP-scoped native SGLang groups sharing a LEGOSpec distributed WORLD."""

import torch.distributed as dist


def initialize_target_parallel(*, server_args, local_rank):
    """Give each teacher a TP-scoped SGLang world; preserve the trainer WORLD.

    Collocated ranks create groups collectively. Disaggregated target ranks
    reuse the groups created before target/draft role divergence.
    SGLang owns these new groups, never the FSDP/optimizer process groups.
    """
    from sglang.srt.distributed import parallel_state
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.runtime_context import get_parallel

    from legospec.distributed import get_disaggregated_topology, get_tp_group

    if not dist.is_initialized():
        raise RuntimeError(
            "Initialize training distributed groups before the SGLang teacher"
        )
    topology = get_disaggregated_topology()
    if topology is not None and not topology.is_target:
        raise ValueError("Only target ranks initialize the SGLang runtime")
    tp_size = dist.get_world_size(get_tp_group())
    world_size = dist.get_world_size()
    if tp_size != server_args.tp_size:
        raise ValueError("SGLang and trainer TP widths disagree")
    ranks = dist.get_process_group_ranks(get_tp_group())
    expected = list(
        range(
            (dist.get_rank() // tp_size) * tp_size,
            (dist.get_rank() // tp_size + 1) * tp_size,
        )
    )
    if topology is None and ranks != expected:
        raise ValueError(
            "Native SGLang capture requires the trainer contiguous TP mesh"
        )
    if parallel_state._WORLD is not None:
        raise RuntimeError("A SGLang world already exists in this process")
    options = dict(
        local_rank=local_rank,
        torch_distributed_backend=dist.get_backend(),
        use_pynccl=False,
        use_mscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
    )
    if topology is None:
        coordinator = parallel_state.GroupCoordinator(
            group_ranks=[
                list(range(start, start + tp_size))
                for start in range(0, world_size, tp_size)
            ],
            group_name="legospec_target_tp",
            **options,
        )
        singleton = (
            coordinator
            if tp_size == 1
            else parallel_state.GroupCoordinator(
                group_ranks=[[rank] for rank in range(world_size)],
                group_name="legospec_target_singleton",
                **options,
            )
        )
        ep, moe_tp = singleton, coordinator
    else:
        if server_args.ep_size != topology.target_ep_size:
            raise ValueError("SGLang and trainer expert parallel widths disagree")
        coordinator = _coordinator_from_existing_group(
            topology.target_tp_group, topology.target_tp_cpu_group, local_rank, "tp"
        )
        singleton = _coordinator_from_existing_group(
            topology.target_singleton_group,
            topology.target_singleton_cpu_group,
            local_rank,
            "singleton",
        )
        ep = _coordinator_from_existing_group(
            topology.target_moe_ep_group,
            topology.target_moe_ep_cpu_group,
            local_rank,
            "ep",
        )
        moe_tp = _coordinator_from_existing_group(
            topology.target_moe_tp_group,
            topology.target_moe_tp_cpu_group,
            local_rank,
            "moe_tp",
        )
    groups = dict(
        world_group=coordinator,
        tp_group=coordinator,
        pp_group=singleton,
        attn_tp_group=coordinator,
        attn_cp_group=singleton,
        moe_ep_group=ep,
        moe_dp_group=singleton,
        moe_tp_group=moe_tp,
        shared_experts_tp_group=coordinator,
    )
    for name, value in {
        "_WORLD": coordinator,
        "_TP": coordinator,
        "_ATTN_TP": coordinator,
        "_MOE_TP": moe_tp,
        "_SHARED_EXPERTS_TP": coordinator,
        "_PP": singleton,
        "_SELF_PP": singleton,
        "_ATTN_CP": singleton,
        "_MOE_EP": ep,
        "_MOE_DP": singleton,
    }.items():
        setattr(parallel_state, name, value)
    get_parallel().override_permanently(**groups)
    initialize_dp_attention(server_args)


def _coordinator_from_existing_group(group, cpu_group, local_rank: int, name: str):
    """Wrap groups collectively created before target/draft role divergence."""
    import os

    import torch
    from sglang.srt.distributed import parallel_state
    from sglang.srt.distributed.parallel_state import GroupCoordinator

    if group is None or cpu_group is None:
        raise RuntimeError(f"missing pre-created SGLang {name} process group")
    coordinator = object.__new__(GroupCoordinator)
    coordinator.unique_name = parallel_state._get_unique_name(f"legospec_{name}")
    parallel_state._register_group(coordinator)
    coordinator.rank = dist.get_rank()
    coordinator.local_rank = local_rank
    coordinator.device_group = group
    coordinator.cpu_group = cpu_group
    coordinator.local_size = int(os.environ.get("LOCAL_SIZE", "0"))
    coordinator.device = torch.device("cuda", local_rank)
    coordinator.device_module = torch.cuda
    coordinator.ranks = dist.get_process_group_ranks(group)
    coordinator.world_size = dist.get_world_size(group)
    coordinator.rank_in_group = dist.get_rank(group)
    coordinator.active_ranks = torch.ones(
        coordinator.world_size, dtype=torch.int32, device=coordinator.device
    )
    coordinator.active_ranks_cpu = torch.ones(coordinator.world_size, dtype=torch.int32)
    coordinator.use_pynccl = False
    coordinator.pynccl_use_current_stream = False
    coordinator.use_mscclpp = False
    coordinator.use_custom_allreduce = False
    coordinator.use_torch_symm_mem_all_reduce = False
    coordinator.use_hpu_communicator = False
    coordinator.use_xpu_communicator = False
    coordinator.use_npu_communicator = False
    coordinator.use_message_queue_broadcaster = False
    coordinator.pynccl_comm = None
    coordinator.pymscclpp_comm = None
    coordinator._fi_workspace_hint = None
    coordinator.pcie_ipc_comm = None
    coordinator.debug_check_symmetric_mempool = lambda *args, **kwargs: None
    coordinator.ca_comm = None
    coordinator.qr_comm = None
    coordinator.torch_symm_mem_comm = None
    coordinator.hpu_communicator = None
    coordinator.xpu_communicator = None
    coordinator.npu_communicator = None
    coordinator.mq_broadcaster = None
    coordinator.is_symmetric_memory_enabled = lambda *args, **kwargs: False
    coordinator.use_symmetric_memory = lambda *args, **kwargs: False
    coordinator.is_allocation_symmetric = lambda *args, **kwargs: False
    return coordinator
