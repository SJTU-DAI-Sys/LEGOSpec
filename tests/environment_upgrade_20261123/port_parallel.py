from pathlib import Path
p=Path("legospec/modeling/target/sglang_backend/parallel.py");s=p.read_text()
s=s.replace("def initialize_collocated_parallel", "def initialize_target_parallel").replace('    if get_disaggregated_topology() is not None:\n        raise ValueError("Native SGLang capture currently requires collocated training")','    topology = get_disaggregated_topology()\n    if topology is not None and not topology.is_target:\n        raise ValueError("Only target ranks initialize the SGLang runtime")')
s=s.replace("    if world_size % tp_size or tp_size != server_args.tp_size:","    if tp_size != server_args.tp_size:")
s=s.replace("    if ranks != expected:", "    if topology is None and ranks != expected:")
s=s.replace('        use_npu_communicator=False,','        use_npu_communicator=False,\n        use_local_synchronization=topology is not None,')
s=s.replace('        group_ranks=[\n            list(range(start, start + tp_size))\n            for start in range(0, world_size, tp_size)\n        ],','        group_ranks=[ranks] if topology is not None else [\n            list(range(start, start + tp_size))\n            for start in range(0, world_size, tp_size)\n        ],')
s=s.replace("group_ranks=[[rank] for rank in range(world_size)],","group_ranks=[[dist.get_rank()]] if topology is not None else [[rank] for rank in range(world_size)],")
a=s.index('    for name in ("_WORLD"');s=s[:a]+'''    ep, moe_tp = singleton, coordinator
    if topology is not None:
        if server_args.ep_size != topology.target_ep_size:
            raise ValueError("SGLang and trainer expert parallel widths disagree")
        if topology.target_ep_size > 1:
            ep = parallel_state.GroupCoordinator(
                group_ranks=[dist.get_process_group_ranks(topology.target_moe_ep_group)],
                group_name="legospec_target_ep", **options,
            )
            moe_tp = parallel_state.GroupCoordinator(
                group_ranks=[dist.get_process_group_ranks(topology.target_moe_tp_group)],
                group_name="legospec_target_moe_tp", **options,
            )
    groups = dict(
        world_group=coordinator, tp_group=coordinator, pp_group=singleton,
        attn_tp_group=coordinator, attn_cp_group=singleton,
        moe_ep_group=ep, moe_dp_group=singleton, moe_tp_group=moe_tp,
        shared_experts_tp_group=coordinator,
    )
    for name, value in {
        "_WORLD":coordinator, "_TP":coordinator, "_ATTN_TP":coordinator,
        "_MOE_TP":moe_tp, "_SHARED_EXPERTS_TP":coordinator,
        "_PP":singleton, "_SELF_PP":singleton, "_ATTN_CP":singleton,
        "_MOE_EP":ep, "_MOE_DP":singleton,
    }.items():
        setattr(parallel_state, name, value)
    get_parallel().override_permanently(**groups)
    initialize_dp_attention(server_args)
'''
s=s.replace('    All trainer ranks create every subgroup in the same collective order.','    Collocated ranks create groups collectively. Disaggregated target ranks\n    create only local-synchronization subgroups, without waiting for draft ranks.')
p.write_text(s)
for file in ["legospec/modeling/target/sglang_backend/capture.py","tests/unit/test_native_capture.py"]:
 p=Path(file);p.write_text(p.read_text().replace("initialize_collocated_parallel","initialize_target_parallel"))
