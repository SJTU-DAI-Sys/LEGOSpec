from pathlib import Path
import tarfile
p=Path("legospec/modeling/target/sglang_backend/parallel.py")
s=p.read_text().replace('        use_local_synchronization=topology is not None,\n','')
a=s.index('    coordinator = parallel_state.GroupCoordinator(');b=s.index('    groups = dict(',a)
s=s[:a]+'''    if topology is None:
        coordinator = parallel_state.GroupCoordinator(
            group_ranks=[list(range(start, start + tp_size)) for start in range(0, world_size, tp_size)],
            group_name="legospec_target_tp", **options,
        )
        singleton = coordinator if tp_size == 1 else parallel_state.GroupCoordinator(
            group_ranks=[[rank] for rank in range(world_size)],
            group_name="legospec_target_singleton", **options,
        )
        ep, moe_tp = singleton, coordinator
    else:
        if server_args.ep_size != topology.target_ep_size:
            raise ValueError("SGLang and trainer expert parallel widths disagree")
        coordinator = _coordinator_from_existing_group(
            topology.target_tp_group, topology.target_tp_cpu_group, local_rank, "tp")
        singleton = _coordinator_from_existing_group(
            topology.target_singleton_group, topology.target_singleton_cpu_group, local_rank, "singleton")
        ep = _coordinator_from_existing_group(
            topology.target_moe_ep_group, topology.target_moe_ep_cpu_group, local_rank, "ep")
        moe_tp = _coordinator_from_existing_group(
            topology.target_moe_tp_group, topology.target_moe_tp_cpu_group, local_rank, "moe_tp")
'''+s[b:]
s=s.replace('    create only local-synchronization subgroups, without waiting for draft ranks.', '    reuse the groups created before target/draft role divergence.')
with tarfile.open("tests/environment_upgrade_20261123/source_before.tgz") as t:
 old=t.extractfile("legospec/modeling/target/sglang_backend/patch.py").read().decode()
a=old.index("def _coordinator_from_existing_group");b=old.index("\ndef init_distributed_environment",a)
helper=old[a:b].replace(") -> GroupCoordinator:","):")
helper=helper.replace('    """Wrap groups collectively created before target/draft role divergence."""','''    """Wrap groups collectively created before target/draft role divergence."""
    import os
    import torch
    from sglang.srt.distributed import parallel_state
    from sglang.srt.distributed.parallel_state import GroupCoordinator''')
helper=helper.replace("    coordinator.use_pymscclpp = False","    coordinator.use_mscclpp = False")
helper=helper.replace("    coordinator.ca_comm = None","    coordinator._fi_workspace_hint = None\n    coordinator.pcie_ipc_comm = None\n    coordinator.debug_check_symmetric_mempool = lambda *args, **kwargs: None\n    coordinator.ca_comm = None")
p.write_text(s+"\n\n"+helper)
p=Path("legospec/training/optim/fsdp_layout.py");s=p.read_text();a=s.index('        if torch.__version__');b=s.index('        entries = {}',a);s=s[:a]+s[b:];p.write_text(s)
