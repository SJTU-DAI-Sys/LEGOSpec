"""Resolve shared execution defaults before models or workers are created.

The mode controls packing/projection/planning in the composable adapter.
Correctness fixes and bounded diagnostic kernels remain shared by both modes.
"""


def resolve_execution_options(args):
    mode = getattr(args, "training_mode", "legacy")
    if mode not in ("legacy", "optimized"):
        raise ValueError("Unknown training execution mode: " + str(mode))
    optimized = mode == "optimized"
    offline = bool(getattr(args, "train_hidden_states_path", None))
    collocated = not getattr(args, "disaggregate", False)
    defaults = {
        "hidden_cache_selective_read": optimized,
        "cpu_batch_prefetch": 2 if optimized and offline and collocated else 0,
        "async_schedule_planning": optimized and offline and collocated,
    }
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    if args.cpu_batch_prefetch < 0:
        raise ValueError("cpu-batch-prefetch must be nonnegative")
    if args.cpu_batch_prefetch and not (offline and collocated):
        raise ValueError("cpu-batch-prefetch requires collocated offline training")
    if args.async_schedule_planning and not (optimized and offline and collocated):
        raise ValueError("async-schedule-planning requires optimized collocated offline training")
    if int(getattr(args, "dataloader_prefetch_factor", 2)) < 1:
        raise ValueError("dataloader-prefetch-factor must be positive")
    # W stays 1 unless explicitly requested: W>1 changes update composition.
    return args
