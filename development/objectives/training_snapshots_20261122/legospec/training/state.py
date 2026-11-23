from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.utils.data import DataLoader
    from legospec.modeling.target.target_utils import (
        TargetEmbeddingsAndHead,
        SharedTargetEmbeddingsAndHead,
    )
    from legospec.modeling.target.training_target_model import TrainingTargetModel
    from legospec.training.optimizer import BF16Optimizer
    from legospec.training.logging import Tracker
    from legospec.distributed.groups import DisaggregatedTopology
    from legospec.distributed.transport import RouteFragment


@dataclass(frozen=True)
class Phase:
    name: str
    phase_id: int
    epochs: int
    max_batches: int | None = None
    log_at_schedule_end: bool = True
    schedule_full_epochs: bool = False
    checkpoint_prefix: str = ""
    final_checkpoint: str = "final"

    def batch_count(self, steps: int) -> int:
        count = self.epochs * steps
        return count if self.max_batches is None else min(count, self.max_batches)


@dataclass
class RunState:
    phase_index: int = 0
    epoch: int = 0
    next_batch: int = 0
    phase_step: int = 0
    global_step: int = 0
    optimizer_step: int = 0


@dataclass
class TrainingSession:
    """Runtime container for one training run.
    Holds configuration, cursor, draft/target models, FSDP training step, data,
    optimizer, tracker and distributed topology. Shared training reads dependencies from here
    instead of implicit globals.
    """

    args: Any
    phases: list[Phase]
    state: RunState = field(default_factory=RunState)
    draft: Any = None
    draft_online: Any = None
    fsdp_model: FullyShardedDataParallel | None = None
    components: TargetEmbeddingsAndHead | SharedTargetEmbeddingsAndHead | None = None
    target: TrainingTargetModel | None = None
    dataloader: DataLoader | None = None
    optimizer: BF16Optimizer | None = None
    tracker: Tracker | None = None
    topology: DisaggregatedTopology | None = None
    routes: list[RouteFragment] | None = None
    packet_metadata: dict[str, Any] = field(default_factory=dict)
    resume_state: dict[str, Any] | None = None
    steps_per_epoch: int = 0
    total_steps: int = 0
    tp_draft_rank: int | None = None
    training_yaml: str | None = field(
        default_factory=lambda: os.environ.get("LEGOSPEC_TRAINING_YAML")
    )

    @property
    def process_group(self) -> ProcessGroup | None:
        return self.topology.draft_group if self.topology is not None else None

    @property
    def coordinator_rank(self) -> int:
        return self.topology.draft_global_ranks[0] if self.topology is not None else 0

    @property
    def phase(self) -> Phase:
        return self.phases[self.state.phase_index]
