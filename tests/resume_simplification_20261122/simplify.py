from pathlib import Path
import ast
R=Path(__file__).resolve().parents[2]
T=R/'legospec/training'
def change(path,fn):
 p=R/path;s=fn(p.read_text());ast.parse(s);p.write_text(s)
def remove(s,pred):
 lines=s.splitlines(True);ranges=[]
 for n in ast.walk(ast.parse(s)):
  if pred(n):ranges.append((min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1,n.end_lineno))
 chosen=[]
 for a,b in sorted(ranges,key=lambda x:(x[0],-x[1])):
  if chosen and a<chosen[-1][1]:continue
  chosen.append((a,b))
 for a,b in reversed(chosen):lines[a:b]=[]
 return ''.join(lines)
def fnreplace(s,name,replacement):
 n=next(n for n in ast.walk(ast.parse(s)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
 lines=s.splitlines(True);lines[n.lineno-1:n.end_lineno]=[replacement+'\n'];return ''.join(lines)
def const(n,value):return any(isinstance(x,ast.Constant) and x.value==value for x in ast.walk(n))

def checkpoint(s):
 names={'validate_session_resume','training_identity','validate_training_identity','topology_identity','validate_resume_topology','distributed_training_state_exists','_rank_and_world_size'}
 s=remove(s,lambda n:isinstance(n,ast.FunctionDef) and n.name in names)
 s=remove(s,lambda n:isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'_DISTRIBUTED_STATE_KEY','_DISTRIBUTED_STATE_FORMAT','_OPTIMIZER_STATE_KEY'} for t in n.targets))
 s=fnreplace(s,'save_distributed_training_state','''def save_distributed_training_state(
    checkpoint_dir: str,
    state: dict[str, Any],
    *,
    rank: int | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> None:
    """Save each rank's complete state, then publish common metadata."""
    if rank is None:
        rank = dist.get_rank(process_group) if dist.is_initialized() else 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    _atomic_torch_save(state, ranked_training_state_path(checkpoint_dir, rank))
    if dist.is_initialized():
        dist.barrier(group=process_group)
    if rank == 0:
        common = {
            key: value for key, value in state.items()
            if key not in {"optimizer_state_dict", "fp32_master_params", "rng_state"}
        }
        _atomic_torch_save(common, os.path.join(checkpoint_dir, "training_state.pt"))
    if dist.is_initialized():
        dist.barrier(group=process_group)''')
 s=fnreplace(s,'load_distributed_training_state','''def load_distributed_training_state(
    checkpoint_dir: str,
    *,
    map_location: Any = "cpu",
    rank: int | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, Any]:
    """Load this rank's saved training state directly."""
    if rank is None:
        rank = dist.get_rank(process_group) if dist.is_initialized() else 0
    return torch.load(
        ranked_training_state_path(checkpoint_dir, rank),
        map_location=map_location,
        weights_only=False,
    )''')
 s=fnreplace(s,'load_training_state','''def load_training_state(checkpoint_dir: Optional[str]) -> Optional[dict]:
    if checkpoint_dir is None:
        return None
    return load_distributed_training_state(checkpoint_dir, map_location="cpu")''')
 s=fnreplace(s,'_load_common_state','''def _load_common_state(path: Optional[str]) -> Optional[dict]:
    if path is None:
        return None
    return torch.load(
        os.path.join(path, "training_state.pt"), map_location="cpu", weights_only=False
    )''')
 s=s.replace('from legospec.modeling.draft.composable.loss_config import LOSS_CONTRACT_VERSION\n','')
 s=s.replace('    from legospec.training.data.loader import training_data_identity\n','')
 for line in ['        "train_data_identity": training_data_identity(session.args),\n','        "loss_contract_version": LOSS_CONTRACT_VERSION,\n','        "anchor_sampling_version": "sample_epoch_v1",\n','        metadata["optimizer_kind"] = "muon_adamw_v1"\n','        metadata["execution_contract"] = "composable_packed_v1"\n','        model_metadata.update(training_identity(training_args))\n','        algorithm = draft_model.method_config_key.removesuffix("_config")\n','        training_args=session.args,\n','    training_args,\n']:
  s=s.replace(line,'')
 s=remove(s,lambda n:isinstance(n,ast.If) and ast.unparse(n.test)=='session.topology is not None' and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='topology_identity' for x in ast.walk(n)))
 return s
change('legospec/training/runtime/checkpoint.py',checkpoint)

def setup(s):
 s=remove(s,lambda n:isinstance(n,ast.If) and (const(n.test,'loss_contract_version') or (ast.unparse(n.test)=='session.resume_state')))
 # Remove config and topology identity imports/calls, not native tensor loading.
 s=remove(s,lambda n:isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id in {'validate_session_resume','validate_resume_topology'})
 for line in ['    LOSS_CONTRACT_VERSION,\n','    training_data_identity,\n','    topology_identity,\n','    validate_resume_topology,\n','        validate_session_resume,\n']:
  s=s.replace(line,'')
 s=s.replace('_load_common_state(args.resume_from, args=args)','_load_common_state(args.resume_from)').replace('load_training_state(args.resume_from, args=args)','load_training_state(args.resume_from)')
 # The cursor is runtime state; restoring it needs no compatibility checker.
 a='''        session.optimizer.load_state_dict(session.resume_state)'''
 b='''        from legospec.training.runtime.state import RunState

        session.state = RunState(**session.resume_state["cursor"])
        session.optimizer.load_state_dict(session.resume_state)'''
 assert a in s;s=s.replace(a,b)
 # This only compared the requested override against saved config.
 s=remove(s,lambda n:isinstance(n,ast.If) and 'requested_width' in ast.unparse(n.test))
 s=remove(s,lambda n:isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='requested_width' for t in n.targets))
 return s
change('legospec/training/runtime/setup.py',setup)
change('legospec/training/data/loader.py',lambda s:remove(s,lambda n:isinstance(n,ast.FunctionDef) and n.name=='training_data_identity'))

def planning(s):
 s=remove(s,lambda n:isinstance(n,ast.Assign) and any((isinstance(t,ast.Name) and t.id in {'PLAN_VERSION','ROUND_ORDER_VERSION'}) or (isinstance(t,ast.Attribute) and t.attr=='signature') for t in n.targets))
 s=s.replace('key = f"{ROUND_ORDER_VERSION}:{seed}:{phase}:{window_start}"','key = f"{seed}:{phase}:{window_start}"')
 s=remove(s,lambda n:isinstance(n,ast.If) and ('signature' in ast.unparse(n.test) or ast.unparse(n.test)=='not 0 <= self.committed <= self.total_batches' or 'pending[\'start\']' in ast.unparse(n.test) or 'planner.committed != session.state.phase_step' in ast.unparse(n.test)))
 s=s.replace('            round_order_version=ROUND_ORDER_VERSION,\n','').replace('signature=self.signature, committed=self.committed, pending=pending','committed=self.committed, pending=pending')
 return s
change('legospec/training/data/planning.py',planning)
change('legospec/training/data/batch.py',lambda s:s.replace('ANCHOR_SAMPLING_VERSION = "sample_epoch_v1"\n','').replace('[ANCHOR_SAMPLING_VERSION, int(seed), int(epoch), identity]','[int(seed), int(epoch), identity]'))
change('legospec/modeling/draft/composable/loss_config.py',lambda s:s[:s.index('# Checkpoint semantic identity')]+s[s.index('DEFAULT_LOSS_DECAY_GAMMA'):])

def optimizer(s):
 s=remove(s,lambda n:isinstance(n,ast.FunctionDef) and n.name in {'_param_state_compatible','_optimizer_state_compatible'})
 # Metadata used only by preflight comparison is no longer stored or constructed.
 s=remove(s,lambda n:isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='optimizer_config' for t in n.targets))
 s=remove(s,lambda n:isinstance(n,ast.Expr) and 'self.optimizer_config.update(' in ast.unparse(n))
 s=fnreplace(s,'load_state_dict','''    def load_state_dict(self, state_dict) -> bool:
        masters = state_dict["fp32_master_params"]
        with torch.no_grad():
            for model_param, master, saved in zip(
                self.model_params, self.fp32_params, masters, strict=True
            ):
                master.copy_(saved)
                model_param.copy_(master)
        optimizers = state_dict["optimizer_state_dict"]
        schedulers = state_dict["scheduler_state_dict"]
        if self.use_muon:
            for name, optimizer, scheduler in (
                ("adamw", self.optimizer, self.scheduler),
                ("muon", self.muon_optimizer, self.muon_scheduler),
            ):
                if optimizer is not None:
                    optimizer.load_state_dict(optimizers[name])
                    scheduler.load_state_dict(schedulers[name])
        else:
            self.optimizer.load_state_dict(optimizers)
            self.scheduler.load_state_dict(schedulers)
        print_on_rank0("Successfully loaded optimizer and scheduler state.")
        return True''')
 for line in ['                "kind": "muon_adamw_v1",\n','                "layouts": [l.state_dict() for l in self.parameter_layouts],\n','            "optimizer_parameter_names": self.parameter_names,\n','            "optimizer_config": self.optimizer_config,\n']:
  s=s.replace(line,'')
 # No checkpoint compatibility adjustment to the current Muon recipe.
 s=remove(s,lambda n:isinstance(n,ast.If) and 'not any((spec.muon_heads' in ast.unparse(n.test))
 return s
change('legospec/training/optim/optimizer.py',optimizer)

def scheduler(s):
 s=fnreplace(s,'state_dict','''    def state_dict(self):
        return {
            **{key: value for key, value in self.__dict__.items()
               if key not in {"optimizer", "after_scheduler"}},
            "after_scheduler": self.after_scheduler.state_dict(),
        }''')
 s=fnreplace(s,'load_state_dict','''    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict["after_scheduler"])
        super().load_state_dict({
            key: value for key, value in state_dict.items() if key != "after_scheduler"
        })''')
 return s
change('legospec/training/optim/scheduler.py',scheduler)
# Remove tests whose only purpose was rejecting changes to previous-run metadata.
retired={
 'test_execution_modes.py':{'test_resume_cannot_cross_execution_contract'},
 'test_dpal_local_actors.py':{'test_resume_requires_explicit_objective_checkpointing'},
 'test_composable_eagle3.py':{'test_resume_identity_rejects_supervision_mode_changes'},
 'test_training_planning.py':{'test_rejects_topology_change_on_resume'},
}
for name,names in retired.items():change('tests/unit/'+name,lambda s:remove(s,lambda n:isinstance(n,ast.FunctionDef) and n.name in names))
change('tests/unit/test_training_planning.py',lambda s:s.replace('    saved["pending"].pop("round_order_version", None)\n','').replace('    # An already persisted plan is authoritative even if its ordering differs\n    # from the current heuristic (e.g. a checkpoint from the older planner).','    # Resume consumes the saved pending assignments without rebuilding them.'))
print('Removed resume version/identity comparisons and serialized compatibility metadata.')
