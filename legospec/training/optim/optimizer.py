"""FP32-master parameter updates, AdamW/Muon routing and gradient normalization."""

from __future__ import annotations

import logging
import math

import torch
import torch.distributed as dist

from legospec.distributed import print_on_rank0
from legospec.training.optim.scheduler import CosineAnnealingWarmupLR


def register_optimizer_parameters(model, parameters=None):
    """Classify original parameters before FSDP destroys their matrix shapes."""
    from legospec.training.optim.fsdp_layout import ParameterSpec

    selected = list(
        dict.fromkeys(
            (p for p in model.parameters() if p.requires_grad)
            if parameters is None
            else parameters
        )
    )
    names = {id(p): name for name, p in model.named_parameters()}
    policies = {id(p): set() for p in selected}
    # Muon exceptions for optimizer ownership
    muon_exceptions = {
        "indexer",
        "selector",
        "candidate_selector",
        "markov_head",
        "sequential_head",
        "confidence_head",
    }
    for path, module in model.named_modules(remove_duplicate=False):
        for name, p in module.named_parameters(recurse=False):
            if id(p) not in policies:
                continue
            if (
                muon_exceptions.intersection(path.split("."))
                or isinstance(module, torch.nn.Embedding)
                or "norm" in type(module).__name__.lower()
                or isinstance(module, torch.nn.Conv1d)
                or name
                in ("bias", "base_kernel", "layer_fusion_weights", "A_log", "dt_bias")
            ):
                kind = "adamw"
            elif (
                isinstance(module, torch.nn.Linear) and name == "weight" and p.ndim == 2
            ):
                if not p.is_contiguous():
                    raise ValueError(
                        f"Muon requires contiguous matrix weights: {path}.{name}"
                    )
                kind = "muon"
            else:
                raise ValueError(
                    f"Unclassified optimizer parameter: {path}.{name} {tuple(p.shape)}"
                )
            policies[id(p)].add(kind)
    # Read actual MLA modules, not global config or projection names alone.
    # Composable layers can have different head geometries; DFlare has a second KV path.
    from legospec.modeling.draft.backbone.attention.latent import (
        MLAProjection,
    )

    head_layouts = {}
    for module in model.modules():
        if not isinstance(module, MLAProjection):
            continue
        query = module.q_proj if module.q_lora_rank is None else module.q_b_proj
        projections = [(query, (module.qk_head_dim,))]
        kv_dims = tuple(d for d in (module.qk_nope_head_dim, module.v_head_dim) if d)
        projections.append((module.kv_b_proj, kv_dims))
        if getattr(module, "separate_context", False):
            projections.append((module.context_kv_b_proj, kv_dims))
        for projection, dims in projections:
            geometry = (module.num_heads, dims)
            previous = head_layouts.setdefault(id(projection.weight), geometry)
            if previous != geometry:
                raise ValueError("Conflicting MLA Muon head layouts for tied weight")
    specs = []
    for p in selected:
        choices = policies[id(p)]
        if id(p) not in names or len(choices) != 1:
            raise ValueError(
                f"Missing/conflicting optimizer ownership: {names.get(id(p), '<external>')}"
            )
        kind = choices.pop()
        heads, dims = head_layouts.get(id(p), (0, ())) if kind == "muon" else (0, ())
        specs.append(ParameterSpec(names[id(p)], tuple(p.shape), kind, heads, dims))
    return selected, specs


def optimizer_step_count(phases, steps_per_epoch, accumulation_steps):
    """Each phase flushes its own tail; epochs within a phase do not."""
    return sum(
        math.ceil(
            (
                phase.epochs * steps_per_epoch
                if phase.schedule_full_epochs
                else phase.batch_count(steps_per_epoch)
            )
            / accumulation_steps
        )
        for phase in phases
    )


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        parameters=None,
        process_group=None,
        muon_config=None,
        parameter_specs=None,
        parameter_layouts=None,
    ):
        self.model = model
        self.use_muon = muon_config is not None
        self.muon_optimizer = None
        self.muon_scheduler = None
        if self.use_muon:
            from legospec.training.optim.fsdp_layout import local_layouts
            from legospec.training.optim.muon import validate_muon_config

            muon_config = validate_muon_config(muon_config)
            if parameter_specs is None:
                parameters, parameter_specs = register_optimizer_parameters(
                    model, parameters
                )
            if parameter_layouts is None:
                if dist.is_initialized() and dist.get_world_size(process_group) > 1:
                    raise ValueError("Distributed Muon requires explicit FSDP layouts")
                parameter_layouts = local_layouts(parameter_specs)
            self.parameter_specs = list(parameter_specs)
            self.parameter_layouts = list(parameter_layouts)
        self.model_params = (
            [p for p in model.parameters() if p.requires_grad]
            if parameters is None
            else list(parameters)
        )
        if not self.model_params:
            raise ValueError("BF16Optimizer requires at least one model parameter.")
        parameter_names = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        self.parameter_names = [
            parameter_names[id(parameter)] for parameter in self.model_params
        ]
        if self.use_muon:
            if [s.name for s in self.parameter_specs] != self.parameter_names or [
                l.spec for l in self.parameter_layouts
            ] != self.parameter_specs:
                raise ValueError("Optimizer parameter registration/layout mismatch")
            rank = dist.get_rank(process_group) if dist.is_initialized() else 0
            world = dist.get_world_size(process_group) if dist.is_initialized() else 1
            if any(
                len(layout.shards) != world
                or parameter.numel() != layout.shards[rank][1]
                for parameter, layout in zip(self.model_params, self.parameter_layouts)
            ):
                raise ValueError(
                    "Optimizer local parameter shapes do not match shard layouts"
                )
        self.max_grad_norm = float(max_grad_norm)
        self.process_group = process_group
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}.")
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.adamw_indices = list(range(len(self.fp32_params)))
        self.muon_indices = []
        if self.use_muon:
            from legospec.training.optim.muon import ShardedMuon

            self.adamw_indices = [
                i for i, s in enumerate(self.parameter_specs) if s.optimizer == "adamw"
            ]
            self.muon_indices = [
                i for i, s in enumerate(self.parameter_specs) if s.optimizer == "muon"
            ]
            if len(self.adamw_indices) + len(self.muon_indices) != len(
                self.fp32_params
            ):
                raise ValueError("Unknown optimizer assignment")
            if self.muon_indices:
                self.muon_optimizer = ShardedMuon(
                    [self.fp32_params[i] for i in self.muon_indices],
                    [self.parameter_layouts[i] for i in self.muon_indices],
                    process_group=process_group,
                    **{
                        k: v
                        for k, v in muon_config.items()
                        if not k.startswith("adamw_")
                    },
                )
            marked = [s for s in self.parameter_specs if s.muon_heads]
            if marked and (
                not dist.is_initialized() or dist.get_rank(process_group) == 0
            ):
                logging.getLogger(__name__).info(
                    "MLA per-head Muon %s: %s projections, geometries=%s",
                    "enabled" if muon_config["mla_per_head"] else "disabled",
                    len(marked),
                    sorted({(s.muon_heads, s.muon_head_dims) for s in marked}),
                )
            lr, weight_decay = (
                muon_config["adamw_lr"],
                muon_config["adamw_weight_decay"],
            )
            for kind in ("adamw", "muon"):
                specs = [s for s in self.parameter_specs if s.optimizer == kind]
                if not dist.is_initialized() or dist.get_rank(process_group) == 0:
                    logging.getLogger(__name__).info(
                        "%s: %s tensors, %s parameters",
                        kind,
                        len(specs),
                        f"{sum(math.prod(s.shape) for s in specs):,}",
                    )
        self.optimizer = (
            torch.optim.AdamW(
                [self.fp32_params[i] for i in self.adamw_indices],
                lr=lr,
                weight_decay=weight_decay,
                betas=muon_config["adamw_betas"] if self.use_muon else (0.9, 0.999),
            )
            if self.adamw_indices
            else None
        )
        self.scheduler = (
            CosineAnnealingWarmupLR(
                self.optimizer,
                total_steps=total_steps,
                warmup_steps=int(warmup_ratio * total_steps),
            )
            if self.optimizer is not None
            else None
        )
        if self.muon_optimizer is not None:
            self.muon_scheduler = CosineAnnealingWarmupLR(
                self.muon_optimizer,
                total_steps=total_steps,
                warmup_steps=int(warmup_ratio * total_steps),
            )

    def zero_grad(self) -> None:
        """Clear both FSDP model gradients and FP32 optimizer gradients."""
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        if self.muon_optimizer is not None:
            self.muon_optimizer.zero_grad(set_to_none=True)
        for param in self.model_params:
            param.grad = None
        for param in self.fp32_params:
            param.grad = None

    def scale_model_gradients(self, factor: float) -> None:
        """Scale accumulated FSDP gradients before they are copied to FP32."""
        if factor <= 0:
            raise ValueError(f"gradient scale must be positive, got {factor}.")
        with torch.no_grad():
            for param in self.model_params:
                if param.grad is not None:
                    param.grad.mul_(factor)

    def _global_grad_norm(self) -> torch.Tensor:
        """Return the global L2 norm using one scalar all-reduce.

        Gradients remain sharded and local. Only their local squared L2 norm is
        summed across ranks.
        """
        if not self.fp32_params:
            return torch.tensor(0.0)

        device = self.fp32_params[0].device
        gradients = [param.grad for param in self.fp32_params if param.grad is not None]
        if gradients:
            local_norms = torch._foreach_norm(gradients, 2)
            local_squared_norm = (
                torch.stack([norm.double() for norm in local_norms]).square().sum()
            )
        else:
            local_squared_norm = torch.zeros((), device=device, dtype=torch.float64)

        if dist.is_available() and dist.is_initialized():
            # Communication payload: exactly one 0-D scalar, never gradients.
            dist.all_reduce(
                local_squared_norm,
                op=dist.ReduceOp.SUM,
                group=self.process_group,
            )
        return local_squared_norm.sqrt()

    def step(self) -> float:
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32)
                    if p.grad is not None and (not self.use_muon or p.requires_grad)
                    else None
                )
            if self.use_muon:
                mask = torch.tensor(
                    [p.grad is not None for p in self.fp32_params],
                    dtype=torch.int32,
                    device=self.fp32_params[0].device,
                )
                if dist.is_initialized():
                    dist.all_reduce(
                        mask, op=dist.ReduceOp.MAX, group=self.process_group
                    )
                active = mask.tolist()
                for enabled, p in zip(active, self.fp32_params):
                    if enabled and p.grad is None:
                        p.grad = torch.zeros_like(p)

        grad_norm = self._global_grad_norm()
        grad_norm_value = float(grad_norm.item())
        if not math.isfinite(grad_norm_value):
            raise FloatingPointError(
                "Non-finite gradient norm; optimizer state was not advanced"
            )
        clip_coefficient = min(
            self.max_grad_norm / (grad_norm_value + 1e-6),
            1.0,
        )
        if clip_coefficient < 1.0:
            with torch.no_grad():
                for param in self.fp32_params:
                    if param.grad is not None:
                        param.grad.mul_(clip_coefficient)

        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
        if self.muon_optimizer is not None:
            self.muon_optimizer.step(active=[active[i] for i in self.muon_indices])
            self.muon_optimizer.zero_grad(set_to_none=True)
            self.muon_scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.copy_(mp)
                p.grad = None
        return grad_norm_value

    def load_state_dict(self, state_dict) -> bool:
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
        return True

    def state_dict(self):
        if self.use_muon:
            optimizer_state = {
                "adamw": self.optimizer.state_dict() if self.optimizer else None,
                "muon": (
                    self.muon_optimizer.state_dict() if self.muon_optimizer else None
                ),
            }
            scheduler_state = {
                "adamw": self.scheduler.state_dict() if self.scheduler else None,
                "muon": (
                    self.muon_scheduler.state_dict() if self.muon_scheduler else None
                ),
            }
        else:
            optimizer_state = self.optimizer.state_dict()
            scheduler_state = self.scheduler.state_dict()
        return {
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "fp32_master_params": [p.detach().cpu().clone() for p in self.fp32_params],
        }

    def get_learning_rate(self):
        return (self.optimizer or self.muon_optimizer).param_groups[0]["lr"]

    def get_learning_rates(self):
        return {
            name: opt.param_groups[0]["lr"]
            for name, opt in (("adamw", self.optimizer), ("muon", self.muon_optimizer))
            if opt is not None
        }


def normalize_accumulated_gradients(
    optimizer,
    local_denominator: torch.Tensor,
    accumulation_steps: int,
    group=None,
) -> float:
    """Normalize rank-averaged gradients by the global optimizer-window weight.

    Every microbatch backpropagates its additive loss numerator divided by the
    configured accumulation length. FSDP averages those gradients across ranks.
    Multiplying by ``world_size * accumulation_steps / global_denominator``
    therefore produces the gradient of the summed numerator divided by the sum
    of effective weights across every microbatch and DP rank in the window.
    """
    if int(accumulation_steps) <= 0:
        raise ValueError("accumulation_steps must be positive")
    denominator = local_denominator.detach().float().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group=group)
        if world_size > 1:
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=group)
    denominator_value = float(denominator.item())
    if not math.isfinite(denominator_value) or denominator_value <= 0:
        raise ValueError("global loss denominator must be finite and positive")
    scale = world_size * int(accumulation_steps) / denominator_value
    optimizer.scale_model_gradients(scale)
    return denominator_value
