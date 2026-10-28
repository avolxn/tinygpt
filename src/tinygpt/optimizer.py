"""Nanochat Muon updates, native AdamW, and FSDP2-compatible optimizer state.

PyTorch Muon lacks nanochat's Polar Express, row equilibration, renormalization,
variance reduction, and cautious decay, so only that update is implemented here.
AdamW, serialization, and distributed tensor communication use PyTorch.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

from tinygpt.utils import compute_dtype

POLAR_EXPRESS_COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def make_param_groups(
    model: nn.Module,
    *,
    matrix_lr: float = 0.02,
    embedding_lr: float = 0.3,
    scalar_lr: float = 0.5,
    lm_head_lr: float = 0.008,
    weight_decay: float = 0.28,
) -> list[dict[str, Any]]:
    """Use nanochat's learning rates, betas, and decay for each parameter family."""
    params = dict(model.named_parameters())
    scale = (cast(Any, model).config.hidden_size / 768) ** -0.5
    specifications = (
        ("lm_head.", lm_head_lr * scale, (0.8, 0.96), 0.01),
        ("embed_tokens.", embedding_lr * scale, (0.8, 0.995), 0.001),
        ("value_embeds.", embedding_lr * scale * 0.5, (0.8, 0.995), 0.01),
        ("resid_lambdas", scalar_lr * 0.01, (0.8, 0.95), 0.05),
        ("x0_lambdas", scalar_lr, (0.96, 0.95), 0.0),
    )
    groups: list[dict[str, Any]] = []
    assigned: set[str] = set()
    for prefix, lr, betas, decay in specifications:
        names = [name for name in params if name.startswith(prefix)]
        if names:
            groups.append(
                dict(
                    kind="adamw",
                    params=[params[name] for name in names],
                    lr=lr,
                    betas=betas,
                    eps=1e-10,
                    weight_decay=decay,
                )
            )
            assigned.update(names)
    smear_names = [name for name in params if name.startswith(("smear_", "backout_"))]
    if smear_names:
        groups.append(
            dict(
                kind="adamw",
                params=[params[name] for name in smear_names],
                lr=0.2,
                betas=(0.8, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            )
        )
        assigned.update(smear_names)
    matrices = [p for name, p in params.items() if name not in assigned]
    if any(p.ndim != 2 for p in matrices):
        raise ValueError("All remaining nanochat block parameters must be matrices")
    groups.append(
        dict(
            kind="muon", params=matrices, lr=matrix_lr, momentum=0.95, beta2=0.9, ns_steps=5, weight_decay=weight_decay
        )
    )
    for group in groups:
        group["initial_lr"] = group["lr"]
        group["initial_weight_decay"] = group["weight_decay"]
    return groups


def muon_update(
    gradient: torch.Tensor,
    momentum_buffer: torch.Tensor,
    variance_buffer: torch.Tensor,
    momentum: float,
    beta2: float,
    ns_steps: int,
) -> torch.Tensor:
    """Compute nanochat's Polar Express and NorMuon update on a complete matrix.

    Args:
        gradient: Full synchronized matrix gradient.
        momentum_buffer: Full first moment, updated in place.
        variance_buffer: Factored second moment, updated in place.
        momentum: Current Nesterov momentum coefficient.
        beta2: Second moment decay.
        ns_steps: Number of Polar Express iterations, from one to five.

    Returns:
        Orthogonalized update with reference variance normalization.
    """
    momentum_buffer.lerp_(gradient, 1 - momentum)
    g = gradient.lerp(momentum_buffer, momentum)
    x = g.bfloat16() if compute_dtype == torch.bfloat16 else g
    target = x.float().norm(dim=(-2, -1), keepdim=True) / math.sqrt(x.size(-2))
    x = x * (target / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(x.dtype)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS[:ns_steps]:
        if x.size(-2) > x.size(-1):
            gram = x.mT @ x
            x = a * x + x @ (b * gram + c * (gram @ gram))
        else:
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    g = x.to(gradient.dtype)
    g = g * (math.sqrt(min(g.shape[-2:])) / g.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)).to(g.dtype)
    red_dim = -1 if g.size(-2) >= g.size(-1) else -2
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    variance_buffer.lerp_(v_mean.to(variance_buffer.dtype), 1 - beta2)
    step_size = variance_buffer.clamp_min(1e-10).rsqrt()
    v_norm = (v_mean.sum() * g.size(red_dim)).sqrt()
    v_norm_new = ((v_mean * g.size(red_dim)) * step_size.float().square()).sum().sqrt()
    return cast(torch.Tensor, g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype))


class MuonAdamW(torch.optim.AdamW):
    """Use native AdamW and nanochat Muon with a single checkpointable state.

    Args:
        param_groups: Parameter families with a per-group optimizer kind.
    """

    def __init__(self, param_groups: list[dict[str, Any]]) -> None:
        super().__init__(param_groups)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        """Update AdamW groups and gather complete Muon matrices under FSDP2."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = self.param_groups
        try:
            self.param_groups = [g for g in groups if g["kind"] == "adamw"]
            super().step()
        finally:
            self.param_groups = groups
        for group in groups:
            if group["kind"] != "muon":
                continue
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                sharded = isinstance(parameter, DTensor)
                gradient = parameter.grad.full_tensor() if sharded else parameter.grad
                full_parameter = parameter.full_tensor() if sharded else parameter
                state = self.state[parameter]
                if not state:
                    state["step"] = torch.tensor(0.0)
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                    rows, cols = parameter.shape
                    state["variance_buffer"] = gradient.new_zeros((rows, 1) if rows >= cols else (1, cols))
                state["step"] += 1
                first = state["momentum_buffer"]
                full_first = first.full_tensor() if sharded else first
                variance = state["variance_buffer"]
                full_variance = variance.full_tensor() if isinstance(variance, DTensor) else variance
                update = muon_update(
                    gradient, full_first, full_variance, group["momentum"], group["beta2"], group["ns_steps"]
                )
                lr = group["lr"] * math.sqrt(max(1.0, parameter.size(-2) / parameter.size(-1)))
                full_parameter.sub_(
                    lr * update + lr * group["weight_decay"] * full_parameter * (update * full_parameter >= 0)
                )
                if sharded:
                    parameter.copy_(distribute_tensor(full_parameter, parameter.device_mesh, parameter.placements))
                    first.copy_(distribute_tensor(full_first, parameter.device_mesh, parameter.placements))
                    if isinstance(variance, DTensor):
                        variance.copy_(distribute_tensor(full_variance, variance.device_mesh, variance.placements))
        return loss


def make_optimizer(model: nn.Module, **kwargs: Any) -> MuonAdamW:
    """Create the reference optimizer from the model's parameter families."""
    return MuonAdamW(make_param_groups(model, **kwargs))


def get_lr_multiplier(
    step: int,
    num_steps: int,
    warmup_steps: int,
    warmdown_ratio: float = 0.65,
    final_lr_frac: float = 0.05,
) -> float:
    """Return nanochat's linear warmup, plateau, and linear warmdown multiplier."""
    if not 0 <= warmdown_ratio <= 1:
        raise ValueError("warmdown_ratio must be between zero and one")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    warmdown = round(warmdown_ratio * num_steps)
    if not warmdown or step <= num_steps - warmdown:
        return 1.0
    progress = max(0.0, (num_steps - step) / warmdown)
    return progress + (1 - progress) * final_lr_frac


def schedule_muon(optimizer: torch.optim.Optimizer, step: int, num_steps: int, warmdown_ratio: float) -> None:
    """Apply nanochat's pretraining momentum and cosine cautious-decay schedules."""
    warmdown = round(warmdown_ratio * num_steps)
    for group in optimizer.param_groups:
        if group["kind"] != "muon":
            continue
        if step < 400:
            momentum = 0.85 + (0.97 - 0.85) * step / 400
        elif warmdown and step >= num_steps - warmdown:
            momentum = 0.97 + (0.90 - 0.97) * (step - (num_steps - warmdown)) / warmdown
        else:
            momentum = 0.97
        group["momentum"] = momentum
        group["weight_decay"] = group["initial_weight_decay"] * 0.5 * (1 + math.cos(math.pi * step / num_steps))
