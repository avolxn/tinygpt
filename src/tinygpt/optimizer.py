"""
Combined MuonAdamW optimizer for tinygpt.

Uses torch.optim.Muon for 2D transformer block weight matrices, and
torch.optim.AdamW for embeddings, lm_head, and scalar parameters.

NOTE: torch.optim.Muon requires full (unsharded) parameter matrices for
Newton-Schulz orthogonalization to be correct. When training with FSDP,
use ShardingStrategy.NO_SHARD (equivalent to DDP). FULL_SHARD / SHARD_GRAD_OP
shard matrices along the first dimension, which breaks NS orthogonalization.
"""

from typing import Any, cast

import torch
import torch.nn as nn


def make_param_groups(
    model: nn.Module,
    *,
    matrix_lr: float = 0.02,
    embedding_lr: float = 0.3,
    scalar_lr: float = 0.5,
    lm_head_lr: float | None = None,
    weight_decay: float = 0.1,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
) -> list[dict[str, Any]]:
    """
    Split model parameters into MuonAdamW-friendly groups.

    Groups:
      - matrix_params   : 2-D transformer block weights → Muon (weight decay)
      - lm_head_params  : lm_head weight                → AdamW (weight decay)
      - embedding_params: wte + value_embeds            → AdamW (no decay)
      - scalar_params   : 1-D params (biases, lambdas)  → AdamW (no decay)

    Works with both unwrapped and FSDP-wrapped models because filtering is
    done by parameter name rather than module reference.
    """
    if lm_head_lr is None:
        lm_head_lr = matrix_lr

    matrix_params: list[nn.Parameter] = []
    lm_head_params: list[nn.Parameter] = []
    embedding_params: list[nn.Parameter] = []
    scalar_params: list[nn.Parameter] = []

    seen: set[int] = set()
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))

        if "wte" in name or "value_embeds" in name:
            embedding_params.append(param)
        elif "lm_head" in name:
            lm_head_params.append(param)
        elif param.dim() < 2 or "smear" in name or "lambda" in name or "ve_gate" in name:
            scalar_params.append(param)
        else:
            matrix_params.append(param)

    groups: list[dict[str, Any]] = []

    if matrix_params:
        groups.append(
            {
                "kind": "muon",
                "params": matrix_params,
                "lr": matrix_lr,
                "weight_decay": weight_decay,
                "momentum": muon_momentum,
                "ns_steps": muon_ns_steps,
            }
        )
    if lm_head_params:
        groups.append(
            {
                "kind": "adamw",
                "params": lm_head_params,
                "lr": lm_head_lr,
                "weight_decay": weight_decay,
                "betas": adamw_betas,
                "eps": adamw_eps,
            }
        )
    if embedding_params:
        groups.append(
            {
                "kind": "adamw",
                "params": embedding_params,
                "lr": embedding_lr,
                "weight_decay": 0.0,
                "betas": adamw_betas,
                "eps": adamw_eps,
            }
        )
    if scalar_params:
        groups.append(
            {
                "kind": "adamw",
                "params": scalar_params,
                "lr": scalar_lr,
                "weight_decay": 0.0,
                "betas": adamw_betas,
                "eps": adamw_eps,
            }
        )

    return groups


class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: torch.optim.Muon for 2D matrix params, AdamW for rest.

    Wraps torch.optim.Muon and torch.optim.AdamW as sub-optimizers.
    Gradient synchronization is handled externally (FSDP with NO_SHARD or DDP)
    before step() is called — this class does no distributed communication.

    The combined param_groups list exposes both sub-optimizers' groups so that
    standard LambdaLR schedulers work transparently: modifying group['lr'] in
    the combined list propagates to the actual sub-optimizer param groups via
    shared dict references.

    Args:
        param_groups: List of dicts with 'kind' key ('muon' or 'adamw').
            Muon groups: params, lr, weight_decay, momentum, ns_steps.
            AdamW groups: params, lr, weight_decay, betas, eps.
    """

    def __init__(self, param_groups: list[dict[str, Any]]) -> None:
        muon_groups = [
            {k: v for k, v in g.items() if k not in ("kind", "betas", "eps")}
            for g in param_groups
            if g["kind"] == "muon"
        ]
        adamw_groups = [
            {k: v for k, v in g.items() if k not in ("kind", "momentum", "ns_steps")}
            for g in param_groups
            if g["kind"] == "adamw"
        ]

        if not muon_groups:
            raise ValueError("MuonAdamW requires at least one 'muon' param group")
        if not adamw_groups:
            raise ValueError("MuonAdamW requires at least one 'adamw' param group")

        self._muon = torch.optim.Muon(muon_groups)
        self._adamw = torch.optim.AdamW(adamw_groups)

        # Initialise base Optimizer for isinstance compatibility (LambdaLR, etc.).
        # All params in one flat group; param_groups is replaced immediately after.
        all_params = [p for g in param_groups for p in g["params"]]
        super().__init__([{"params": all_params}], defaults={})

        # Replace base class param_groups with the sub-optimizers' actual groups.
        # The dicts are shared references, so LambdaLR can set group['lr'] and
        # it will take effect inside the sub-optimizers.
        self.param_groups = self._muon.param_groups + self._adamw.param_groups

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        # torch.optim.Muon currently exposes an untyped step() to mypy.
        cast(Any, self._muon).step(closure)
        self._adamw.step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])


def make_optimizer(
    model: nn.Module,
    *,
    matrix_lr: float = 0.02,
    embedding_lr: float = 0.3,
    scalar_lr: float = 0.5,
    lm_head_lr: float | None = None,
    weight_decay: float = 0.1,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
) -> MuonAdamW:
    """
    Create a MuonAdamW optimizer with per-group learning rates.

    Call this AFTER FSDP wrapping so parameter objects are the correct ones.
    Requires FSDP ShardingStrategy.NO_SHARD for multi-GPU correctness —
    Muon's Newton-Schulz step needs full unsharded matrices.
    """
    param_groups = make_param_groups(
        model,
        matrix_lr=matrix_lr,
        embedding_lr=embedding_lr,
        scalar_lr=scalar_lr,
        lm_head_lr=lm_head_lr,
        weight_decay=weight_decay,
        muon_momentum=muon_momentum,
        muon_ns_steps=muon_ns_steps,
        adamw_betas=adamw_betas,
        adamw_eps=adamw_eps,
    )
    optimizer = MuonAdamW(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer
