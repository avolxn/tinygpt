"""
Optimizer utilities for tinygpt.

Provides parameter group construction (with weight-decay exclusions) for a GPT
model.  The actual AdamW optimizer is created by the caller, after FSDP
wrapping, so parameter identity is correct.
"""

from typing import Any

import torch
import torch.nn as nn


def make_param_groups(
    model: nn.Module,
    *,
    matrix_lr: float = 0.001,
    embedding_lr: float = 0.01,
    scalar_lr: float = 0.1,
    lm_head_lr: float | None = None,
    weight_decay: float = 0.1,
) -> list[dict[str, Any]]:
    """
    Split model parameters into AdamW-friendly groups.

    Groups:
      - matrix_params   : all 2-D weight matrices in transformer blocks  → weight decay
      - lm_head_params  : lm_head weight                                 → weight decay
      - embedding_params: wte + value_embeds                              → no decay
      - scalar_params   : 1-D params (lambdas, biases)                   → no decay

    Works with both unwrapped and FSDP-wrapped models because we filter by
    parameter name rather than module reference.

    Args:
        model: unwrapped or FSDP-wrapped GPT
        matrix_lr: LR for transformer block weight matrices
        embedding_lr: LR for embedding parameters
        scalar_lr: LR for scalar parameters
        lm_head_lr: LR for lm_head (defaults to matrix_lr)
        weight_decay: weight decay coefficient for matrix and lm_head params

    Returns:
        list of param-group dicts suitable for torch.optim.AdamW
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

    groups = [
        {
            "params": matrix_params,
            "lr": matrix_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": lm_head_params,
            "lr": lm_head_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": embedding_params,
            "lr": embedding_lr,
            "weight_decay": 0.0,
        },
        {
            "params": scalar_params,
            "lr": scalar_lr,
            "weight_decay": 0.0,
        },
    ]
    return [g for g in groups if len(g["params"]) > 0]  # type: ignore[arg-type]


def make_optimizer(
    model: nn.Module,
    *,
    matrix_lr: float = 0.001,
    embedding_lr: float = 0.01,
    scalar_lr: float = 0.1,
    lm_head_lr: float | None = None,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool = True,
) -> torch.optim.AdamW:
    """
    Create an AdamW optimizer with per-group learning rates and weight decay.

    Call this AFTER FSDP wrapping so parameter objects are the sharded ones.

    Args:
        model: FSDP-wrapped (or plain) GPT
        fused: use fused AdamW kernel (CUDA only; ignored on CPU/MPS)

    Returns:
        Configured AdamW optimizer
    """
    use_fused = fused and torch.cuda.is_available()
    param_groups = make_param_groups(
        model,
        matrix_lr=matrix_lr,
        embedding_lr=embedding_lr,
        scalar_lr=scalar_lr,
        lm_head_lr=lm_head_lr,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps, fused=use_fused)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer
