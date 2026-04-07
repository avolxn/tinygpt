"""
Bits-per-byte evaluation metric.

Taken from nanochat/loss_eval.py as-is.

Bits-per-byte (bpb) is a vocab-size-independent loss metric: it normalises the
sum of cross-entropy nats by the number of UTF-8 bytes the target tokens
represent, then converts to bits.

Usage:
    token_bytes = compute_token_bytes(tokenizer)   # once, cached on device
    bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
"""

import itertools
import math
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist

from tinygpt.utils import get_model_device


@torch.no_grad()
def evaluate_bpb(
    model: torch.nn.Module,
    batches: Iterator[tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    token_bytes: torch.Tensor,
) -> float:
    """
    Evaluate bits-per-byte on *steps* batches from *batches*.

    Args:
        model: GPT model with forward(x, y, loss_reduction='none') interface
        batches: iterable of (x, y) batches
        steps: number of batches to evaluate
        token_bytes: 1-D int tensor of shape (vocab_size,); entry i is the
                     number of UTF-8 bytes for token i, or 0 for special tokens

    Returns:
        Bits per byte (lower is better).
    """
    device = get_model_device(model)
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

    for x, y in itertools.islice(batches, steps):
        loss2d = model(x, y, loss_reduction="none")
        loss2d = loss2d.view(-1)
        y = y.view(-1)

        valid = y >= 0
        y_safe = torch.where(valid, y, torch.zeros_like(y))
        num_bytes2d = torch.where(valid, token_bytes[y_safe], torch.zeros_like(y, dtype=token_bytes.dtype))
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    total_nats_val: float = float(total_nats.item())
    total_bytes_val: int = int(total_bytes.item())
    if total_bytes_val == 0:
        return float("inf")
    return total_nats_val / (math.log(2) * total_bytes_val)


def compute_token_bytes(tokenizer: Any, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Build the token_bytes tensor from a tokenizer.

    Entry i is the number of UTF-8 bytes that token i decodes to, or 0 for
    special tokens (they should not contribute to the bpb metric).

    Args:
        tokenizer: HuggingFaceTokenizer instance
        device: where to place the resulting tensor

    Returns:
        1-D int32 tensor of shape (vocab_size,)
    """
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_bytes_list: list[int] = []
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    return torch.tensor(token_bytes_list, dtype=torch.int32, device=device)
