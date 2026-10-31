"""Bits-per-byte evaluation metric.

Bits-per-byte (bpb) is a vocab-size-independent loss metric: it normalises the
sum of cross-entropy nats by the number of UTF-8 bytes the target tokens
represent, then converts to bits.

Usage:
    token_bytes = compute_token_bytes(tokenizer)   # once, cached on device
    bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
"""

import itertools
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from tokenizers.decoders import ByteLevel

from tinygpt.train import causal_lm_loss


@dataclass(frozen=True)
class TokenBytes:
    """UTF-8 byte counts in running text and immediately after a document separator."""

    values: torch.Tensor
    initial_values: torch.Tensor
    bos_token_id: int


@torch.no_grad()
def evaluate_bpb(
    model: torch.nn.Module,
    batches: Iterator[tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    token_bytes: TokenBytes,
) -> float:
    """Evaluate bits-per-byte on *steps* batches from *batches*.

    Args:
        model: Transformers causal language model
        batches: iterable of (x, y) batches
        steps: number of batches to evaluate
        token_bytes: Byte counts with document-boundary handling; special tokens count as zero.

    Returns:
        Bits per byte (lower is better).
    """
    device = next(model.parameters()).device
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

    for x, y in itertools.islice(batches, steps):
        loss2d = causal_lm_loss(model, x, y, reduction="none")
        loss2d = loss2d.view(-1)
        y = y.view(-1)

        valid = y >= 0
        y_safe = torch.where(valid, y, torch.zeros_like(y))
        sizes = torch.where(
            x.reshape(-1).eq(token_bytes.bos_token_id),
            token_bytes.initial_values[y_safe],
            token_bytes.values[y_safe],
        )
        num_bytes2d = torch.where(valid, sizes, torch.zeros_like(sizes))
        total_nats += (loss2d * (valid & token_bytes.values[y_safe].gt(0))).sum()
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


def compute_token_bytes(tokenizer: Any, device: torch.device | str = "cpu") -> TokenBytes:
    """Count decoded UTF-8 bytes, including SentencePiece document-boundary behavior.

    Byte-fallback tokens count as one raw byte even when a single token cannot
    form valid UTF-8. Special tokens do not contribute to the BPB denominator.

    Args:
        tokenizer: HuggingFaceTokenizer instance
        device: where to place the resulting tensor

    Returns:
        Byte-count tables and the document separator ID.
    """
    native = tokenizer.tokenizer
    decoder = native.backend_tokenizer.decoder
    if decoder is None:
        raise ValueError("BPB evaluation requires a tokenizer decoder")
    special_ids = set(native.all_special_ids)
    tokens = native.convert_ids_to_tokens(list(range(tokenizer.get_vocab_size())))

    prefix_bytes = len(decoder.decode(["a"]).encode("utf-8"))
    token_bytes_list = []
    initial_bytes_list = []
    for token_id, token in enumerate(tokens):
        if token_id in special_ids:
            size = initial_size = 0
        elif isinstance(decoder, ByteLevel):
            size = initial_size = len(token)
        elif re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", token):
            size = initial_size = 1
        else:
            size = len(decoder.decode(["a", token]).encode("utf-8")) - prefix_bytes
            initial_size = len(decoder.decode([token]).encode("utf-8"))
        token_bytes_list.append(size)
        initial_bytes_list.append(initial_size)
    return TokenBytes(
        torch.tensor(token_bytes_list, dtype=torch.int32, device=device),
        torch.tensor(initial_bytes_list, dtype=torch.int32, device=device),
        tokenizer.get_bos_token_id(),
    )
