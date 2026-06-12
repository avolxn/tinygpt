"""GPT model configuration and reference-derived scaling helpers."""

import math
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Model architecture hyperparameters."""

    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


REFERENCE_BATCH_SIZE = 2**19


def make_config(
    depth: int,
    *,
    aspect_ratio: int = 64,
    head_dim: int = 128,
    vocab_size: int = 32768,
    sequence_len: int = 2048,
    window_pattern: str = "SSSL",
) -> GPTConfig:
    """Build a GPTConfig from a depth scalar.

    model_dim is set to depth * aspect_ratio, rounded up to the next multiple
    of head_dim so that head_dim divides evenly.

    Args:
        depth: Number of transformer layers.
        aspect_ratio: Multiplier for model width relative to depth.
        head_dim: Attention head dimension; model_dim is rounded up to a multiple.
        vocab_size: Vocabulary size.
        sequence_len: Maximum sequence length.
        window_pattern: Sliding window pattern string (e.g. "SSSL").

    Returns:
        A GPTConfig with n_layer, n_head, n_kv_head, and n_embd derived from depth.
    """
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )


def compute_scaled_total_batch_size(
    *,
    scaling_params: int,
    d12_scaling_params: int,
    target_param_data_ratio: float,
    requested_total_batch_size: int,
) -> int:
    """Return the total token batch size using the reference scaling law."""
    if requested_total_batch_size > 0:
        return requested_total_batch_size
    target_tokens = target_param_data_ratio * scaling_params
    d12_target_tokens = target_param_data_ratio * d12_scaling_params
    batch_size_ratio = target_tokens / d12_target_tokens
    predicted_batch_size = REFERENCE_BATCH_SIZE * batch_size_ratio**0.383
    return 2 ** round(math.log2(predicted_batch_size))


def compute_scaled_weight_decay(
    *,
    base_weight_decay: float,
    total_batch_size: int,
    target_tokens: int,
    d12_target_tokens: float,
) -> float:
    """Return the reference scaled pretraining weight decay."""
    return base_weight_decay * math.sqrt(total_batch_size / REFERENCE_BATCH_SIZE) * (
        d12_target_tokens / target_tokens
    )
