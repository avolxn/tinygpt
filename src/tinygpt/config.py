"""
GPT model configuration. Data only, no logic.
"""

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
    # Sliding window pattern tiled across layers. Final layer always full context.
    # L = long (full context), S = short (quarter context). Examples: "L", "SSSL"
    window_pattern: str = "SSSL"


def make_config(
    depth: int,
    *,
    aspect_ratio: int = 64,
    head_dim: int = 128,
    vocab_size: int = 32768,
    sequence_len: int = 2048,
    window_pattern: str = "SSSL",
) -> GPTConfig:
    """
    Build a GPTConfig from a depth scalar (nanochat-style autoconfig).

    model_dim = depth * aspect_ratio, rounded up to a multiple of head_dim so
    that head_dim divides evenly.
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
