"""
KV cache for fast incremental decoding with flash attention.

Allocates dense tensors (B, T, H, D) for K and V across all layers,
enabling O(1) append operations during token generation.
"""

from __future__ import annotations

import torch


class KVCache:
    """Pre-allocated KV cache for flash attention (B, T, H, D) layout.

    Compatible with both FA2 and SDPA backends.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Allocate zeroed K/V cache tensors for the full model.

        Args:
            batch_size: Number of parallel sequences.
            num_heads: Number of KV heads per layer.
            seq_len: Maximum sequence length the cache can hold.
            head_dim: Dimensionality of each attention head.
            num_layers: Number of transformer layers.
            device: Device on which to allocate the cache tensors.
            dtype: dtype of the cache tensors.
        """
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.prev_embedding: torch.Tensor | None = None

    def get_pos(self) -> int:
        """Return the current fill position (number of tokens already cached).

        Returns:
            Integer position of the first free slot, read from cache_seqlens[0].
        """
        return self.cache_seqlens[0].item()  # type: ignore[return-value]

    def get_layer_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the K and V cache slices for a specific layer.

        Args:
            layer_idx: Index of the transformer layer.

        Returns:
            A (k_cache, v_cache) tuple each of shape (B, T_max, H, D).
        """
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens: int) -> None:
        """Increment all sequence positions by num_tokens.

        Args:
            num_tokens: Number of newly appended tokens.
        """
        self.cache_seqlens += num_tokens

    def prefill(self, other: KVCache) -> None:
        """Copy KV state from a batch-1 prefill cache into this decode cache.

        Args:
            other: Source batch-1 cache populated during the prefill forward pass.

        Raises:
            RuntimeError: If this cache already has tokens (get_pos() != 0).
            ValueError: If other has a different number of layers.
        """
        if self.get_pos() != 0:
            raise RuntimeError("prefill() called on non-empty cache")
        if self.n_layers != other.n_layers:
            raise ValueError(f"Layer count mismatch: {self.n_layers} vs {other.n_layers}")
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()
