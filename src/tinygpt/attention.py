"""
Unified attention backend: FA2 (Ampere+) with SDPA fallback.

FA2 is tried first; any GPU below SM80 (or CPU/MPS) falls back to PyTorch SDPA
automatically.

Usage:
    from tinygpt.attention import flash_attn_func, flash_attn_with_kvcache

    # Training (no KV cache)
    y = flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""

import importlib
import logging
from typing import Any

import torch
import torch.nn.functional as F

from tinygpt.runtime import compute_dtype

logger = logging.getLogger(__name__)

flash_attn: Any = None
try:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        flash_attn = importlib.import_module("flash_attn")
except ImportError:
    pass

flash_attn_available = flash_attn is not None

# Default backend choice based on hardware/dtype at load time
use_flash_attn = flash_attn is not None and compute_dtype == torch.bfloat16


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: tuple[int, int],
    enable_gqa: bool,
) -> torch.Tensor:
    """SDPA attention with optional sliding window.

    Args:
        q: Query tensor of shape (B, H, T, D).
        k: Key tensor of shape (B, H, T, D).
        v: Value tensor of shape (B, H, T, D).
        window_size: (left, right) sliding window; left=-1 means unlimited.
        enable_gqa: Whether to enable grouped-query attention.

    Returns:
        Output tensor of shape (B, H, T, D).
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    if Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: (B, T, H, D)
        causal: use causal masking
        window_size: (left, right) sliding window; -1 = unlimited

    Returns:
        (B, T, H, D)
    """
    if use_flash_attn:
        return flash_attn.flash_attn_func(q, k, v, causal=causal, window_size=window_size)  # type: ignore[no-any-return]

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """
    Flash Attention with KV cache for inference.

    FA2 updates k_cache/v_cache in-place; SDPA fallback does the same.

    Args:
        q: (B, T_new, H, D)
        k_cache, v_cache: (B, T_max, H_kv, D)
        k, v: new keys/values (B, T_new, H_kv, D)
        cache_seqlens: current position per batch element, shape (B,) int32
        causal: causal masking
        window_size: (left, right) sliding window

    Returns:
        (B, T_new, H, D)
    """
    if use_flash_attn:
        return flash_attn.flash_attn_with_kvcache(  # type: ignore[no-any-return]
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=causal,
            window_size=window_size,
        )

    # SDPA fallback: manually manage cache
    B, T_new, H, D = q.shape
    pos: int = int(cache_seqlens[0].item())  # type: ignore[index]

    if k is not None and v is not None:
        k_cache[:, pos : pos + T_new, :, :] = k
        v_cache[:, pos : pos + T_new, :, :] = v

    end_pos: int = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y = sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)
    return y.transpose(1, 2)
