"""
Unified attention backend with automatic FA4/FA3/FA2/SDPA fallback chain.

Priority per operation:
    flash_attn_func         : FA4 > FA3 > FA2 > SDPA
    flash_attn_with_kvcache : FA3 > FA2 > SDPA  (FA4 cute has no kvcache API)

GPU / dtype constraints:
    FA4 (flash_attn.cute)          — Hopper SM90+ or Blackwell SM100+, bf16
    FA3 (kernels-community/flash-attn3) — Hopper SM90 only, bf16
    FA2 (flash_attn package)       — Ampere SM80+, bf16 / fp16
    SDPA                           — any device / dtype

Installation:
    FA4: pip install flash-attn-4
    FA3: pip install kernels   (pre-compiled Hopper kernels, no local build needed)
    FA2: pip install flash-attn --no-build-isolation
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F

from tinygpt.utils import compute_dtype

logger = logging.getLogger(__name__)

_fa4: Any = None  # flash_attn.cute  (FA4 — SM90+)
_fa3: Any = None  # kernels-community/flash-attn3 (FA3 — SM90 only)
_fa2: Any = None  # flash_attn package (FA2 — SM80+)

if torch.cuda.is_available():
    _sm_major, _ = torch.cuda.get_device_capability()

    if _sm_major >= 9:
        try:
            import flash_attn.cute as _fa4  # type: ignore[import]
        except Exception:
            pass

    if _sm_major == 9:
        try:
            from kernels import get_kernel  # type: ignore[import]
            _fa3 = get_kernel("kernels-community/flash-attn3", version=1)
        except Exception:
            pass

    if _sm_major >= 8:
        try:
            import flash_attn as _fa2  # type: ignore[import]
        except ImportError:
            pass

_bf16 = compute_dtype == torch.bfloat16
_fp16 = compute_dtype == torch.float16

# flash_attn_func: FA4 > FA3 > FA2 > SDPA
if _fa4 is not None and _bf16:
    _fwd, _fwd_ver = _fa4, 4
elif _fa3 is not None and _bf16:
    _fwd, _fwd_ver = _fa3, 3
elif _fa2 is not None and (_bf16 or _fp16):
    _fwd, _fwd_ver = _fa2, 2
else:
    _fwd, _fwd_ver = None, 0

# flash_attn_with_kvcache: FA3 > FA2 > SDPA  (FA4 cute has no kvcache API)
if _fa3 is not None and _bf16:
    _kvc, _kvc_ver = _fa3, 3
elif _fa2 is not None and (_bf16 or _fp16):
    _kvc, _kvc_ver = _fa2, 2
else:
    _kvc, _kvc_ver = None, 0

flash_attn_available: bool = _fwd is not None
use_flash_attn: bool = _fwd is not None
flash_attn_backend: str | None = f"FA{_fwd_ver}" if _fwd_ver else None


def _to_fa4_window(
    window_size: tuple[int, int],
) -> tuple[int | None, int | None]:
    """Convert FA2/FA3 window convention (-1 = unlimited) to FA4 (None = unlimited)."""
    left, right = window_size
    return (None if left == -1 else left, None if right == -1 else right)


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: tuple[int, int],
    enable_gqa: bool,
) -> torch.Tensor:
    """SDPA with optional causal sliding window. Inputs in (B, H, T, D) layout."""
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    if Tq == 1:
        if 0 <= window < Tk:
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if 0 <= window < Tk:
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
        causal: causal masking
        window_size: (left, right) sliding window; -1 = unlimited

    Returns:
        (B, T, H, D)
    """
    if _fwd is not None:
        if _fwd_ver == 4:
            return _fwd.flash_attn_func(  # type: ignore[no-any-return]
                q, k, v, causal=causal, window_size=_to_fa4_window(window_size)
            )
        else:
            return _fwd.flash_attn_func(  # type: ignore[no-any-return]
                q, k, v, causal=causal, window_size=window_size
            )

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

    FA3/FA2 update k_cache/v_cache in-place; SDPA fallback does the same.
    FA4 (cute) has no kvcache API — this function uses FA3 > FA2 > SDPA.

    Args:
        q: (B, T_new, H, D)
        k_cache, v_cache: (B, T_max, H_kv, D)
        k, v: new keys/values (B, T_new, H_kv, D)
        cache_seqlens: current position per batch element, shape (B,) int32
        causal: causal masking
        window_size: (left, right) sliding window; -1 = unlimited

    Returns:
        (B, T_new, H, D)
    """
    if _kvc is not None:
        return _kvc.flash_attn_with_kvcache(  # type: ignore[no-any-return]
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=causal,
            window_size=window_size,
        )

    T_new = q.size(1)
    pos: int = int(cache_seqlens[0].item())  # type: ignore[index]

    if k is not None and v is not None:
        k_cache[:, pos : pos + T_new, :, :] = k
        v_cache[:, pos : pos + T_new, :, :] = v

    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y = sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)
    return y.transpose(1, 2)
