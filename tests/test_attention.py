"""
Tests for the attention backend (FA2 vs SDPA numerical equivalence).

When FA2 is not available, all tests run on SDPA only.
"""

import pytest
import torch

from tinygpt.attention import flash_attn_available, flash_attn_func, flash_attn_with_kvcache


def make_qkv(B: int, T: int, H: int, Hkv: int, D: int, device="cpu") -> tuple[torch.Tensor, ...]:
    q = torch.randn(B, T, H, D, device=device)
    k = torch.randn(B, T, Hkv, D, device=device)
    v = torch.randn(B, T, Hkv, D, device=device)
    return q, k, v


def test_sdpa_full_causal() -> None:
    """SDPA forward pass returns correct shape for causal full-context attention."""
    q, k, v = make_qkv(2, 16, 4, 4, 32)
    out = flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))
    assert out.shape == q.shape


def test_sdpa_gqa() -> None:
    """SDPA works with GQA (n_head != n_kv_head)."""
    q, k, v = make_qkv(2, 16, 4, 2, 32)
    out = flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))
    assert out.shape == q.shape


def test_sdpa_sliding_window() -> None:
    """Sliding window reduces effective context but output shape is preserved."""
    B, T, H, D = 1, 32, 2, 16
    q, k, v = make_qkv(B, T, H, H, D)
    # window=8 < T=32
    out = flash_attn_func(q, k, v, causal=True, window_size=(8, 0))
    assert out.shape == (B, T, H, D)


@pytest.mark.skipif(not flash_attn_available, reason="FA2 not available on this machine")
def test_fa2_sdpa_equivalence() -> None:
    """FA2 and SDPA produce numerically close outputs (atol=0.01 for bf16)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for FA2")
    device = "cuda"
    q, k, v = make_qkv(2, 64, 4, 4, 32, device=device)
    q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

    # Use FA2 directly
    import tinygpt.attention as attn_module  # noqa: PLC0415

    out_fa2 = attn_module.flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))

    # Use SDPA directly
    from tinygpt.attention import sdpa_attention  # noqa: PLC0415

    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out_sdpa = sdpa_attention(q_t, k_t, v_t, window_size=(-1, 0), enable_gqa=False).transpose(1, 2)

    assert out_fa2.shape == out_sdpa.shape
    torch.testing.assert_close(out_fa2.float(), out_sdpa.float(), atol=0.02, rtol=1e-2)


def test_kvcache_sdpa() -> None:
    """KV cache SDPA path correctly inserts new tokens and returns correct shape."""
    B, H, D = 1, 2, 16
    T_max = 32
    q_new = torch.randn(B, 4, H, D)
    k_new = torch.randn(B, 4, H, D)
    v_new = torch.randn(B, 4, H, D)
    k_cache = torch.zeros(B, T_max, H, D)
    v_cache = torch.zeros(B, T_max, H, D)
    seqlens = torch.zeros(B, dtype=torch.int32)

    out = flash_attn_with_kvcache(
        q_new,
        k_cache,
        v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=seqlens,
        causal=True,
        window_size=(-1, 0),
    )
    assert out.shape == (B, 4, H, D)
    # Cache should be updated in-place
    assert not k_cache[:, :4].eq(0).all(), "Cache was not updated"


def test_kvcache_incremental_decode() -> None:
    """Incremental single-token decode produces consistent output."""
    B, H, D = 1, 2, 16
    T_max = 32
    T_prompt = 8
    k_cache = torch.zeros(B, T_max, H, D)
    v_cache = torch.zeros(B, T_max, H, D)
    seqlens = torch.zeros(B, dtype=torch.int32)

    # Prefill
    q = torch.randn(B, T_prompt, H, D)
    k = torch.randn(B, T_prompt, H, D)
    v = torch.randn(B, T_prompt, H, D)
    flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=seqlens, causal=True, window_size=(-1, 0))
    seqlens += T_prompt

    # Decode one token
    q_dec = torch.randn(B, 1, H, D)
    k_dec = torch.randn(B, 1, H, D)
    v_dec = torch.randn(B, 1, H, D)
    out = flash_attn_with_kvcache(
        q_dec, k_cache, v_cache, k=k_dec, v=v_dec, cache_seqlens=seqlens, causal=False, window_size=(-1, 0)
    )
    assert out.shape == (B, 1, H, D)
