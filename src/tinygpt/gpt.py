"""
GPT model architecture (from scratch).

Architecture mirrors nanochat exactly:
- RoPE (no positional embeddings)
- QK norm
- Untied embedding / lm_head weights
- relu² MLP
- RMSNorm (no learnable params, F.rms_norm)
- No bias in linear layers
- GQA (Group-Query Attention)
- Sliding window attention
- Value embeddings (ResFormer-style)
- Smear gate (bigram info)
- Per-layer resid/x0 scalars
- Backout (mid-layer residual subtraction)

Differences from nanochat gpt.py:
- setup_optimizer() removed (optimizer created in training script after FSDP wrap)
- Imports from tinygpt.* instead of nanochat.*
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinygpt.attention import flash_attn_func, flash_attn_with_kvcache
from tinygpt.config import GPTConfig
from tinygpt.runtime import compute_dtype, print0

if TYPE_CHECKING:
    from tinygpt.engine import KVCache

qk_scale = 1.2  # post-QK-norm scale applied to queries and keys
softcap = 15.0  # logit soft-cap to prevent outlier logits


def norm(x: torch.Tensor) -> torch.Tensor:
    """Apply RMSNorm over the last dimension of x.

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        Normalised tensor of the same shape as x.
    """
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """
    nn.Linear that casts weights to match the input dtype in forward.

    Master weights stay fp32 for optimizer precision; matmuls run in the
    activation dtype (typically bf16 from embeddings).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation, casting weights to match x's dtype.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after the linear transformation.
        """
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Return whether a layer should have a Value Embedding.

    Value embeddings are placed on alternating layers with the last layer always included.

    Args:
        layer_idx: Zero-based layer index.
        n_layer: Total number of transformer layers.

    Returns:
        True if this layer should have a Value Embedding.
    """
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embeddings to a 4-D query or key tensor.

    Args:
        x: Input tensor of shape (B, T, H, D).
        cos: Cosine values of shape (1, T, 1, D//2).
        sin: Sine values of shape (1, T, 1, D//2).

    Returns:
        Rotated tensor of shape (B, T, H, D).

    Raises:
        ValueError: If x is not a 4-D tensor.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {x.ndim}D")
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        """Initialise attention projections and optional value-embedding gate.

        Args:
            config: Model hyperparameters.
            layer_idx: Zero-based index of this layer, used to decide VE placement.

        Raises:
            ValueError: If n_embd is not divisible by n_head, or n_head is not
                divisible by n_kv_head.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_kv_head > self.n_head or self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = (
            Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        )

    def forward(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        """Compute multi-head causal self-attention with optional KV cache.

        Args:
            x: Input hidden states of shape (B, T, C).
            ve: Optional value embedding of shape (B, T, n_kv_head * head_dim),
                or None if this layer has no value embedding.
            cos_sin: (cos, sin) rotary embedding tensors for positions T0..T0+T.
            window_size: (left, right) sliding window sizes for flash attention.
            kv_cache: KV cache for incremental decoding, or None for training.

        Returns:
            Output tensor of shape (B, T, C).
        """
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            if self.ve_gate is None:
                raise RuntimeError("ve_gate is None but ve input was provided")
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * qk_scale
        k = k * qk_scale

        if kv_cache is None:
            y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)  # type: ignore[no-any-return]


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        """Initialise the two-layer relu² MLP.

        Args:
            config: Model hyperparameters; only n_embd is used.
        """
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the relu² MLP to the input.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        """Initialise one transformer block (attention + MLP).

        Args:
            config: Model hyperparameters.
            layer_idx: Zero-based layer index forwarded to CausalSelfAttention.
        """
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        """Apply pre-norm attention then pre-norm MLP with residual connections.

        Args:
            x: Residual stream tensor of shape (B, T, C).
            ve: Optional value embedding passed to attention.
            cos_sin: Rotary embedding tensors for this layer's token positions.
            window_size: Sliding window sizes forwarded to flash attention.
            kv_cache: Optional KV cache for incremental decoding.

        Returns:
            Updated residual stream of shape (B, T, C).
        """
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64) -> None:
        """Register all sub-modules; allocate no tensors (meta-device safe).

        Actual tensor allocation happens in init_weights().

        Args:
            config: Model architecture hyperparameters.
            pad_vocab_size_to: Pad vocab to the next multiple of this value for
                efficiency; default 64.
        """
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.wte = nn.Embedding(padded_vocab_size, config.n_embd)
        self.h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(padded_vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )

        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize all model parameters (called after meta-device allocation)."""
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block_mod in self.h:
            block: Block = block_mod  # type: ignore[assignment]
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        for ve_mod in self.value_embeds.values():
            ve: nn.Embedding = ve_mod  # type: ignore[assignment]
            torch.nn.init.uniform_(ve.weight, -s, s)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        if compute_dtype != torch.float16:
            self.wte.to(dtype=compute_dtype)
            for ve_mod in self.value_embeds.values():
                ve_mod.to(dtype=compute_dtype)

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: int = 100000,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute RoPE cos/sin tensors for up to seq_len positions.

        Args:
            seq_len: Number of positions to precompute.
            head_dim: Attention head dimension; must be even.
            base: RoPE base frequency (theta).
            device: Target device; defaults to the embedding weight device.

        Returns:
            A (cos, sin) tuple each of shape (1, seq_len, 1, head_dim // 2).
        """
        if device is None:
            device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos = cos.to(compute_dtype)[None, :, None, :]
        sin = sin.to(compute_dtype)[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config: GPTConfig) -> list[tuple[int, int]]:
        """Compute per-layer (left, right) window size tuples for flash attention.

        Args:
            config: Model config supplying window_pattern, sequence_len, and n_layer.

        Returns:
            A list of length n_layer containing (left, right) window size tuples.
            The last layer always uses full context.

        Raises:
            ValueError: If window_pattern contains characters other than 'S' or 'L'.
        """
        pattern = config.window_pattern.upper()
        if not all(c in "SL" for c in pattern):
            raise ValueError(f"Invalid window_pattern: {pattern!r} (only 'S' and 'L' allowed)")
        long_window = config.sequence_len
        short_window = math.ceil(long_window / 4 / 128) * 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_window, 0)  # final layer always full context
        return sizes

    def get_device(self) -> torch.device:
        """Return the device on which this model's parameters reside.

        Returns:
            The device of the embedding weight tensor.
        """
        return self.wte.weight.device

    def estimate_flops(self) -> int:
        """Estimate FLOPs per token for both forward and backward passes.

        Returns:
            Approximate integer FLOPs per token.
        """
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())  # type: ignore[operator,misc]
        nparams_exclude = (
            self.wte.weight.numel()
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.smear_gate.weight.numel()
            + self.smear_lambda.numel()
            + self.backout_lambda.numel()
        )
        n_head = self.config.n_head
        head_dim = self.config.n_embd // self.config.n_head
        seq_len = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = seq_len if window < 0 else min(window, seq_len)
            attn_flops += 12 * n_head * head_dim * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self) -> dict[str, int]:
        """Return parameter counts broken down by component for scaling law analysis.

        Returns:
            Dict with keys 'wte', 'value_embeds', 'lm_head', 'transformer_matrices',
            'scalars', and 'total'.
        """
        wte = sum(p.numel() for p in self.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.h.parameters())
        scalars = (
            self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.smear_gate.weight.numel()
            + self.smear_lambda.numel()
            + self.backout_lambda.numel()
        )
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        loss_reduction: str = "mean",
    ) -> torch.Tensor:
        """Run a forward pass, returning either logits or cross-entropy loss.

        Args:
            idx: Token id tensor of shape (B, T).
            targets: Optional target token ids of shape (B, T); if provided,
                returns the cross-entropy loss instead of logits.
            kv_cache: Optional KV cache for incremental decoding.
            loss_reduction: 'mean', 'sum', or 'none'; passed to F.cross_entropy.

        Returns:
            Cross-entropy loss scalar (or unreduced tensor) when targets is given,
            otherwise logits of shape (B, T, vocab_size).

        Raises:
            ValueError: If T exceeds the rotary cache length, or T <= 1 in
                training mode.
        """
        B, T = idx.size()

        if T > self.cos.size(1):
            raise ValueError(f"Sequence length {T} > rotary cache {self.cos.size(1)}")
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0 : T0 + T], self.sin[:, T0 : T0 + T]

        x = self.wte(idx)
        x = x.to(compute_dtype)
        x = norm(x)

        # Smear: mix previous token's embedding into current position
        smear = self.smear_lambda.to(x.dtype)
        smear_ch = self.smear_gate.in_features
        if kv_cache is None:
            if T <= 1:
                raise ValueError("Training forward pass requires T > 1")
            gate = smear * torch.sigmoid(self.smear_gate(x[:, 1:, :smear_ch]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                gate = smear * torch.sigmoid(self.smear_gate(x[:, 1:, :smear_ch]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                gate = smear * torch.sigmoid(self.smear_gate(x[:, :, :smear_ch]))
                x = x + gate * x_pre_smear

        x0 = x
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None
        for i, block_mod in enumerate(self.h):
            block: Block = block_mod  # type: ignore[assignment]
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if str(i) in self.value_embeds:
                ve_emb: nn.Embedding = self.value_embeds[str(i)]  # type: ignore[assignment]
                ve = ve_emb(idx).to(x.dtype)
            else:
                ve = None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        logits = self.lm_head(x)
        logits = logits[..., : self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            if loss_reduction == "none":
                loss = loss.view(targets.shape)
            return loss
        return logits

    @torch.inference_mode()
    def generate(
        self,
        tokens: list[int],
        max_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
    ) -> Iterator[int]:
        """Naive autoregressive generation without a KV cache.

        Args:
            tokens: Prompt token ids.
            max_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature; 0.0 selects the argmax greedily.
            top_k: If set, restrict sampling to the top-k most probable tokens.
            seed: RNG seed used when temperature > 0.

        Yields:
            Integer token ids, one per generation step.

        Raises:
            TypeError: If tokens is not a list.
        """
        if not isinstance(tokens, list):
            raise TypeError("tokens must be a list")
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            yield int(next_ids.item())
