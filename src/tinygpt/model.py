"""Nanochat student using Transformers model, attention, and serialization APIs.

The architecture follows karpathy/nanochat's GPT at the revision in config.py.
Transformers supplies RoPE, ReLU² MLPs, attention backends, and checkpoint loading.
Value embeddings, residual scalars, smear, backout, and initialization follow
the reference because these components are absent from Transformers NanoChat.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.nanochat.modeling_nanochat import (
    NanoChatMLP,
    NanoChatRotaryEmbedding,
    apply_rotary_pos_emb,
)

from tinygpt.config import TinyGPTConfig
from tinygpt.utils import compute_dtype


def norm(x: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """Apply nanochat's parameter-free RMS normalization."""
    return F.rms_norm(x, (x.size(-1),), eps=eps)


class Attention(nn.Module):
    """Apply rotary QK-normalized attention with optional value embeddings.

    Args:
        config: Student architecture.
        layer_idx: Layer position used to select the attention window.
    """

    def __init__(self, config: TinyGPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // kv_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.layer_idx = layer_idx
        self.sliding_window = config.attention_windows()[layer_idx]
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.ve_gate = (
            nn.Linear(min(12, config.hidden_size), kv_heads, bias=False)
            if layer_idx % 2 == (config.transformer_depth - 1) % 2
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        value_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the projected attention output for a complete training sequence."""
        shape = (*x.shape[:-1], -1, self.head_dim)
        q, k, v = (projection(x).view(shape).transpose(1, 2) for projection in (self.q_proj, self.k_proj, self.v_proj))
        if value_embedding is not None:
            assert self.ve_gate is not None
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate.in_features]))
            v = v + gate.transpose(1, 2).unsqueeze(-1) * value_embedding.view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        q, k = norm(q, self.config.rms_norm_eps) * 1.2, norm(k, self.config.rms_norm_eps) * 1.2
        if self.config._attn_implementation.startswith("flash_attention"):
            output, _ = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation](
                self, q, k, v, None, dropout=0.0, scaling=self.scaling, sliding_window=self.sliding_window + 1
            )
        else:
            length = x.size(1)
            positions = torch.arange(length, device=x.device)
            distance = positions[:, None] - positions[None, :]
            mask = (distance >= 0) & (distance <= self.sliding_window)
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=self.scaling, enable_gqa=self.num_key_value_groups != 1
            ).transpose(1, 2)
        return cast(torch.Tensor, self.o_proj(output.reshape(*x.shape)))


class Block(nn.Module):
    """Nanochat Transformer block, used as the FSDP wrapping boundary.

    Args:
        config: Student architecture.
        layer_idx: Transformer layer position.
    """

    def __init__(self, config: TinyGPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.eps = config.rms_norm_eps
        self.self_attn = Attention(config, layer_idx)
        self.mlp = NanoChatMLP(config)  # type: ignore[no-untyped-call]

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        value_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply attention and ReLU² MLP residuals."""
        x = x + self.self_attn(norm(x, self.eps), position_embeddings, value_embedding)
        return cast(torch.Tensor, x + self.mlp(norm(x, self.eps)))


class TinyGPTForCausalLM(PreTrainedModel):  # type: ignore[no-untyped-call]
    """Train the nanochat architecture and save standard Hugging Face checkpoints.

    Args:
        config: Nanochat student architecture and tokenizer vocabulary size.
    """

    config_class = TinyGPTConfig
    base_model_prefix = "model"
    _no_split_modules = ["Block"]
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.padded_vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Block(config, i) for i in range(config.transformer_depth))
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(
                    config.padded_vocab_size,
                    (config.num_key_value_heads or config.num_attention_heads)
                    * (config.hidden_size // config.num_attention_heads),
                )
                for i, block in enumerate(self.layers)
                if i % 2 == (config.transformer_depth - 1) % 2
            }
        )
        self.lm_head: nn.Linear = nn.Linear(config.hidden_size, config.padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.empty(config.transformer_depth))
        self.x0_lambdas = nn.Parameter(torch.empty(config.transformer_depth))
        self.smear_gate = nn.Linear(min(24, config.hidden_size), 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.empty(1))
        self.backout_lambda = nn.Parameter(torch.empty(1))
        self.rotary_emb = NanoChatRotaryEmbedding(config)
        self.post_init()  # type: ignore[no-untyped-call]

    def get_input_embeddings(self) -> nn.Embedding:
        """Return token embeddings for Transformers integrations."""
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        """Return the untied language-model projection."""
        return self.lm_head

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the exact reference parameter groups, including residual scalars."""
        if isinstance(module, nn.Embedding):
            bound = math.sqrt(3 / self.config.hidden_size)
            if module is self.embed_tokens:
                nn.init.normal_(module.weight, std=0.8)
            else:
                nn.init.uniform_(module.weight, -bound, bound)
        elif isinstance(module, nn.Linear):
            bound = math.sqrt(3 / self.config.hidden_size)
            nn.init.uniform_(module.weight, -bound, bound)
            if module is self.lm_head:
                nn.init.normal_(module.weight, std=0.001)
            elif module is self.smear_gate:
                nn.init.uniform_(module.weight, 0.0, 0.02)
        elif isinstance(module, Attention):
            nn.init.zeros_(module.o_proj.weight)
            if module.ve_gate is not None:
                nn.init.uniform_(module.ve_gate.weight, 0.0, 0.02)
        elif isinstance(module, NanoChatMLP):
            bound = math.sqrt(3 / self.config.hidden_size) * 0.4
            nn.init.uniform_(module.fc1.weight, -bound, bound)
            nn.init.zeros_(module.fc2.weight)
        elif isinstance(module, NanoChatRotaryEmbedding):
            super()._init_weights(module)  # type: ignore[no-untyped-call]
        elif module is self:
            nn.init.zeros_(self.smear_lambda)
            nn.init.constant_(self.backout_lambda, 0.2)
            for i in range(self.config.transformer_depth):
                self.resid_lambdas[i] = 1.15 - 0.1 * i / max(self.config.transformer_depth - 1, 1)
                self.x0_lambdas[i] = 0.2 - 0.15 * i / max(self.config.transformer_depth - 1, 1)

    def scaling_param_counts(self) -> dict[str, int]:
        """Count the reference buckets used to derive the training token budget."""
        counts = {
            "wte": self.embed_tokens.weight.numel(),
            "value_embeds": sum(p.numel() for p in self.value_embeds.parameters()),
            "lm_head": self.lm_head.weight.numel(),
            "transformer_matrices": sum(p.numel() for p in self.layers.parameters()),
            "scalars": sum(
                p.numel() for p in (self.resid_lambdas, self.x0_lambdas, self.smear_lambda, self.backout_lambda)
            )
            + self.smear_gate.weight.numel(),
        }
        counts["total"] = sum(counts.values())
        return counts

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Compute full-sequence logits; generation and caching are handled by vLLM.

        Args:
            input_ids: Token IDs of shape ``(batch, sequence)`` without left padding.
            labels: Optional unshifted Transformers labels with ``-100`` ignored.
            **kwargs: Unused Trainer integration arguments.

        Returns:
            Float32 softcapped logits and optional causal language-model loss.
        """
        dtype = compute_dtype if input_ids.device.type == "cuda" else self.embed_tokens.weight.dtype
        with torch.autocast(input_ids.device.type, dtype=dtype, enabled=dtype in (torch.bfloat16, torch.float16)):
            x = norm(self.embed_tokens(input_ids).to(dtype), self.config.rms_norm_eps)
            gate = self.smear_lambda.to(dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, : self.smear_gate.in_features]))
            x = torch.cat((x[:, :1], x[:, 1:] + gate * x[:, :-1]), dim=1)
            x0 = x
            positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(x, positions)
            backout = None
            for i, block in enumerate(self.layers):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](input_ids).to(dtype) if str(i) in self.value_embeds else None
                x = block(x, position_embeddings, ve)
                if i == self.config.transformer_depth // 2:
                    backout = x
            if backout is not None:
                x = x - self.backout_lambda.to(dtype) * backout
            logits = self.lm_head(norm(x, self.config.rms_norm_eps))[..., : self.config.vocab_size].float()
        logits = 15.0 * torch.tanh(logits / 15.0)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        return CausalLMOutputWithPast(loss=loss, logits=logits)  # type: ignore[arg-type]


AutoConfig.register(TinyGPTConfig.model_type, TinyGPTConfig)
AutoModelForCausalLM.register(TinyGPTConfig, TinyGPTForCausalLM)


def enable_float8_training(model: TinyGPTForCausalLM, recipe: str) -> None:
    """Use TorchAO FP8 linear layers with nanochat's shape filter and BF16 evaluation.

    Args:
        model: Initialized CUDA student before FSDP wrapping.
        recipe: Native TorchAO ``tensorwise`` or ``rowwise`` scaling recipe.

    Raises:
        ValueError: If the model is not on an FP8-capable CUDA device.
    """
    if model.device.type != "cuda" or torch.cuda.get_device_capability(model.device)[0] < 9:
        raise ValueError("FP8 training requires an H100 or newer CUDA GPU")
    from torchao.float8 import Float8Linear as TorchFloat8Linear
    from torchao.float8 import Float8LinearConfig

    class Float8Linear(TorchFloat8Linear):  # type: ignore[misc]
        """Use TorchAO in training and ordinary linear evaluation, as in nanochat."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Evaluate held-out tokens without FP8 quantization."""
            if self.training:
                return cast(torch.Tensor, super().forward(x))
            return F.linear(x, self.weight, self.bias)

    config = Float8LinearConfig.from_recipe_name(recipe)
    for name, module in list(model.named_modules()):
        if (
            isinstance(module, nn.Linear)
            and min(module.in_features, module.out_features) >= 128
            and module.in_features % 16 == 0
            and module.out_features % 16 == 0
        ):
            model.set_submodule(name, Float8Linear.from_float(module, config=config))
