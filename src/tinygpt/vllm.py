"""Nanochat inference using vLLM attention, tensor parallelism, and state caches.

Smear is a fixed two-tap convolution [1, 0] of normalized input embeddings.
The native short-convolution cache tracks its previous embedding per request,
including interleaved decoding and chunked prefill. No request state is kept in
Python. All other additions are stateless nanochat architecture components.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator, MambaStateDtypeCalculator
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.utils.torch_utils import direct_register_custom_op

from tinygpt.model import norm


class PreviousEmbedding(nn.Module, MambaBase):  # type: ignore[misc]
    """Retrieve the preceding embedding through vLLM's native short-conv cache.

    Args:
        config: vLLM runtime configuration.
        prefix: Unique cache layer name.
    """

    def __init__(self, config: VllmConfig, prefix: str) -> None:
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.kv_cache = (torch.tensor([]),)
        self.register_buffer(
            "weight",
            torch.tensor([1.0, 0.0]).expand(config.model_config.hf_config.hidden_size, 2).contiguous(),
            persistent=False,
        )
        get_current_vllm_config().compilation_config.static_forward_context[prefix] = self

    def get_state_shape(self) -> tuple[tuple[int, int]]:
        """Replicate the one-embedding state across tensor-parallel ranks."""
        return ((1, self.config.model_config.hf_config.hidden_size),)

    def get_state_dtype(self) -> tuple[torch.dtype]:
        """Use the model dtype for the native convolution cache."""
        return cast(
            tuple[torch.dtype],
            MambaStateDtypeCalculator.short_conv_state_dtype(
                self.config.model_config.dtype, self.config.cache_config.mamba_cache_dtype
            ),
        )

    @property
    def mamba_type(self) -> str:
        """Select vLLM's short-convolution attention metadata backend."""
        return "short_conv"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read and advance the preceding embedding for each active request."""
        output = torch.empty_like(x)
        torch.ops.vllm.tinygpt_previous_embedding(x, output, self.prefix)
        return output


def previous_embedding(x: torch.Tensor, output: torch.Tensor, prefix: str) -> None:
    """Dispatch prefill/decode to native cache-aware convolution kernels."""
    context = get_forward_context()
    layer = context.no_compile_layers[prefix]
    if context.attn_metadata is None:
        output.zero_()
        return
    metadata = context.attn_metadata[prefix]
    state = layer.kv_cache[0].transpose(-1, -2)
    decodes = metadata.num_decode_tokens
    prefills = metadata.num_prefill_tokens
    weight = layer.weight.to(x.dtype)
    if decodes:
        # The native update overwrites its input; preserve the current embedding
        # for the smear gate and residual after retrieving the previous one.
        output[:decodes] = causal_conv1d_update(
            x[:decodes].clone(),
            state,
            weight,
            None,
            activation=None,
            conv_state_indices=metadata.state_indices_tensor_d,
        )
    if prefills:
        output[decodes : decodes + prefills] = causal_conv1d_fn(
            x[decodes : decodes + prefills].transpose(0, 1),
            weight,
            None,
            activation=None,
            conv_states=state,
            has_initial_state=metadata.has_initial_states_p,
            cache_indices=metadata.state_indices_tensor_p,
            metadata=metadata,
            query_start_loc=metadata.query_start_loc_p,
        ).transpose(0, 1)[:prefills]


def previous_embedding_fake(x: torch.Tensor, output: torch.Tensor, prefix: str) -> None:
    """Provide tensor shapes to vLLM's graph tracing."""


direct_register_custom_op(
    op_name="tinygpt_previous_embedding",
    op_func=previous_embedding,
    mutates_args=["output"],
    fake_impl=previous_embedding_fake,
)


class Block(nn.Module):
    """Evaluate a nanochat layer with native tensor-parallel projections and attention.

    Args:
        runtime: vLLM runtime configuration.
        index: Transformer layer position.
    """

    def __init__(self, runtime: VllmConfig, index: int, config: Any) -> None:
        super().__init__()
        self.config = config
        self.index = index
        self.head_dim = config.hidden_size // config.num_attention_heads
        size = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % size:
            raise ValueError("Tensor parallelism must divide the query head count")
        if max(size, config.num_key_value_heads) % min(size, config.num_key_value_heads):
            raise ValueError("KV heads must divide the tensor-parallel size or be divisible by it")
        self.heads = config.num_attention_heads // size
        self.kv_heads = max(1, config.num_key_value_heads // size)
        self.kv_start = get_tensor_model_parallel_rank() * config.num_key_value_heads // size
        self.q_size = self.heads * self.head_dim
        self.kv_size = self.kv_heads * self.head_dim
        prefix = f"layers.{index + 1}"
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(config.hidden_size, config.hidden_size, bias=False, prefix=f"{prefix}.o_proj")
        self.fc1 = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, bias=False, prefix=f"{prefix}.fc1"
        )
        self.fc2 = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False, prefix=f"{prefix}.fc2")
        self.ve_gate = (
            ReplicatedLinear(
                min(12, config.hidden_size), config.num_key_value_heads, bias=False, prefix=f"{prefix}.ve_gate"
            )
            if index % 2 == (config.transformer_depth - 1) % 2
            else None
        )
        self.rotary_emb = get_rope(
            self.head_dim, max_position=config.max_position_embeddings, rope_parameters=config.rope_parameters
        )
        window = config.attention_windows()[index]
        self.attn = Attention(
            self.heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.kv_heads,
            cache_config=runtime.cache_config,
            per_layer_sliding_window=window + 1,
            prefix=f"{prefix}.attn",
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor, value_embedding: torch.Tensor | None) -> torch.Tensor:
        """Apply the same QK normalization, value gate, and residuals as training."""
        residual = x
        normalized = norm(x, self.config.rms_norm_eps)
        qkv, _ = self.qkv_proj(normalized)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        if value_embedding is not None:
            assert self.ve_gate is not None
            gate, _ = self.ve_gate(normalized[..., : min(12, self.config.hidden_size)])
            gate = gate.narrow(-1, self.kv_start, self.kv_heads)
            value_embedding = value_embedding.narrow(-1, self.kv_start * self.head_dim, self.kv_size)
            v = v + (3 * torch.sigmoid(gate)).repeat_interleave(self.head_dim, dim=-1) * value_embedding
        # vLLM RoPE rotates in the positive direction; nanochat uses its transpose.
        q = q.view(-1, self.heads, self.head_dim)
        k = k.view(-1, self.kv_heads, self.head_dim)
        q = torch.cat((q[..., : self.head_dim // 2], -q[..., self.head_dim // 2 :]), dim=-1).reshape(-1, self.q_size)
        k = torch.cat((k[..., : self.head_dim // 2], -k[..., self.head_dim // 2 :]), dim=-1).reshape(-1, self.kv_size)
        q, k = self.rotary_emb(positions, q, k)
        q = q.view(-1, self.heads, self.head_dim)
        k = k.view(-1, self.kv_heads, self.head_dim)
        q = torch.cat((q[..., : self.head_dim // 2], -q[..., self.head_dim // 2 :]), dim=-1).reshape(-1, self.q_size)
        k = torch.cat((k[..., : self.head_dim // 2], -k[..., self.head_dim // 2 :]), dim=-1).reshape(-1, self.kv_size)
        q = (norm(q.view(-1, self.heads, self.head_dim), self.config.rms_norm_eps) * 1.2).reshape(-1, self.q_size)
        k = (norm(k.view(-1, self.kv_heads, self.head_dim), self.config.rms_norm_eps) * 1.2).reshape(-1, self.kv_size)
        attention = self.attn(q, k, v)
        attention, _ = self.o_proj(attention)
        x = residual + attention
        hidden, _ = self.fc1(norm(x, self.config.rms_norm_eps))
        hidden, _ = self.fc2(torch.relu(hidden).square())
        return cast(torch.Tensor, x + hidden)


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1})
class TinyGPTForCausalLM(nn.Module, HasInnerState, IsHybrid):  # type: ignore[misc]
    """Expose the nanochat student through vLLM's model plugin protocol.

    Args:
        vllm_config: Engine configuration, supplied by vLLM.
        prefix: Optional model name prefix.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.padded_vocab_size, config.hidden_size)
        self.value_embeds = nn.ModuleDict(
            {
                str(i): VocabParallelEmbedding(
                    config.padded_vocab_size,
                    config.num_key_value_heads * (config.hidden_size // config.num_attention_heads),
                )
                for i in range(config.transformer_depth)
                if i % 2 == (config.transformer_depth - 1) % 2
            }
        )
        self.layers = nn.ModuleList(Block(vllm_config, i, config) for i in range(config.transformer_depth))
        self.lm_head = ParallelLMHead(config.padded_vocab_size, config.hidden_size)
        self.resid_lambdas = nn.Parameter(torch.empty(config.transformer_depth, dtype=torch.float32))
        self.x0_lambdas = nn.Parameter(torch.empty(config.transformer_depth, dtype=torch.float32))
        self.smear_gate = nn.Linear(min(24, config.hidden_size), 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.backout_lambda = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.previous = PreviousEmbedding(vllm_config, "layers.0.smear")
        self.logits_processor = LogitsProcessor(config.vocab_size, soft_cap=15.0)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype]:
        """Expose the preceding-embedding cache dtype to the engine."""
        return cast(
            tuple[torch.dtype],
            MambaStateDtypeCalculator.short_conv_state_dtype(
                vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
            ),
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple[tuple[int, int]]:
        """Expose one full previous embedding per request and TP rank."""
        return ((1, vllm_config.model_config.hf_config.hidden_size),)

    @classmethod
    def get_mamba_state_copy_func(cls) -> Any:
        """Use vLLM's native state copying for short convolutions."""
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings through vLLM's model interface."""
        return cast(torch.Tensor, self.embed_tokens(input_ids))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return final hidden states for interleaved requests and chunked prefill."""
        x = norm(self.embed_input_ids(input_ids), self.config.rms_norm_eps)
        previous = self.previous(x)
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[..., : self.smear_gate.in_features]))
        x = x + gate * previous
        x0, backout = x, None
        for i, layer in enumerate(self.layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            value = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            x = layer(x, positions, value)
            if i == self.config.transformer_depth // 2:
                backout = x
        if backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * backout
        return norm(x, self.config.rms_norm_eps)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Apply the tensor-parallel output projection and reference logit softcap."""
        return cast(torch.Tensor | None, self.logits_processor(self.lm_head, hidden_states))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load training checkpoints through native sharded parameter loaders."""
        parameters = dict(self.named_parameters())
        loaded: set[str] = set()
        mapping = {
            "self_attn.q_proj": ("qkv_proj", "q"),
            "self_attn.k_proj": ("qkv_proj", "k"),
            "self_attn.v_proj": ("qkv_proj", "v"),
        }
        for name, weight in weights:
            for source, (target, shard) in mapping.items():
                if source in name:
                    destination = name.replace(source, target)
                    parameter = parameters[destination]
                    loader = cast(Any, parameter).weight_loader
                    loader(parameter, weight, shard)
                    loaded.add(destination)
                    break
            else:
                destination = name.replace("self_attn.", "").replace("mlp.", "")
                parameter = parameters[destination]
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, weight)
                loaded.add(destination)
        return loaded
