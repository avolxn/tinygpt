"""Nanochat architecture and shared training-run configuration."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
from transformers import PreTrainedConfig

# Architecture and recipe reference, recorded in every student checkpoint.
NANOCHAT_REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"


class TinyGPTConfig(PreTrainedConfig):  # type: ignore[no-untyped-call]
    """Describe the nanochat architecture used by the student.

    Args:
        window_pattern: Repeated short/full attention pattern, with a full final layer.
        transformer_depth: Number of Transformer blocks, excluding the input smear cache.
        **kwargs: Standard Transformers NanoChat configuration fields.
    """

    model_type = "tinygpt"
    num_key_value_heads: int | None

    def __init__(self, window_pattern: str = "SSSL", transformer_depth: int | None = None, **kwargs: Any) -> None:
        depth = transformer_depth if transformer_depth is not None else int(kwargs.get("num_hidden_layers", 12))
        if depth < 1:
            raise ValueError("transformer_depth must be positive")
        # Transformers/vLLM hybrid configs count all cache-bearing blocks:
        # one input convolution, followed by the requested Transformer depth.
        kwargs["num_hidden_layers"] = depth + 1
        kwargs.setdefault("intermediate_size", 4 * int(kwargs.get("hidden_size", 768)))
        # PyTorch 2.10's RMSNorm uses the accumulator epsilon, also for bf16 inputs.
        kwargs.setdefault("rms_norm_eps", torch.finfo(torch.float32).eps)
        defaults = {
            "vocab_size": 50304,
            "hidden_size": 768,
            "num_attention_heads": 6,
            "num_key_value_heads": None,
            "max_position_embeddings": 2048,
            "hidden_act": "relu2",
            "attention_dropout": 0.0,
            "initializer_range": 0.02,
            "attention_bias": False,
            "final_logit_softcapping": 15.0,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 1,
            "tie_word_embeddings": False,
        }
        for name, value in defaults.items():
            kwargs.setdefault(name, value)
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "hidden_act",
            "attention_dropout",
            "rms_norm_eps",
            "initializer_range",
            "attention_bias",
            "final_logit_softcapping",
        ):
            setattr(self, name, kwargs.pop(name))
        super().__init__(**kwargs)
        self.transformer_depth = depth
        self.layers_block_type = ["mamba"] + ["attention"] * depth
        self.num_key_value_heads = self.num_key_value_heads or self.num_attention_heads
        if not window_pattern or set(window_pattern) - {"S", "L"}:
            raise ValueError("window_pattern must contain only S and L")
        self.window_pattern = window_pattern
        self.nanochat_revision = NANOCHAT_REVISION
        self.use_cache = False
        self.padded_vocab_size = math.ceil(self.vocab_size / 64) * 64
        self.rope_parameters = {"rope_type": "default", "rope_theta": 100000.0}

    def attention_windows(self) -> list[int]:
        """Return the inclusive left window of each layer, as in nanochat."""
        short = math.ceil(self.max_position_embeddings / 512) * 128
        windows = [
            short if self.window_pattern[i % len(self.window_pattern)] == "S" else self.max_position_embeddings
            for i in range(self.transformer_depth)
        ]
        windows[-1] = self.max_position_embeddings
        return windows


@dataclass(frozen=True)
class RuntimeConfig:
    """Common runtime settings shared by all training entry points."""

    run: str = ""
    device_type: str = ""
    out_dir: str = "data"
    run_name: str = ""
    seed: int = 42

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> RuntimeConfig:
        """Build the shared config from an argparse namespace."""
        return cls(
            run=str(args.run),
            device_type=str(args.device_type),
            out_dir=str(args.out_dir),
            run_name=str(args.run_name),
            seed=int(args.seed),
        )

    def with_run_name(self, run_name: str) -> RuntimeConfig:
        """Return a copy with a resolved run name."""
        return replace(self, run_name=run_name)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for checkpoint metadata."""
        return asdict(self)


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared runtime arguments to a training CLI parser."""
    parser.add_argument("--run", type=str, default=RuntimeConfig.run, help="MLflow run name")
    parser.add_argument(
        "--device-type", type=str, default=RuntimeConfig.device_type, help="cuda|cpu|mps (empty = autodetect)"
    )
    parser.add_argument("--out-dir", type=str, default=RuntimeConfig.out_dir)
    parser.add_argument("--run-name", type=str, default=RuntimeConfig.run_name)
    parser.add_argument("--seed", type=int, default=RuntimeConfig.seed, help="Random seed for training")


REFERENCE_BATCH_SIZE = 2**19


def make_config(
    depth: int,
    *,
    aspect_ratio: int = 64,
    head_dim: int = 128,
    vocab_size: int = 32768,
    sequence_len: int = 2048,
    window_pattern: str = "SSSL",
) -> TinyGPTConfig:
    """Build the nanochat architecture from a depth scalar.

    model_dim is set to depth * aspect_ratio, rounded up to the next multiple
    of head_dim so that head_dim divides evenly.

    Args:
        depth: Number of transformer layers.
        aspect_ratio: Multiplier for model width relative to depth.
        head_dim: Attention head dimension; model_dim is rounded up to a multiple.
        vocab_size: Vocabulary size.
        sequence_len: Maximum sequence length.
        window_pattern: Repeated short/full attention pattern.

    Returns:
        A Transformers configuration for the nanochat student.
    """
    for name, value in (
        ("depth", depth),
        ("aspect_ratio", aspect_ratio),
        ("head_dim", head_dim),
        ("vocab_size", vocab_size),
        ("sequence_len", sequence_len),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return TinyGPTConfig(
        window_pattern=window_pattern,
        vocab_size=vocab_size,
        max_position_embeddings=sequence_len,
        hidden_size=model_dim,
        intermediate_size=4 * model_dim,
        num_hidden_layers=depth,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        tie_word_embeddings=False,
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
    if scaling_params <= 0 or d12_scaling_params <= 0:
        raise ValueError("scaling parameter counts must be positive")
    if target_param_data_ratio <= 0:
        raise ValueError("target_param_data_ratio must be positive when batch size is inferred")
    target_tokens = target_param_data_ratio * scaling_params
    d12_target_tokens = target_param_data_ratio * d12_scaling_params
    batch_size_ratio = target_tokens / d12_target_tokens
    predicted_batch_size = REFERENCE_BATCH_SIZE * batch_size_ratio**0.383
    return int(2 ** round(math.log2(predicted_batch_size)))


def compute_scaled_weight_decay(
    *,
    base_weight_decay: float,
    total_batch_size: int,
    target_tokens: int,
    d12_target_tokens: float,
) -> float:
    """Return the reference scaled pretraining weight decay."""
    if total_batch_size <= 0 or target_tokens <= 0 or d12_target_tokens <= 0:
        raise ValueError("batch size and target token counts must be positive")
    return base_weight_decay * math.sqrt(total_batch_size / REFERENCE_BATCH_SIZE) * (d12_target_tokens / target_tokens)
