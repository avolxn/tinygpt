"""GPT model and shared training-run configuration."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from transformers import LlamaConfig


@dataclass(init=False)
class GPTConfig(LlamaConfig):
    """Llama configuration with tinygpt's concise architecture aliases."""

    model_type = "llama"
    sequence_len: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    window_pattern: str

    def __init__(
        self,
        sequence_len: int | None = None,
        vocab_size: int | None = None,
        n_layer: int | None = None,
        n_head: int | None = None,
        n_kv_head: int | None = None,
        n_embd: int | None = None,
        window_pattern: str = "SSSL",
        **kwargs: Any,
    ) -> None:
        hf_sequence_len = kwargs.pop("max_position_embeddings", None)
        hf_vocab_size = kwargs.pop("vocab_size", None)
        hf_n_layer = kwargs.pop("num_hidden_layers", None)
        hf_n_head = kwargs.pop("num_attention_heads", None)
        hf_n_kv_head = kwargs.pop("num_key_value_heads", None)
        hf_n_embd = kwargs.pop("hidden_size", None)
        intermediate_size = int(kwargs.pop("intermediate_size", 4 * (n_embd or hf_n_embd or 768)))
        sequence_len = sequence_len or int(hf_sequence_len or 2048)
        vocab_size = vocab_size or int(hf_vocab_size or 32768)
        n_layer = n_layer or int(hf_n_layer or 12)
        n_head = n_head or int(hf_n_head or 6)
        n_kv_head = n_kv_head or int(hf_n_kv_head or n_head)
        n_embd = n_embd or int(hf_n_embd or 768)

        super().__init__(  # type: ignore[no-untyped-call]
            vocab_size=vocab_size,
            max_position_embeddings=sequence_len,
            hidden_size=n_embd,
            intermediate_size=intermediate_size,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_kv_head,
            **kwargs,
        )
        self.sequence_len = sequence_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.window_pattern = window_pattern


@dataclass(frozen=True)
class RuntimeConfig:
    """Common runtime settings shared by all training entry points."""

    run: str = ""
    device_type: str = ""
    tokenizer_dir: str = "data/tokenizer"
    out_dir: str = "data"
    run_name: str = ""
    seed: int = 42

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> RuntimeConfig:
        """Build the shared config from an argparse namespace."""
        return cls(
            run=str(args.run),
            device_type=str(args.device_type),
            tokenizer_dir=str(args.tokenizer_dir),
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
    parser.add_argument("--run", type=str, default=RuntimeConfig.run, help="W&B run name")
    parser.add_argument(
        "--device-type", type=str, default=RuntimeConfig.device_type, help="cuda|cpu|mps (empty = autodetect)"
    )
    parser.add_argument("--tokenizer-dir", type=str, default=RuntimeConfig.tokenizer_dir)
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
    return int(2 ** round(math.log2(predicted_batch_size)))


def compute_scaled_weight_decay(
    *,
    base_weight_decay: float,
    total_batch_size: int,
    target_tokens: int,
    d12_target_tokens: float,
) -> float:
    """Return the reference scaled pretraining weight decay."""
    return base_weight_decay * math.sqrt(total_batch_size / REFERENCE_BATCH_SIZE) * (d12_target_tokens / target_tokens)
