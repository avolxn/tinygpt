"""Transformers-compatible causal language model used by tinygpt."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from tinygpt.config import GPTConfig


class GPT(LlamaForCausalLM):  # type: ignore[no-untyped-call]
    """Training-facing wrapper around Hugging Face's native Llama model.

    The dataloaders provide already-shifted labels, so this wrapper computes
    loss directly against the returned logits. Saved checkpoints advertise
    ``LlamaForCausalLM`` and can be loaded by vLLM without custom code.
    """

    config_class = GPTConfig  # type: ignore[assignment]

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)  # type: ignore[no-untyped-call]

    def init_weights(self) -> None:
        """Initialize weights using Transformers' canonical initializer."""
        super().init_weights()  # type: ignore[no-untyped-call]

    @property
    def window_sizes(self) -> list[tuple[int, int]]:
        """Return window metadata retained for run introspection."""
        long_window = self.config.sequence_len
        short_window = math.ceil(long_window / 4 / 128) * 128
        pattern = self.config.window_pattern.upper()
        windows = [
            (long_window if pattern[i % len(pattern)] == "L" else short_window, 0)
            for i in range(self.config.n_layer)
        ]
        windows[-1] = (long_window, 0)
        return windows

    def get_device(self) -> torch.device:
        embedding = cast(nn.Embedding, self.get_input_embeddings())
        return embedding.weight.device

    def num_scaling_params(self) -> dict[str, int]:
        """Return parameter counts using the original scaling-law buckets."""
        embedding = cast(nn.Embedding, self.get_input_embeddings()).weight.numel()
        lm_head = self.lm_head.weight.numel()
        transformer = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith("model.layers.")
        )
        scalars = sum(parameter.numel() for parameter in self.parameters() if parameter.dim() < 2)
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "wte": embedding,
            "value_embeds": 0,
            "lm_head": lm_head,
            "transformer_matrices": transformer,
            "scalars": scalars,
            "total": total,
        }

    def forward(
        self,
        idx: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return logits or loss for tinygpt's pre-batched labels."""
        if idx is None:
            idx = kwargs.pop("input_ids")
        if targets is None:
            targets = kwargs.pop("labels", None)

        output = super().forward(input_ids=idx, labels=None, **kwargs)
        logits = cast(torch.Tensor, output.logits)
        if targets is None:
            return logits
        if loss_reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unknown loss reduction: {loss_reduction}")
        if loss_reduction == "mean" and not bool(targets.ne(-1).any()):
            return logits.sum() * 0.0
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        return loss.reshape(targets.shape) if loss_reduction == "none" else loss
__all__ = ["GPT", "GPTConfig", "LlamaDecoderLayer"]
