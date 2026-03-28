"""
Tests for the GPT model: forward pass, tensor shapes, weight init, generate.
"""

import pytest
import torch

from tinygpt.config import GPTConfig
from tinygpt.model import GPT


@pytest.fixture()
def tiny_config() -> GPTConfig:
    """A tiny GPT config that runs quickly on CPU."""
    return GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )


@pytest.fixture()
def tiny_model(tiny_config: GPTConfig) -> GPT:
    model = GPT(tiny_config)
    model.init_weights()
    model.eval()
    return model


def test_forward_no_targets(tiny_model: GPT, tiny_config: GPTConfig) -> None:
    """Forward without targets returns logits of correct shape."""
    B, T = 2, 16
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))
    logits = tiny_model(idx)
    assert logits.shape == (B, T, tiny_config.vocab_size), f"Unexpected logits shape: {logits.shape}"


def test_forward_with_targets(tiny_model: GPT, tiny_config: GPTConfig) -> None:
    """Forward with targets returns a scalar mean loss."""
    B, T = 2, 16
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_config.vocab_size, (B, T))
    loss = tiny_model(idx, targets)
    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"


def test_forward_loss_reduction_none(tiny_model: GPT, tiny_config: GPTConfig) -> None:
    """loss_reduction='none' returns per-token losses."""
    B, T = 2, 16
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_config.vocab_size, (B, T))
    loss = tiny_model(idx, targets, loss_reduction="none")
    assert loss.shape == (B, T), f"Expected ({B}, {T}), got {loss.shape}"


def test_forward_ignore_index(tiny_model: GPT, tiny_config: GPTConfig) -> None:
    """Targets with -1 (ignore_index) don't raise errors."""
    B, T = 2, 16
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_config.vocab_size, (B, T))
    targets[0, :5] = -1  # mask first 5 tokens of first batch
    loss = tiny_model(idx, targets)
    assert loss.item() > 0


def test_init_weights_not_nan(tiny_model: GPT) -> None:
    """No NaN values after init_weights()."""
    for name, param in tiny_model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in parameter: {name}"


def test_estimate_flops(tiny_model: GPT) -> None:
    """estimate_flops() returns a positive integer."""
    flops = tiny_model.estimate_flops()
    assert isinstance(flops, int)
    assert flops > 0


def test_num_scaling_params(tiny_model: GPT) -> None:
    """num_scaling_params() returns a dict summing to total param count."""
    counts = tiny_model.num_scaling_params()
    total = sum(p.numel() for p in tiny_model.parameters())
    assert counts["total"] == total


def test_naive_generate(tiny_model: GPT, tiny_config: GPTConfig) -> None:
    """model.generate() yields tokens."""
    tokens = [0, 1, 2, 3]
    generated = list(tiny_model.generate(tokens, max_tokens=5, temperature=0))
    assert len(generated) == 5
    for tok in generated:
        assert 0 <= tok < tiny_config.vocab_size


def test_window_sizes_last_layer_full(tiny_config: GPTConfig) -> None:
    """The last layer always gets full context regardless of window_pattern."""
    for pattern in ("L", "SL", "SSSL"):
        config = GPTConfig(
            sequence_len=64,
            vocab_size=256,
            n_layer=4,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            window_pattern=pattern,
        )
        model = GPT(config)
        last_window = model.window_sizes[-1]
        assert last_window == (64, 0), f"Last window wrong for pattern {pattern}: {last_window}"
