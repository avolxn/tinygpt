"""Tests for the Transformers causal language model."""

import pytest
import torch

from tinygpt.config import TinyGPTConfig
from tinygpt.model import TinyGPTForCausalLM


@pytest.fixture()
def tiny_config() -> TinyGPTConfig:
    return TinyGPTConfig(
        max_position_embeddings=64,
        vocab_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=256,
    )


@pytest.fixture()
def tiny_model(tiny_config: TinyGPTConfig) -> TinyGPTForCausalLM:
    model = TinyGPTForCausalLM(tiny_config)
    model.eval()
    return model


def test_forward_no_labels(tiny_model: TinyGPTForCausalLM, tiny_config: TinyGPTConfig) -> None:
    idx = torch.randint(0, tiny_config.vocab_size, (2, 16))
    output = tiny_model(input_ids=idx)
    assert output.logits.shape == (2, 16, tiny_config.vocab_size)


def test_forward_with_labels(tiny_model: TinyGPTForCausalLM, tiny_config: TinyGPTConfig) -> None:
    idx = torch.randint(0, tiny_config.vocab_size, (2, 16))
    labels = torch.randint(0, tiny_config.vocab_size, (2, 16))
    output = tiny_model(input_ids=idx, labels=labels)
    assert output.loss is not None
    assert output.loss.shape == ()
    assert output.loss.item() > 0


def test_causal_prefix_invariance(tiny_model: TinyGPTForCausalLM, tiny_config: TinyGPTConfig) -> None:
    idx = torch.randint(0, tiny_config.vocab_size, (1, 12))
    tiny_model.smear_lambda.data.fill_(0.3)
    for layer in tiny_model.layers:
        torch.nn.init.normal_(layer.self_attn.o_proj.weight, std=0.03)
    full = tiny_model(idx).logits
    prefix = tiny_model(idx[:, :7]).logits
    torch.testing.assert_close(full[:, :7], prefix)


def test_init_weights_not_nan(tiny_model: TinyGPTForCausalLM) -> None:
    for name, param in tiny_model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in parameter: {name}"
