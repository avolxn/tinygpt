"""Tests for the Transformers causal language model."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture()
def tiny_config() -> LlamaConfig:
    return LlamaConfig(
        max_position_embeddings=64,
        vocab_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=256,
    )


@pytest.fixture()
def tiny_model(tiny_config: LlamaConfig) -> LlamaForCausalLM:
    model = LlamaForCausalLM(tiny_config)
    model.eval()
    return model


def test_forward_no_labels(tiny_model: LlamaForCausalLM, tiny_config: LlamaConfig) -> None:
    idx = torch.randint(0, tiny_config.vocab_size, (2, 16))
    output = tiny_model(input_ids=idx)
    assert output.logits.shape == (2, 16, tiny_config.vocab_size)


def test_forward_with_labels(tiny_model: LlamaForCausalLM, tiny_config: LlamaConfig) -> None:
    idx = torch.randint(0, tiny_config.vocab_size, (2, 16))
    labels = torch.randint(0, tiny_config.vocab_size, (2, 16))
    output = tiny_model(input_ids=idx, labels=labels)
    assert output.loss is not None
    assert output.loss.shape == ()
    assert output.loss.item() > 0


def test_generation_api(tiny_model: LlamaForCausalLM, tiny_config: LlamaConfig) -> None:
    idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
    generated = tiny_model.generate(idx, max_new_tokens=3)
    assert generated.shape == (1, 7)


def test_init_weights_not_nan(tiny_model: LlamaForCausalLM) -> None:
    for name, param in tiny_model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in parameter: {name}"
