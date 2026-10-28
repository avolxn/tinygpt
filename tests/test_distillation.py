"""Tests for online distillation helpers."""

import pytest
import torch

from tinygpt.distillation import masked_distillation_loss, validate_teacher_tokenizer_compatibility
from tinygpt.tokenizer import SPECIAL_TOKENS


class FakeTokenizer:
    def __init__(self, vocab_size: int = 32, offset: int = 0) -> None:
        self._vocab_size = vocab_size
        self._specials = {token: i + offset for i, token in enumerate(SPECIAL_TOKENS)}

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def encode_special(self, text: str) -> int | None:
        return self._specials.get(text)

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % 17 for ch in text]


def test_masked_distillation_loss_zero_for_matching_logits() -> None:
    logits = torch.randn(2, 3, 5)
    labels = torch.tensor([[1, 2, -1], [3, 4, 0]])
    loss = masked_distillation_loss(logits, logits.clone(), labels, temperature=1.0)
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_masked_distillation_loss_ignores_masked_positions() -> None:
    student = torch.tensor([[[3.0, 1.0], [0.0, 0.0]]])
    teacher = torch.tensor([[[1.0, 3.0], [9.0, -9.0]]])
    labels = torch.tensor([[1, -1]])

    masked = masked_distillation_loss(student, teacher, labels, temperature=1.0)
    expected = masked_distillation_loss(student[:, :1], teacher[:, :1], torch.tensor([[1]]), temperature=1.0)
    torch.testing.assert_close(masked, expected)


def test_chunked_distillation_matches_full_loss_and_gradients() -> None:
    student = torch.randn(2, 1050, 7, requires_grad=True)
    teacher = torch.randn_like(student)
    labels = torch.randint(-1, 7, (2, 1050))
    temperature = 1.5
    actual = masked_distillation_loss(student, teacher, labels, temperature)
    per_token = torch.nn.functional.kl_div(
        torch.log_softmax(student / temperature, -1), torch.softmax(teacher / temperature, -1), reduction="none"
    ).sum(-1)
    expected = per_token[labels != -1].mean() * temperature**2
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, student, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, student)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_validate_teacher_tokenizer_compatibility_accepts_matching_tokenizers() -> None:
    student = FakeTokenizer(vocab_size=64)
    teacher = FakeTokenizer(vocab_size=64)
    validate_teacher_tokenizer_compatibility(student, teacher)


def test_validate_teacher_tokenizer_compatibility_rejects_vocab_mismatch() -> None:
    student = FakeTokenizer(vocab_size=64)
    teacher = FakeTokenizer(vocab_size=65)

    with pytest.raises(ValueError, match="vocab sizes differ"):
        validate_teacher_tokenizer_compatibility(student, teacher)


def test_validate_teacher_tokenizer_compatibility_rejects_special_token_mismatch() -> None:
    student = FakeTokenizer(vocab_size=64)
    teacher = FakeTokenizer(vocab_size=64, offset=1)

    with pytest.raises(ValueError, match="special token ids differ"):
        validate_teacher_tokenizer_compatibility(student, teacher)
