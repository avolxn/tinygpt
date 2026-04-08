"""
Tests for online distillation helpers.
"""

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


def test_validate_teacher_tokenizer_compatibility_accepts_matching_tokenizers() -> None:
    student = FakeTokenizer(vocab_size=64)
    teacher = FakeTokenizer(vocab_size=64)
    validate_teacher_tokenizer_compatibility(student, teacher)


def test_validate_teacher_tokenizer_compatibility_rejects_vocab_mismatch() -> None:
    student = FakeTokenizer(vocab_size=64)
    teacher = FakeTokenizer(vocab_size=65)

    try:
        validate_teacher_tokenizer_compatibility(student, teacher)
    except ValueError as exc:
        assert "vocab sizes differ" in str(exc)
    else:
        raise AssertionError("Expected compatibility check to fail on vocab mismatch")


def test_validate_teacher_tokenizer_compatibility_rejects_special_token_mismatch() -> None:
    student = FakeTokenizer(vocab_size=64)
    teacher = FakeTokenizer(vocab_size=64, offset=1)

    try:
        validate_teacher_tokenizer_compatibility(student, teacher)
    except ValueError as exc:
        assert "special token ids differ" in str(exc)
    else:
        raise AssertionError("Expected compatibility check to fail on special token mismatch")
