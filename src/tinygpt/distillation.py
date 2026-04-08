"""
Helpers for online teacher-student distillation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from tinygpt.checkpoint import build_model_from_checkpoint
from tinygpt.tokenizer import SPECIAL_TOKENS

_TOKENIZER_PROBES = (
    "Hello world!",
    "The quick brown fox jumps over 13 lazy dogs.\n",
    "2 + 2 = 4, 17 + 5 = 22.",
    "Whitespace:\n  indented line\n\nlast line.",
    "Punctuation: ()[]{}.,!?-_'\"",
)


def load_teacher_model(
    model_ref: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load an inference-only teacher model from a local directory or Hub repo."""
    model, metadata = build_model_from_checkpoint(model_ref, device=device, phase="eval")
    for param in model.parameters():
        param.requires_grad_(False)
    return model, metadata


def validate_teacher_tokenizer_compatibility(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
) -> None:
    """Raise if student/teacher tokenizers are not compatible for logit distillation."""
    student_vocab = student_tokenizer.get_vocab_size()
    teacher_vocab = teacher_tokenizer.get_vocab_size()
    if student_vocab != teacher_vocab:
        raise ValueError(
            "Teacher/student vocab sizes differ. "
            f"student={student_vocab}, teacher={teacher_vocab}. "
            "Online KL distillation requires the exact same tokenizer."
        )

    for token in SPECIAL_TOKENS:
        student_id = student_tokenizer.encode_special(token)
        teacher_id = teacher_tokenizer.encode_special(token)
        if student_id != teacher_id:
            raise ValueError(
                "Teacher/student special token ids differ for "
                f"{token!r}: student={student_id}, teacher={teacher_id}. "
                "Online KL distillation requires identical token ids."
            )

    for probe in _TOKENIZER_PROBES:
        student_ids = student_tokenizer.encode(probe)
        teacher_ids = teacher_tokenizer.encode(probe)
        if student_ids != teacher_ids:
            raise ValueError(
                "Teacher/student tokenizers produce different ids on a probe string. "
                "Use the exact same tokenizer and id mapping for online KL distillation."
            )


def masked_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute masked KL distillation loss over supervised tokens only."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Teacher/student logits must have identical shape, got "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )

    valid_mask = labels.ne(-1)
    if not bool(valid_mask.any()):
        return student_logits.new_zeros(())

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    per_token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return per_token_kl.masked_select(valid_mask).mean() * (temperature**2)
