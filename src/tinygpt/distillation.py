"""Helpers for online teacher-student distillation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

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
    """Load an inference-only teacher from a prepared local directory."""
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_ref,
        local_files_only=True,
        dtype=torch.float16 if device.type == "cuda" and not torch.cuda.is_bf16_supported() else "auto",
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, {"model_config": model.config.to_dict(), "model_dir": model_ref}


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

    student_get_vocab = getattr(student_tokenizer, "get_vocab", None)
    teacher_get_vocab = getattr(teacher_tokenizer, "get_vocab", None)
    if callable(student_get_vocab) and callable(teacher_get_vocab):
        if student_get_vocab() != teacher_get_vocab():
            raise ValueError("Teacher/student tokenizers have different token-to-id mappings")

    for attribute in ("bos_token_id", "eos_token_id", "chat_template"):
        student = getattr(getattr(student_tokenizer, "tokenizer", student_tokenizer), attribute, None)
        teacher = getattr(getattr(teacher_tokenizer, "tokenizer", teacher_tokenizer), attribute, None)
        if student != teacher:
            raise ValueError(
                f"Teacher/student tokenizers differ in {attribute}; a different tokenizer requires a new student"
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

    def chunk_loss(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(student.float() / temperature, dim=-1)
        probabilities = F.softmax(teacher.float() / temperature, dim=-1)
        per_token = F.kl_div(log_probs, probabilities, reduction="none").sum(dim=-1)
        return per_token.masked_select(mask).sum()

    size = student_logits.size(-1)
    loss = student_logits.new_zeros(())
    for student, teacher, mask in zip(
        torch.split(student_logits.reshape(-1, size), 1024),
        torch.split(teacher_logits.reshape(-1, size), 1024),
        torch.split(valid_mask.reshape(-1), 1024),
        strict=True,
    ):
        loss = loss + checkpoint(chunk_loss, student, teacher, mask, use_reentrant=False, preserve_rng_state=False)
    return loss / valid_mask.sum() * (temperature**2)
