"""Nanochat's conversational task mixture, used for teacher-student distillation."""

from __future__ import annotations

from tasks.base import Task
from tasks.customjson import CustomJSON
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk


def build_distillation_tasks(
    task_names: set[str],
    *,
    conversations: str = "",
    mmlu_epochs: int = 3,
    gsm8k_epochs: int = 4,
) -> tuple[list[Task], list[Task]]:
    """Build nanochat's train/validation mixture or an explicit local test corpus."""
    unknown = task_names - {"smoltalk", "mmlu", "gsm8k", "custom"}
    if unknown:
        raise ValueError(f"Unknown distillation tasks: {', '.join(sorted(unknown))}")
    if mmlu_epochs < 1 or gsm8k_epochs < 1:
        raise ValueError("task epoch multipliers must be positive")
    train: list[Task] = []
    validation: list[Task] = []
    if "smoltalk" in task_names:
        train.append(SmolTalk(split="train"))
        validation.append(SmolTalk(split="test"))
    if "mmlu" in task_names:
        train.extend([MMLU(subset="all", split="auxiliary_train")] * mmlu_epochs)
        validation.append(MMLU(subset="all", split="test", stop=5200))
    if "gsm8k" in task_names:
        train.extend([GSM8K(subset="main", split="train")] * gsm8k_epochs)
        validation.append(GSM8K(subset="main", split="test", stop=420))
    if "custom" in task_names:
        if not conversations:
            raise ValueError("--conversations is required for the custom task")
        local = CustomJSON(filepath=conversations)
        train.append(local)
        validation.append(local)
    return train, validation
