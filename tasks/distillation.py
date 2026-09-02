"""Task construction shared by the distillation pipeline."""

from __future__ import annotations

from tasks.base import Task
from tasks.customjson import CustomJSON
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SimpleSpelling, SpellingBee


def build_distillation_tasks(
    task_names: set[str],
    *,
    identity_conversations: str,
    mmlu_epochs: int,
    gsm8k_epochs: int,
) -> tuple[list[Task], list[Task]]:
    """Build train and validation task lists for distillation."""
    allowed = {"smoltalk", "mmlu", "gsm8k", "identity", "spelling"}
    unknown = task_names - allowed
    if unknown:
        raise ValueError(f"Unknown distillation tasks: {', '.join(sorted(unknown))}")
    if mmlu_epochs < 1 or gsm8k_epochs < 1:
        raise ValueError("task epoch multipliers must be positive")

    train_tasks: list[Task] = []

    if "smoltalk" in task_names:
        train_tasks.append(SmolTalk(split="train"))

    if "identity" in task_names and identity_conversations:
        train_tasks += [CustomJSON(filepath=identity_conversations)] * 2

    if "mmlu" in task_names:
        train_tasks += [MMLU(subset="all", split="auxiliary_train")] * mmlu_epochs

    if "gsm8k" in task_names:
        train_tasks += [GSM8K(subset="main", split="train")] * gsm8k_epochs

    if "spelling" in task_names:
        train_tasks += [
            SimpleSpelling(size=200000, split="train"),
            SpellingBee(size=80000, split="train"),
        ]

    val_tasks: list[Task] = []
    default_eval_names = {"smoltalk", "mmlu", "gsm8k"}
    if default_eval_names.issubset(task_names):
        val_tasks = [
            SmolTalk(split="test"),
            MMLU(subset="all", split="test", stop=5200),
            GSM8K(subset="main", split="test", stop=420),
        ]
    else:
        if "smoltalk" in task_names:
            val_tasks.append(SmolTalk(split="test"))
        if "identity" in task_names and identity_conversations:
            val_tasks.append(CustomJSON(filepath=identity_conversations))
        if "mmlu" in task_names:
            val_tasks.append(MMLU(subset="all", split="test", stop=5200))
        if "gsm8k" in task_names:
            val_tasks.append(GSM8K(subset="main", split="test", stop=420))

    return train_tasks, val_tasks
