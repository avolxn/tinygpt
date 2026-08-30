"""SFT task construction shared by finetune and distillation scripts."""

from __future__ import annotations

from tasks.base import Task
from tasks.customjson import CustomJSON
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SimpleSpelling, SpellingBee


def build_sft_task_lists(
    task_names: set[str],
    *,
    identity_conversations: str,
    mmlu_epochs: int,
    gsm8k_epochs: int,
) -> tuple[list[Task], list[Task]]:
    """Build reference-compatible train/eval task lists.

    The default full mixture evaluates on SmolTalk/MMLU/GSM8K for the full task mixture.
    Narrow local runs, for example identity-only smoke tests, evaluate on the
    same narrow task set so they don't fetch unrelated datasets.
    """
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
