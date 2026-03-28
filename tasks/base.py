"""
Base classes for all Tasks.

A Task is a dataset of conversations with optional evaluation criteria.
Taken from nanochat/tasks/common.py; renamed to tasks/base.py.
"""

import random


class Task:
    """Base class; supports lightweight slicing over an underlying dataset."""

    def __init__(self, start: int = 0, stop: int | None = None, step: int = 1) -> None:
        assert start >= 0, f"start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"stop must be >= start"
        assert step >= 1, f"step must be >= 1, got {step}"
        self.start = start
        self.stop = stop
        self.step = step

    @property
    def eval_type(self) -> str:
        """'generative' or 'categorical'."""
        raise NotImplementedError

    def num_examples(self) -> int:
        raise NotImplementedError

    def get_example(self, index: int) -> dict:
        raise NotImplementedError

    def evaluate(self, problem: dict, completion: str) -> bool:
        raise NotImplementedError

    def __len__(self) -> int:
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        span = stop - start
        return max(0, (span + self.step - 1) // self.step)

    def __getitem__(self, index: int) -> dict:
        assert isinstance(index, int), f"Index must be int, got {type(index)}"
        physical = self.start + index * self.step
        return self.get_example(physical)


class TaskMixture(Task):
    """
    Uniformly-shuffled mixture of multiple Task objects.

    Pass a task multiple times to oversample it.
    """

    def __init__(self, tasks: list[Task], **kwargs) -> None:
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(t) for t in tasks]
        self.num_conversations = sum(self.lengths)
        self.index_map: list[tuple[int, int]] = []
        for task_idx, task_len in enumerate(self.lengths):
            for local_idx in range(task_len):
                self.index_map.append((task_idx, local_idx))
        rng = random.Random(42)
        rng.shuffle(self.index_map)

    def num_examples(self) -> int:
        return self.num_conversations

    def get_example(self, index: int) -> dict:
        assert 0 <= index < self.num_conversations
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """Sequential concatenation of multiple Task objects."""

    def __init__(self, tasks: list[Task], **kwargs) -> None:
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(t) for t in tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self) -> int:
        return self.num_conversations

    def get_example(self, index: int) -> dict:
        assert 0 <= index < self.num_conversations
        for task_idx, task_len in enumerate(self.lengths):
            if index < task_len:
                return self.tasks[task_idx][index]
            index -= task_len
        raise IndexError(f"Index out of range: {index}")


def render_mc(question: str, letters: tuple | list, choices: list[str]) -> str:
    """
    Render a multiple-choice question in the standard tinygpt format.

    The letter appears *after* the choice text (better for small models).
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices))
    query += "\nRespond only with the letter of the correct answer."
    return query
