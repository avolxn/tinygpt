"""
Base classes for all Tasks.

A Task is a dataset of conversations with optional evaluation criteria.
Taken from nanochat/tasks/common.py; renamed to tasks/base.py.
"""

import random
from typing import Any


class Task:
    """Base class; supports lightweight slicing over an underlying dataset."""

    def __init__(self, start: int = 0, stop: int | None = None, step: int = 1) -> None:
        """Initialise slice parameters for the task.

        Args:
            start: First physical index to include.
            stop: One past the last physical index; None means use all examples.
            step: Stride between successive physical indices.
        """
        assert start >= 0, f"start must be non-negative, got {start}"
        assert stop is None or stop >= start, "stop must be >= start"
        assert step >= 1, f"step must be >= 1, got {step}"
        self.start = start
        self.stop = stop
        self.step = step

    @property
    def eval_type(self) -> str:
        """Return the evaluation strategy: 'generative' or 'categorical'."""
        raise NotImplementedError

    def num_examples(self) -> int:
        """Return the total number of examples in the underlying dataset.

        Returns:
            Integer count of examples before any slicing is applied.
        """
        raise NotImplementedError

    def get_example(self, index: int) -> dict[str, Any]:
        """Return a single example by its physical index.

        Args:
            index: Physical (unsliced) index into the dataset.

        Returns:
            A conversation dict for the requested example.
        """
        raise NotImplementedError

    def evaluate(self, problem: dict[str, Any], completion: str) -> bool:
        """Check whether a model completion is correct for the given problem.

        Args:
            problem: The example dict as returned by get_example.
            completion: The model's generated answer string.

        Returns:
            True if the completion is considered correct.
        """
        raise NotImplementedError

    def reward(self, problem: dict[str, Any], completion: str) -> float:
        """Return a reward signal for RL training.

        By default, this is 1.0 if the completion is correct, 0.0 otherwise.
        Subclasses can override for more sophisticated reward schemes.

        Args:
            problem: The example dict as returned by get_example.
            completion: The model's generated answer string.

        Returns:
            A float reward signal (typically 0.0 or 1.0).
        """
        is_correct = self.evaluate(problem, completion)
        return float(is_correct)

    def __len__(self) -> int:
        """Return the number of examples after slicing.

        Returns:
            Number of examples accessible through this Task's slice.
        """
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        span = stop - start
        return max(0, (span + self.step - 1) // self.step)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the example at a logical index within the slice.

        Args:
            index: Logical index (0-based within the sliced view).

        Returns:
            The conversation dict at the corresponding physical index.
        """
        assert isinstance(index, int), f"Index must be int, got {type(index)}"
        physical = self.start + index * self.step
        return self.get_example(physical)


class TaskMixture(Task):
    """
    Uniformly-shuffled mixture of multiple Task objects.

    Pass a task multiple times to oversample it.
    """

    def __init__(self, tasks: list[Task], **kwargs: Any) -> None:
        """Initialise the mixture by building a shuffled flat index map.

        Args:
            tasks: List of Task objects to mix; repeat a task to oversample it.
            **kwargs: Forwarded to Task.__init__ (start, stop, step).
        """
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
        """Return the total number of examples across all mixed tasks.

        Returns:
            Sum of all task lengths.
        """
        return self.num_conversations

    def get_example(self, index: int) -> dict[str, Any]:
        """Return the example at a physical index in the shuffled mixture.

        Args:
            index: Physical index into the shuffled flat index map.

        Returns:
            The conversation dict from the appropriate sub-task.
        """
        assert 0 <= index < self.num_conversations
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """Sequential concatenation of multiple Task objects."""

    def __init__(self, tasks: list[Task], **kwargs: Any) -> None:
        """Initialise the sequence from a list of tasks.

        Args:
            tasks: List of Task objects to concatenate in order.
            **kwargs: Forwarded to Task.__init__ (start, stop, step).
        """
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(t) for t in tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self) -> int:
        """Return the total number of examples across all tasks in sequence.

        Returns:
            Sum of all task lengths.
        """
        return self.num_conversations

    def get_example(self, index: int) -> dict[str, Any]:
        """Return the example at a physical index in the concatenated sequence.

        Args:
            index: Physical index into the concatenated sequence.

        Returns:
            The conversation dict from the appropriate sub-task.

        Raises:
            IndexError: If index is out of range.
        """
        assert 0 <= index < self.num_conversations
        for task_idx, task_len in enumerate(self.lengths):
            if index < task_len:
                return self.tasks[task_idx][index]
            index -= task_len
        raise IndexError(f"Index out of range: {index}")


def render_mc(question: str, letters: tuple[str, ...] | list[str], choices: list[str]) -> str:
    """Render a multiple-choice question in the standard tinygpt format.

    The letter appears after the choice text (better for small models).

    Args:
        question: The question text.
        letters: Sequence of answer letter labels (e.g. ('A', 'B', 'C', 'D')).
        choices: Sequence of choice text strings, parallel to letters.

    Returns:
        A formatted multiple-choice prompt string.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices, strict=True))
    query += "\nRespond only with the letter of the correct answer."
    return query
