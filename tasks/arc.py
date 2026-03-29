"""
ARC (AI2 Reasoning Challenge) task.
https://huggingface.co/datasets/allenai/ai2_arc
"""

from typing import Any

from datasets import load_dataset

from tasks.base import Task, render_mc


class ARC(Task):
    """Multiple-choice science questions from the ARC dataset.

    Args:
        subset: Either 'ARC-Easy' or 'ARC-Challenge'.
        split: One of 'train', 'validation', or 'test'.
        **kwargs: Forwarded to Task.__init__ (start, stop, step).
    """

    def __init__(self, subset: str, split: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert subset in ("ARC-Easy", "ARC-Challenge"), "subset must be ARC-Easy or ARC-Challenge"
        assert split in ("train", "validation", "test"), "split must be train|validation|test"
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self) -> str:
        """Return 'categorical' — ARC is evaluated by argmax over letter logits."""
        return "categorical"

    def num_examples(self) -> int:
        """Return the number of examples in the loaded split.

        Returns:
            Dataset length before any slicing.
        """
        return len(self.ds)

    def get_example(self, index: int) -> dict[str, Any]:
        """Return a single ARC example as a conversation dict.

        Args:
            index: Physical index into the dataset.

        Returns:
            Dict with 'messages' (user/assistant) and 'letters' keys.
        """
        row = self.ds[index]
        question: str = row["question"]
        choices: list[str] = row["choices"]["text"]
        letters: list[str] = row["choices"]["label"]
        answer: str = row["answerKey"]
        assert answer in letters, f"ARC answer {answer!r} not in {letters}"
        return {
            "messages": [
                {"role": "user", "content": render_mc(question, letters, choices)},
                {"role": "assistant", "content": answer},
            ],
            "letters": letters,
        }

    def evaluate(self, problem: dict[str, Any], completion: str) -> bool:
        """Check whether the predicted letter matches the ground-truth answer.

        Args:
            problem: Conversation dict from get_example.
            completion: Predicted letter (e.g. 'A').

        Returns:
            True if completion equals the correct answer letter.
        """
        assert completion in problem["letters"], f"ARC answer {completion!r} must be one of {problem['letters']}"
        return completion == problem["messages"][-1]["content"]
