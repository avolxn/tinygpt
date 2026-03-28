"""
HellaSwag commonsense NLI task.
https://huggingface.co/datasets/Rowan/hellaswag

Given an activity/situation description, the model must choose the most
plausible continuation from 4 options.  Adapted from nanochat's arc.py.
"""

from datasets import load_dataset

from tasks.base import Task, render_mc

LETTERS = ("A", "B", "C", "D")


class HellaSwag(Task):
    """HellaSwag commonsense completion task."""

    def __init__(self, split: str, **kwargs) -> None:
        """Load the HellaSwag dataset.

        Args:
            split: Dataset split; one of 'train' or 'validation'.
            **kwargs: Forwarded to Task.__init__ (start, stop, step).
        """
        super().__init__(**kwargs)
        assert split in ("train", "validation"), f"split must be train|validation, got {split}"
        self.ds = load_dataset("Rowan/hellaswag", split=split).shuffle(seed=42)

    @property
    def eval_type(self) -> str:
        """Return 'categorical' as HellaSwag requires selecting a letter answer."""
        return "categorical"

    def num_examples(self) -> int:
        """Return the total number of examples in the loaded split.

        Returns:
            Length of the dataset.
        """
        return len(self.ds)

    def get_example(self, index: int) -> dict:
        """Return a conversation dict for a HellaSwag problem.

        Args:
            index: Physical index into the dataset.

        Returns:
            A conversation dict with 'messages' and a 'letters' key listing
            the valid answer letters.
        """
        row = self.ds[index]
        # HellaSwag: activity_label + ctx is the question; endings are the choices
        activity = row.get("activity_label", "")
        ctx = row["ctx"]
        question = f"[{activity}] {ctx}".strip() if activity else ctx
        endings = row["endings"]  # list of 4 continuation strings
        label = int(row["label"])  # 0-3

        # Build letter choices
        user_msg = render_mc(question, LETTERS[: len(endings)], endings)
        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": LETTERS[label]},
            ],
            "letters": list(LETTERS[: len(endings)]),
        }

    def evaluate(self, problem: dict, completion: str) -> bool:
        """Check whether the completion matches the correct answer letter.

        Args:
            problem: The conversation dict as returned by get_example.
            completion: The model's generated answer string.

        Returns:
            True if completion (stripped, uppercased) equals the expected letter.
        """
        expected = problem["messages"][-1]["content"]
        return completion.strip().upper() == expected
