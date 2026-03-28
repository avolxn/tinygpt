"""
MMLU evaluation task.
https://huggingface.co/datasets/cais/mmlu
"""

from datasets import load_dataset

from tasks.base import Task, render_mc

LETTERS = ("A", "B", "C", "D")


class MMLU(Task):
    """Massive Multitask Language Understanding benchmark."""

    def __init__(self, subset: str, split: str, **kwargs) -> None:
        """Load the MMLU dataset.

        Args:
            subset: Dataset subset; currently only 'all' is supported.
            split: Dataset split; one of 'auxiliary_train', 'validation', 'dev', or 'test'.
            **kwargs: Forwarded to Task.__init__ (start, stop, step).
        """
        super().__init__(**kwargs)
        assert subset in ("all",), f"subset must be 'all', got {subset}"
        assert split in ("auxiliary_train", "validation", "dev", "test"), (
            f"split must be auxiliary_train|validation|dev|test, got {split}"
        )
        self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self) -> str:
        """Return 'categorical' as MMLU requires selecting a letter answer."""
        return "categorical"

    def num_examples(self) -> int:
        """Return the total number of examples in the loaded split.

        Returns:
            Length of the dataset.
        """
        return len(self.ds)

    def get_example(self, index: int) -> dict:
        """Return a conversation dict for an MMLU problem.

        Args:
            index: Physical index into the dataset.

        Returns:
            A conversation dict with 'messages' and a 'letters' key listing
            the valid answer letters.
        """
        row = self.ds[index]
        question = row["question"]
        choices = row["choices"]
        answer_idx = row["answer"]
        assert len(choices) == 4, "MMLU should have 4 choices"
        user_msg = render_mc(question, LETTERS, choices)
        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": LETTERS[answer_idx]},
            ],
            "letters": list(LETTERS),
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
