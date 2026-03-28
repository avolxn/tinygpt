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
        super().__init__(**kwargs)
        assert subset in ("all",), f"subset must be 'all', got {subset}"
        assert split in ("auxiliary_train", "validation", "dev", "test"), (
            f"split must be auxiliary_train|validation|dev|test, got {split}"
        )
        self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self) -> str:
        return "categorical"

    def num_examples(self) -> int:
        return len(self.ds)

    def get_example(self, index: int) -> dict:
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
        expected = problem["messages"][-1]["content"]
        return completion.strip().upper() == expected
