"""
GSM8K (Grade School Math 8K) task.
https://huggingface.co/datasets/openai/gsm8k

The dataset uses << >> tool call markers which we map to python parts.
"""

import re

from datasets import load_dataset

from tasks.base import Task

_GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(text: str) -> str | None:
    """Extract the numerical answer after the #### marker.

    Args:
        text: Answer text possibly containing a '#### <number>' suffix.

    Returns:
        The extracted number string with commas removed, or None if not found.
    """
    match = _GSM_RE.search(text)
    if match:
        return match.group(1).strip().replace(",", "")
    return None


class GSM8K(Task):
    """Grade School Math 8K dataset."""

    def __init__(self, subset: str, split: str, **kwargs) -> None:
        """Load the GSM8K dataset.

        Args:
            subset: Dataset subset; one of 'main' or 'socratic'.
            split: Dataset split; one of 'train' or 'test'.
            **kwargs: Forwarded to Task.__init__ (start, stop, step).
        """
        super().__init__(**kwargs)
        assert subset in ("main", "socratic"), f"subset must be main|socratic, got {subset}"
        assert split in ("train", "test"), f"split must be train|test, got {split}"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self) -> str:
        """Return 'generative' as GSM8K requires free-form answer generation."""
        return "generative"

    def num_examples(self) -> int:
        """Return the total number of examples in the loaded split.

        Returns:
            Length of the dataset.
        """
        return len(self.ds)

    def get_example(self, index: int) -> dict:
        """Return a conversation dict for a GSM8K problem.

        Args:
            index: Physical index into the dataset.

        Returns:
            A conversation dict with 'messages' containing user question and
            assistant answer with embedded python tool-call parts.
        """
        row = self.ds[index]
        question = row["question"]
        answer = row["answer"]

        # Parse tool calls encoded as <<expr=result>>
        parts = []
        for chunk in re.split(r"(<<[^>]+>>)", answer):
            if chunk.startswith("<<") and chunk.endswith(">>"):
                inner = chunk[2:-2]
                expr, result = inner.rsplit("=", 1) if "=" in inner else (inner, "")
                parts.append({"type": "python", "text": expr})
                parts.append({"type": "python_output", "text": result})
            else:
                parts.append({"type": "text", "text": chunk})

        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": parts},
            ]
        }

    def evaluate(self, problem: dict, completion: str) -> bool:
        """Check whether the completion contains the correct numerical answer.

        Args:
            problem: The conversation dict as returned by get_example.
            completion: The model's generated answer string.

        Returns:
            True if the predicted answer matches the reference after the #### marker.
        """
        assert isinstance(completion, str)
        assistant_msg = problem["messages"][-1]
        assert isinstance(assistant_msg["content"], list)
        last_text = assistant_msg["content"][-1]["text"]
        ref = extract_answer(last_text)
        pred = extract_answer(completion)
        return pred is not None and pred == ref
