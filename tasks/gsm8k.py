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
    """Extract the numerical answer after the #### marker."""
    match = _GSM_RE.search(text)
    if match:
        return match.group(1).strip().replace(",", "")
    return None


class GSM8K(Task):
    """Grade School Math 8K dataset."""

    def __init__(self, subset: str, split: str, **kwargs) -> None:
        super().__init__(**kwargs)
        assert subset in ("main", "socratic"), f"subset must be main|socratic, got {subset}"
        assert split in ("train", "test"), f"split must be train|test, got {split}"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self) -> str:
        return "generative"

    def num_examples(self) -> int:
        return len(self.ds)

    def get_example(self, index: int) -> dict:
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
        assert isinstance(completion, str)
        assistant_msg = problem["messages"][-1]
        assert isinstance(assistant_msg["content"], list)
        last_text = assistant_msg["content"][-1]["text"]
        ref = extract_answer(last_text)
        pred = extract_answer(completion)
        return pred is not None and pred == ref
