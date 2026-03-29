"""
HumanEval coding benchmark.
https://huggingface.co/datasets/openai/openai_humaneval

164 Python programming problems evaluated by running the generated code
against unit tests in a sandboxed subprocess.
"""

from __future__ import annotations

import re
from typing import Any

from datasets import load_dataset

from tasks.base import Task


def _extract_imports(prompt: str) -> str:
    """Extract leading import statements from a code prompt.

    Args:
        prompt: The HumanEval problem prompt (function signature + docstring).

    Returns:
        Newline-joined import statements found at the top of the prompt.
    """
    imports: list[str] = []
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
        elif stripped and not stripped.startswith("#"):
            break
    return "\n".join(imports)


def _extract_program(completion: str) -> str:
    """Extract Python code from an LLM completion.

    Handles ```python ... ``` and ``` ... ``` markdown blocks; falls back to
    returning the whole completion if no code block is found.

    Args:
        completion: Raw model output string.

    Returns:
        Extracted code string.
    """
    matches = re.findall(r"```(?:python)?\s*\n(.*?)\n```", completion, re.DOTALL)
    return matches[0].strip() if matches else completion.strip()


class HumanEval(Task):
    """164 Python coding problems from the HumanEval benchmark.

    Evaluation runs generated code in a sandboxed subprocess and checks the
    provided unit tests.

    Args:
        **kwargs: Forwarded to Task.__init__ (start, stop, step).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)

    @property
    def eval_type(self) -> str:
        """Return 'generative' — evaluation requires code execution."""
        return "generative"

    def num_examples(self) -> int:
        """Return the number of HumanEval problems.

        Returns:
            164 (fixed test split size).
        """
        return len(self.ds)

    def get_example(self, index: int) -> dict[str, Any]:
        """Return a single HumanEval problem as a conversation dict.

        Args:
            index: Physical index into the dataset.

        Returns:
            Dict with 'messages', 'entry_point', and 'test' keys.
        """
        row = self.ds[index]
        prompt: str = row["prompt"]
        solution: str = row["canonical_solution"]
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"{prompt}\n{solution}"},
            ],
            "entry_point": row["entry_point"],
            "test": row["test"],
        }

    def evaluate(self, problem: dict[str, Any], completion: str) -> bool:
        """Execute the completion against the HumanEval unit tests.

        Args:
            problem: Conversation dict from get_example.
            completion: Model's generated code string.

        Returns:
            True if all unit tests pass.
        """
        from tinygpt.execution import execute_code  # noqa: PLC0415

        prompt = problem["messages"][0]["content"]
        imports = _extract_imports(prompt)
        code = _extract_program(completion)
        program = imports + "\n\n" + code + "\n\n" + problem["test"] + "\n" + f"check({problem['entry_point']})"
        result = execute_code(program)
        return result.success
