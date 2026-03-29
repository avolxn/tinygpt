"""
SmolTalk conversational dataset by HuggingFace.
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk

Train split: ~460 K rows. Test split: ~24 K rows.
Training-only; no evaluate() implementation.
"""

from typing import Any

from datasets import load_dataset

from tasks.base import Task


class SmolTalk(Task):
    """Multi-turn conversational data for general SFT.

    Args:
        split: Either 'train' or 'test'.
        **kwargs: Forwarded to Task.__init__ (start, stop, step).
    """

    def __init__(self, split: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert split in ("train", "test"), "split must be train|test"
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)

    @property
    def eval_type(self) -> str:
        """Return 'generative' — SmolTalk is a generative training dataset."""
        return "generative"

    def num_examples(self) -> int:
        """Return the number of examples in the loaded split.

        Returns:
            Dataset length before any slicing.
        """
        return len(self.ds)

    def get_example(self, index: int) -> dict[str, Any]:
        """Return a single SmolTalk conversation dict.

        Args:
            index: Physical index into the dataset.

        Returns:
            Dict with a 'messages' key containing the full conversation.
        """
        row = self.ds[index]
        messages: list[dict[str, str]] = row["messages"]
        first = messages[0]
        rest = messages[1:] if first["role"] == "system" else messages
        assert len(rest) >= 2, "SmolTalk messages must have at least 2 non-system messages"
        for i, msg in enumerate(rest):
            expected = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected, f"Message {i} role {msg['role']!r} != {expected!r}"
            assert isinstance(msg["content"], str), f"Message {i} content must be a string"
        return {"messages": messages}
