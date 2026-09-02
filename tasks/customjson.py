"""
CustomJSON task for loading conversations from a local JSONL file.

Each line must be a JSON array of message objects with 'role' and 'content'
fields, e.g.:
    [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
"""

from __future__ import annotations

import json
import os
from typing import Any

from tasks.base import Task


class CustomJSON(Task):
    """Load conversations from a JSONL file for distillation.

    Args:
        filepath: Path to the .jsonl file. If the file does not exist, an
            empty task is created with a warning printed to stdout.
        **kwargs: Forwarded to Task.__init__ (start, stop, step).
    """

    def __init__(self, filepath: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations: list[list[dict[str, str]]] = []

        if not os.path.exists(filepath):
            print("-" * 80)
            print(f"Warning: {filepath} does not exist — CustomJSON task will be empty.")
            print("Download identity conversations with:")
            print(
                f"  curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
            )
            print("-" * 80)
        else:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    messages = json.loads(line)
                    if not isinstance(messages, list):
                        raise ValueError(f"Expected list of messages, got {type(messages)}")
                    if len(messages) < 2:
                        raise ValueError(f"Conversation must have >= 2 messages, got {len(messages)}")
                    for i, msg in enumerate(messages):
                        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                            raise ValueError(f"Message {i} missing role or content")
                        expected = "user" if i % 2 == 0 else "assistant"
                        if msg["role"] != expected:
                            raise ValueError(f"Message {i} role {msg['role']!r} != {expected!r}")
                        if not isinstance(msg["content"], str):
                            raise ValueError(f"Message {i} content must be a string")
                    self.conversations.append(messages)

    @property
    def eval_type(self) -> str:
        """Return 'generative' — CustomJSON is a training-only dataset."""
        return "generative"

    def num_examples(self) -> int:
        """Return the number of loaded conversations.

        Returns:
            Number of valid conversations parsed from the JSONL file.
        """
        return len(self.conversations)

    def get_example(self, index: int) -> dict[str, Any]:
        """Return the conversation at the given index.

        Args:
            index: Physical index into the conversation list.

        Returns:
            Dict with a 'messages' key.
        """
        return {"messages": self.conversations[index]}
