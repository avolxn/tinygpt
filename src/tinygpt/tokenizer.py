"""
BPE tokenizers used by tinygpt.

On-disk runtime format:
- `tokenizer.json`: HuggingFace tokenizers BPE

Teacher preparation adds the special tokens required by tinygpt.
"""

import copy
import os
from typing import Any, cast, overload

from tokenizers import Tokenizer as HFTokenizer

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

class HuggingFaceTokenizer:
    """Light wrapper around Hugging Face Tokenizer with conversation utilities."""

    def __init__(self, tokenizer: HFTokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "HuggingFaceTokenizer":
        """Load a tokenizer from a local directory containing tokenizer.json.

        Args:
            tokenizer_dir: Path to the directory containing tokenizer.json.

        Returns:
            A HuggingFaceTokenizer loaded from the directory.
        """
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        return cls(HFTokenizer.from_file(tokenizer_path))

    def get_vocab_size(self) -> int:
        """Return the total vocabulary size including special tokens.

        Returns:
            Integer vocabulary size.
        """
        return int(self.tokenizer.get_vocab_size())

    def get_special_tokens(self) -> list[str]:
        """Return the list of added special token strings.

        Returns:
            List of special token string representations.
        """
        return [w.content for w in self.tokenizer.get_added_tokens_decoder().values()]

    def id_to_token(self, token_id: int) -> str:
        """Convert a token id to its string representation.

        Args:
            token_id: Integer token id.

        Returns:
            String representation of the token.
        """
        return str(self.tokenizer.id_to_token(token_id))

    def encode_special(self, text: str) -> int | None:
        """Return the token id for a special token string, or None if not found.

        Args:
            text: Special token string, e.g. '<|bos|>'.

        Returns:
            Integer token id, or None if the token is not in the vocabulary.
        """
        return cast("int | None", self.tokenizer.token_to_id(text))

    def get_bos_token_id(self) -> int:
        """Return the BOS token id, trying <|bos|> then <|endoftext|>.

        Returns:
            Integer BOS token id.

        Raises:
            RuntimeError: If neither <|bos|> nor <|endoftext|> is found.
        """
        bos = self.encode_special("<|bos|>")
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        if bos is None:
            raise RuntimeError("Failed to find BOS token in tokenizer")
        return bos

    def _encode_one(
        self,
        text: str,
        prepend: str | int | None = None,
        append: str | int | None = None,
    ) -> list[int]:
        """Encode a single string, optionally prepending/appending a special token.

        Args:
            text: The string to encode.
            prepend: Optional special token (string name or id) to prepend.
            append: Optional special token (string name or id) to append.

        Returns:
            List of integer token ids.

        Raises:
            ValueError: If prepend or append names an unknown special token.
        """
        ids: list[int] = []
        if prepend is not None:
            if isinstance(prepend, int):
                ids.append(prepend)
            else:
                tok = self.encode_special(prepend)
                if tok is None:
                    raise ValueError(f"Unknown special token: {prepend!r}")
                ids.append(tok)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            if isinstance(append, int):
                ids.append(append)
            else:
                tok = self.encode_special(append)
                if tok is None:
                    raise ValueError(f"Unknown special token: {append!r}")
                ids.append(tok)
        return ids

    @overload
    def encode(
        self,
        text: str,
        prepend: str | int | None = None,
        append: str | int | None = None,
    ) -> list[int]: ...

    @overload
    def encode(
        self,
        text: list[str],
        prepend: str | int | None = None,
        append: str | int | None = None,
    ) -> list[list[int]]: ...

    def encode(
        self,
        text: str | list[str],
        prepend: str | int | None = None,
        append: str | int | None = None,
    ) -> list[int] | list[list[int]]:
        """Encode a string or list of strings into token id(s).

        Args:
            text: A single string or a list of strings to encode.
            prepend: Optional special token (string name or id) to prepend to each encoding.
            append: Optional special token (string name or id) to append to each encoding.

        Returns:
            A flat list of ints for a single string, or a list of lists for batch input.

        Raises:
            ValueError: If text is not a str or list.
        """
        if isinstance(text, str):
            return self._encode_one(text, prepend=prepend, append=append)
        elif isinstance(text, list):
            return [self._encode_one(t, prepend=prepend, append=append) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args: Any, **kwargs: Any) -> list[int] | list[list[int]]:
        """Delegate to encode().

        Args:
            *args: Positional arguments forwarded to encode().
            **kwargs: Keyword arguments forwarded to encode().

        Returns:
            A flat list of ints for a single string, or a list of lists for batch input.
        """
        return self.encode(*args, **kwargs)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to a string, preserving special tokens.

        Args:
            ids: List of integer token ids to decode.

        Returns:
            Decoded string with special tokens included.
        """
        return str(self.tokenizer.decode(ids, skip_special_tokens=False))

    def save(self, tokenizer_dir: str) -> None:
        """Save the tokenizer to a directory as tokenizer.json.

        Args:
            tokenizer_dir: Directory path where tokenizer.json will be written.
        """
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    def render_conversation(
        self,
        conversation: dict[str, Any],
        max_tokens: int = 2048,
    ) -> tuple[list[int], list[int]]:
        """Render a chat conversation dict into token ids and a supervision mask.

        Args:
            conversation: Dict with a 'messages' key containing a list of
                role/content dicts. Roles: 'user' or 'assistant'.
                Content is a string for user messages; a string or list
                of typed parts for assistant messages.
            max_tokens: Truncate output to at most this many tokens.

        Returns:
            A (ids, mask) tuple where ids is a flat list of token ids and mask
            is 1 for assistant tokens (supervised) and 0 for everything else.

        Raises:
            ValueError: If message roles are out of order, content types are
                unexpected, or part types are unknown.
            TypeError: If a user message has non-string content.
            RuntimeError: If a required special token is missing from the tokenizer.
        """
        ids: list[int] = []
        mask: list[int] = []

        def add_tokens(token_ids: list[int] | int | None, mask_val: int) -> None:
            """Append token ids and their supervision mask values to the output lists.

            Args:
                token_ids: Single token id (int), list of token ids, or None (error).
                mask_val: Supervision value (1 for supervised/assistant tokens, 0 otherwise).

            Raises:
                RuntimeError: If token_ids is None (missing special token).
            """
            if token_ids is None:
                raise RuntimeError("Missing special token in tokenizer")
            token_list = [token_ids] if isinstance(token_ids, int) else token_ids
            ids.extend(token_list)
            mask.extend([mask_val] * len(token_list))

        messages = conversation["messages"]

        bos = self.get_bos_token_id()
        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        python_start = self.encode_special("<|python_start|>")
        python_end = self.encode_special("<|python_end|>")
        output_start = self.encode_special("<|output_start|>")
        output_end = self.encode_special("<|output_end|>")

        add_tokens(bos, 0)

        for i, message in enumerate(messages):
            expected = "user" if i % 2 == 0 else "assistant"
            if message["role"] != expected:
                raise ValueError(f"Message {i}: expected {expected}, got {message['role']}")

            content = message["content"]
            if message["role"] == "user":
                if not isinstance(content, str):
                    raise TypeError("User messages must be strings")
                add_tokens(user_start, 0)
                add_tokens(self._encode_one(content), 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    add_tokens(self._encode_one(content), 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self._encode_one(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        return ids[:max_tokens], mask[:max_tokens]

    def render_for_completion(self, conversation: dict[str, Any]) -> list[int]:
        """Render a conversation for completion, omitting the last assistant turn.

        Strips the final assistant message and appends <|assistant_start|> so the
        model can generate the assistant response from scratch.

        Args:
            conversation: Conversation dict whose last message must be from the assistant.

        Returns:
            List of token ids ending with the <|assistant_start|> token.

        Raises:
            ValueError: If the last message is not from the assistant.
            RuntimeError: If <|assistant_start|> is missing from the tokenizer.
        """
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        if messages[-1]["role"] != "assistant":
            raise ValueError("Last message must be from assistant")
        messages.pop()
        ids, _ = self.render_conversation(conversation)
        assistant_start = self.encode_special("<|assistant_start|>")
        if assistant_start is None:
            raise RuntimeError("Missing assistant_start token in tokenizer")
        ids.append(assistant_start)
        return ids
