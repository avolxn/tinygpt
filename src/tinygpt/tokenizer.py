"""Teacher tokenization and chat rendering through Hugging Face."""

import re
from typing import Any, cast, overload

from transformers import AutoTokenizer, PreTrainedTokenizerFast


class HuggingFaceTokenizer:
    """Use the selected teacher's vocabulary and native chat template."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        if not tokenizer.is_fast or not tokenizer.chat_template:
            raise ValueError("A fast tokenizer with a chat template is required")
        self.tokenizer = tokenizer

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "HuggingFaceTokenizer":
        """Load a tokenizer from a local directory containing tokenizer.json.

        Args:
            tokenizer_dir: Path to the directory containing tokenizer.json.

        Returns:
            A HuggingFaceTokenizer loaded from the directory.
        """
        return cls(
            cast(
                PreTrainedTokenizerFast,
                AutoTokenizer.from_pretrained(
                    tokenizer_dir,
                    local_files_only=True,
                    use_fast=True,
                ),
            )
        )

    def get_vocab_size(self) -> int:
        """Return the total vocabulary size including special tokens.

        Returns:
            Integer vocabulary size.
        """
        return len(self.tokenizer)

    def get_vocab(self) -> dict[str, int]:
        """Return the complete token-to-id mapping."""
        return {str(token): int(token_id) for token, token_id in self.tokenizer.get_vocab().items()}

    def get_special_tokens(self) -> list[str]:
        """Return the list of added special token strings.

        Returns:
            List of special token string representations.
        """
        return [w.content for w in self.tokenizer.added_tokens_decoder.values()]

    def encode_special(self, text: str) -> int | None:
        """Return the token id for a special token string, or None if not found.

        Args:
            text: Special token string, as defined by the teacher.

        Returns:
            Integer token id, or None if the token is not in the vocabulary.
        """
        return self.get_vocab().get(text)

    def get_bos_token_id(self) -> int:
        """Return BOS, or EOS as the document separator for tokenizers without BOS.

        Raises:
            ValueError: If the tokenizer defines neither token.
        """
        token_id = self.tokenizer.bos_token_id
        if token_id is None:
            token_id = self.tokenizer.eos_token_id
        if token_id is None:
            raise ValueError("Tokenizer must define BOS or EOS for document boundaries")
        return int(token_id)

    def get_eos_token_id(self) -> int:
        """Return the native end-of-sequence token.

        Raises:
            ValueError: If the tokenizer does not define EOS.
        """
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define EOS")
        return int(self.tokenizer.eos_token_id)

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
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False))
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
        self.tokenizer.save_pretrained(tokenizer_dir)

    @staticmethod
    def _messages(conversation: dict[str, Any]) -> list[dict[str, Any]]:
        """Render dataset calculator annotations as ordinary text without new tokens."""
        messages = []
        for message in conversation["messages"]:
            content = message["content"]
            if isinstance(content, list):
                parts = []
                for part in content:
                    kind, text = part["type"], part["text"]
                    if kind == "python":
                        text = "<calculator>" + text + "</calculator>"
                    elif kind == "python_output":
                        text = "<result>" + text + "</result>"
                    elif kind != "text":
                        raise ValueError(f"Unknown content type: {kind}")
                    parts.append(text)
                content = "".join(parts)
            messages.append({**message, "content": content})
        return messages

    def render_conversation(
        self,
        conversation: dict[str, Any],
        max_tokens: int = 2048,
    ) -> tuple[list[int], list[int]]:
        """Apply the native chat template and supervise assistant text and turn endings.

        Args:
            conversation: Messages with text or structured calculator content.
            max_tokens: Maximum encoded sequence length.

        Returns:
            Token IDs and an assistant-only loss mask, excluding calculator results.

        Raises:
            ValueError: If limits or the template's content rendering are unsupported.
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        messages = self._messages(conversation)
        rendered = cast(str, self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        spans: list[tuple[int, int]] = []
        excluded: list[tuple[int, int]] = []
        cursor = 0
        for i, message in enumerate(messages):
            if message["role"] != "assistant" or not message["content"].strip():
                continue
            prefix = cast(str, self.tokenizer.apply_chat_template(messages[: i + 1], tokenize=False))
            content = message["content"].strip()
            start = prefix.rfind(content)
            if start < 0:
                raise ValueError("Chat template must preserve assistant content")
            suffix = prefix[start + len(content) :]
            start = rendered.find(content + suffix, cursor)
            if start < 0:
                raise ValueError("Chat template must preserve assistant text and turn endings")
            end = start + len(content) + len(suffix)
            spans.append((start, end))
            cursor = end
            excluded.extend(
                (start + m.start(), start + m.end()) for m in re.finditer(r"<result>.*?</result>", content, re.S)
            )
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_tokens,
        )
        mask = [
            int(
                end > start
                and any(start < b and end > a for a, b in spans)
                and not any(start < b and end > a for a, b in excluded)
            )
            for start, end in encoded["offset_mapping"]
        ]
        return list(encoded["input_ids"]), mask

    def render_for_completion(self, conversation: dict[str, Any]) -> list[int]:
        """Apply the teacher's generation prompt to all turns before the final answer.

        Args:
            conversation: Conversation whose final message is an assistant answer.

        Returns:
            Token IDs ready for assistant generation.

        Raises:
            ValueError: If the final assistant message is missing.
        """
        messages = self._messages(conversation)
        if not messages:
            raise ValueError("Conversation must contain at least one assistant message")
        if messages[-1]["role"] != "assistant":
            raise ValueError("Last message must be from assistant")
        return list(
            self.tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )
        )
