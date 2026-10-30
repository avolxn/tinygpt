"""BPE tokenizers used by tinygpt.

On-disk runtime format:
- `tokenizer.json`: HuggingFace tokenizers BPE

Teacher preparation verifies the required special tokens without changing token IDs.
"""

from typing import Any, cast, overload

from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

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


CHAT_TEMPLATE = """
{{- bos_token -}}
{%- for message in messages -%}
  {%- set offset = 1 if messages[0]['role'] == 'system' else 0 -%}
  {%- if loop.index0 < offset -%}{%- continue -%}{%- endif -%}
  {%- if message['role'] != ('user' if (loop.index0 - offset) % 2 == 0 else 'assistant') -%}
    {{- raise_exception('Messages must alternate user and assistant roles') -}}
  {%- endif -%}
  {%- if message['role'] == 'user' -%}
    {{- '<|user_start|>' -}}
    {%- if offset and loop.index0 == 1 -%}{{- messages[0]['content'] + '\n\n' -}}{%- endif -%}
    {{- message['content'] + '<|user_end|>' -}}
  {%- else -%}
    {{- '<|assistant_start|>' -}}
    {%- if message['content'] is string -%}
      {% generation %}{{- message['content'] -}}{% endgeneration %}
    {%- else -%}
      {%- for part in message['content'] -%}
        {%- if part['type'] == 'text' -%}
          {% generation %}{{- part['text'] -}}{% endgeneration %}
        {%- elif part['type'] == 'python' -%}
          {% generation %}{{- '<|python_start|>' + part['text'] + '<|python_end|>' -}}{% endgeneration %}
        {%- elif part['type'] == 'python_output' -%}
          {{- '<|output_start|>' + part['text'] + '<|output_end|>' -}}
        {%- else -%}
          {{- raise_exception('Unknown assistant content type') -}}
        {%- endif -%}
      {%- endfor -%}
    {%- endif -%}
    {% generation %}{{- '<|assistant_end|>' -}}{% endgeneration %}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|assistant_start|>' -}}{%- endif -%}
"""


class HuggingFaceTokenizer:
    """Light wrapper around Hugging Face Tokenizer with conversation utilities."""

    def __init__(self, tokenizer: HFTokenizer | PreTrainedTokenizerFast) -> None:
        if isinstance(tokenizer, HFTokenizer):
            if tokenizer.token_to_id("<|bos|>") is None:
                raise ValueError("Teacher tokenizer must contain <|bos|>")
            tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
                tokenizer_object=tokenizer,
                bos_token="<|bos|>",
                additional_special_tokens=SPECIAL_TOKENS,
                chat_template=CHAT_TEMPLATE,
            )
        self.tokenizer = tokenizer

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "HuggingFaceTokenizer":
        """Load a tokenizer from a local directory containing tokenizer.json.

        Args:
            tokenizer_dir: Path to the directory containing tokenizer.json.

        Returns:
            A HuggingFaceTokenizer loaded from the directory.
        """
        return cls(AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True))

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
            text: Special token string, e.g. '<|bos|>'.

        Returns:
            Integer token id, or None if the token is not in the vocabulary.
        """
        return cast("int | None", self.tokenizer.convert_tokens_to_ids(text))

    def get_bos_token_id(self) -> int:
        """Return the teacher's nanochat BOS token ID.

        Returns:
            Integer BOS token id.

        Raises:
            RuntimeError: If the required <|bos|> token is missing.
        """
        bos = self.encode_special("<|bos|>")
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

    def render_conversation(
        self,
        conversation: dict[str, Any],
        max_tokens: int = 2048,
    ) -> tuple[list[int], list[int]]:
        """Render tokens and assistant supervision with Transformers chat templates."""
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        encoded: Any = self.tokenizer.apply_chat_template(
            conversation["messages"],
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=max_tokens,
        )
        return list(encoded["input_ids"]), list(encoded["assistant_masks"])

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
        messages = conversation["messages"]
        if not messages:
            raise ValueError("Conversation must contain at least one assistant message")
        if messages[-1]["role"] != "assistant":
            raise ValueError("Last message must be from assistant")
        return list(self.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True))
