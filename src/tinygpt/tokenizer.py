"""
BPE Tokenizer based on HuggingFace `tokenizers` library.

Only HuggingFaceTokenizer is included (no rustbpe / tiktoken).
Saved as `tokenizer.json` (human-readable JSON, no pickle).

Special tokens and GPT-4 split pattern are the same as nanochat.
"""

import copy
import os
from collections.abc import Iterator
from typing import Any, cast

from tokenizers import Regex, decoders, pre_tokenizers
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|system_start|>",
    "<|system_end|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

# NOTE: uses \p{N}{1,2} instead of GPT-4's \p{N}{1,3} for smaller vocab sizes.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer with SFT rendering utilities."""

    def __init__(self, tokenizer: HFTokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path: str) -> "HuggingFaceTokenizer":
        return cls(HFTokenizer.from_pretrained(hf_path))

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "HuggingFaceTokenizer":
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        return cls(HFTokenizer.from_file(tokenizer_path))

    @classmethod
    def train_from_iterator(cls, text_iterator: Iterator[str], vocab_size: int) -> "HuggingFaceTokenizer":
        """Train a BPE tokenizer from a text iterator."""
        tokenizer = HFTokenizer(BPE(byte_fallback=True, unk_token=None, fuse_unk=False))
        tokenizer.normalizer = None
        gpt4_regex = Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(pattern=gpt4_regex, behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = None
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def get_special_tokens(self) -> list[str]:
        return [w.content for w in self.tokenizer.get_added_tokens_decoder().values()]

    def id_to_token(self, id: int) -> str:
        return str(self.tokenizer.id_to_token(id))

    def encode_special(self, text: str) -> int | None:
        return cast("int | None", self.tokenizer.token_to_id(text))

    def get_bos_token_id(self) -> int:
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
        num_threads: int | None = None,  # ignored, for API compatibility
    ) -> list[int]:
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

    def encode(
        self,
        text: str | list[str],
        prepend: str | int | None = None,
        append: str | int | None = None,
        num_threads: int | None = None,
    ) -> list[int] | list[list[int]]:
        if isinstance(text, str):
            return self._encode_one(text, prepend=prepend, append=append)
        elif isinstance(text, list):
            return [self._encode_one(t, prepend=prepend, append=append) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args: Any, **kwargs: Any) -> list[int] | list[list[int]]:
        return self.encode(*args, **kwargs)

    def decode(self, ids: list[int]) -> str:
        return str(self.tokenizer.decode(ids, skip_special_tokens=False))

    def save(self, tokenizer_dir: str) -> None:
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    def render_conversation(
        self,
        conversation: dict[str, Any],
        max_tokens: int = 2048,
    ) -> tuple[list[int], list[int]]:
        """
        Render a Chat conversation dict into (ids, mask).

        mask = 1 for assistant tokens (supervised), 0 for everything else.

        Conversation format:
            {"messages": [{"role": "system"|"user"|"assistant", "content": str|list}, ...]}

        System messages are wrapped in <|system_start|>...<|system_end|> special tokens.
        """
        ids: list[int] = []
        mask: list[int] = []

        def add_tokens(token_ids: list[int] | int | None, mask_val: int) -> None:
            if token_ids is None:
                raise RuntimeError("Missing special token in tokenizer")
            token_list = [token_ids] if isinstance(token_ids, int) else token_ids
            ids.extend(token_list)
            mask.extend([mask_val] * len(token_list))

        messages = conversation["messages"]

        bos = self.get_bos_token_id()
        system_start = self.encode_special("<|system_start|>")
        system_end = self.encode_special("<|system_end|>")
        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        python_start = self.encode_special("<|python_start|>")
        python_end = self.encode_special("<|python_end|>")
        output_start = self.encode_special("<|output_start|>")
        output_end = self.encode_special("<|output_end|>")

        add_tokens(bos, 0)

        # Strip optional leading system message
        if messages and messages[0]["role"] == "system":
            add_tokens(system_start, 0)
            add_tokens(self._encode_one(messages[0]["content"]), 0)
            add_tokens(system_end, 0)
            messages = messages[1:]

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
        """Render conversation for RL / completion (no last assistant message)."""
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

    def visualize_tokenization(
        self,
        ids: list[int],
        mask: list[int],
        with_token_id: bool = False,
    ) -> str:
        """Colorize tokenization for debugging."""
        RED = "\033[91m"
        GREEN = "\033[92m"
        GRAY = "\033[90m"
        RESET = "\033[0m"
        parts = []
        for token_id, mask_val in zip(ids, mask, strict=True):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            parts.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                parts.append(f"{GRAY}({token_id}){RESET}")
        return "|".join(parts)
