"""
Tests for HuggingFaceTokenizer: encode/decode round-trip, special tokens,
conversation rendering, loss mask.
"""

import os
import tempfile

import pytest

from tinygpt.tokenizer import HuggingFaceTokenizer


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def tokenizer() -> HuggingFaceTokenizer:
    texts = [
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a great programming language.",
        "Numbers: 1, 2, 3, 42, 100.",
        "Special: @#$%^&*()",
    ]
    return HuggingFaceTokenizer.train_from_iterator(iter(texts * 10), vocab_size=512)


def test_encode_decode_roundtrip(tokenizer: HuggingFaceTokenizer) -> None:
    text = "Hello world!"
    ids = tokenizer.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tokenizer.decode(ids) == text


def test_encode_batch(tokenizer: HuggingFaceTokenizer) -> None:
    texts = ["Hello!", "World?"]
    ids = tokenizer.encode(texts)
    assert isinstance(ids, list)
    assert len(ids) == 2
    for i, text in enumerate(texts):
        assert tokenizer.decode(ids[i]) == text


def test_encode_with_prepend(tokenizer: HuggingFaceTokenizer) -> None:
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode("hello", prepend=bos)
    assert ids[0] == bos


def test_special_tokens_present(tokenizer: HuggingFaceTokenizer) -> None:
    specials = tokenizer.get_special_tokens()
    assert "<|bos|>" in specials
    assert "<|user_start|>" in specials
    assert "<|assistant_start|>" in specials


def test_encode_special(tokenizer: HuggingFaceTokenizer) -> None:
    bos_id = tokenizer.encode_special("<|bos|>")
    assert isinstance(bos_id, int)
    assert bos_id >= 0


def test_get_bos_token_id(tokenizer: HuggingFaceTokenizer) -> None:
    bos = tokenizer.get_bos_token_id()
    assert isinstance(bos, int)
    assert bos >= 0


def test_vocab_size(tokenizer: HuggingFaceTokenizer) -> None:
    # Small corpus may not generate enough merges to fill the full vocab_size budget
    assert tokenizer.get_vocab_size() <= 512


def test_save_load_roundtrip(tokenizer: HuggingFaceTokenizer) -> None:
    with tempfile.TemporaryDirectory() as d:
        tokenizer.save(d)
        assert os.path.exists(os.path.join(d, "tokenizer.json"))
        loaded = HuggingFaceTokenizer.from_directory(d)
        text = "Hello world!"
        assert loaded.encode(text) == tokenizer.encode(text)


def test_load_tokenizer_from_directory_json(tokenizer: HuggingFaceTokenizer) -> None:
    with tempfile.TemporaryDirectory() as d:
        tokenizer.save(d)
        loaded = HuggingFaceTokenizer.from_directory(d)
        text = "Hello world!"
        assert loaded.encode(text) == tokenizer.encode(text)


def test_render_conversation_basic(tokenizer: HuggingFaceTokenizer) -> None:
    conv = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    ids, mask = tokenizer.render_conversation(conv)
    assert len(ids) == len(mask)
    assert ids[0] == tokenizer.get_bos_token_id(), "First token should be BOS"
    # There should be some supervised (mask=1) tokens
    assert any(m == 1 for m in mask), "At least some tokens should be supervised"
    # Non-assistant tokens should have mask=0
    assert mask[0] == 0, "BOS should not be supervised"


def test_render_conversation_mask_assistant_only(tokenizer: HuggingFaceTokenizer) -> None:
    """The user message tokens must all have mask=0."""
    conv = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    }
    ids, mask = tokenizer.render_conversation(conv)
    # Decode and find positions of user_start / user_end
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    if user_start in ids and user_end in ids:
        i_start = ids.index(user_start)
        i_end = ids.index(user_end)
        for i in range(i_start, i_end + 1):
            assert mask[i] == 0, f"User token at position {i} should have mask=0"


def test_render_conversation_max_tokens(tokenizer: HuggingFaceTokenizer) -> None:
    """Output is truncated to max_tokens."""
    conv = {
        "messages": [
            {"role": "user", "content": "A" * 500},
            {"role": "assistant", "content": "B" * 500},
        ]
    }
    ids, mask = tokenizer.render_conversation(conv, max_tokens=50)
    assert len(ids) <= 50
    assert len(mask) <= 50
