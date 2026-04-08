"""
Tests for converting legacy tiktoken tokenizers into tokenizer.json.
"""

import pickle
import tempfile

import tiktoken
from scripts.convert_tokenizer import convert_tiktoken_encoding_to_hf, convert_tokenizer_pickle_to_json
from tokenizers import Tokenizer as HFTokenizer


def make_toy_encoding() -> tiktoken.Encoding:
    mergeable_ranks = {
        b"a": 0,
        b"b": 1,
        b"c": 2,
        b"ab": 3,
        b"abc": 4,
        b" ": 5,
    }
    special_tokens = {"<|bos|>": 6, "<|assistant_start|>": 7}
    return tiktoken.Encoding(name="toy", pat_str=r"\s+|[^\s]+", mergeable_ranks=mergeable_ranks, special_tokens=special_tokens)


def test_convert_tiktoken_encoding_to_hf_keeps_regular_encoding() -> None:
    encoding = make_toy_encoding()
    hf_tokenizer = convert_tiktoken_encoding_to_hf(encoding)

    text = "abc ab"
    assert hf_tokenizer.encode(text, add_special_tokens=False).ids == encoding.encode(text)


def test_convert_tokenizer_pickle_to_json_roundtrip() -> None:
    encoding = make_toy_encoding()
    with tempfile.TemporaryDirectory() as tmp:
        src = f"{tmp}/tokenizer.pkl"
        with open(src, "wb") as f:
            pickle.dump(encoding, f)

        out_dir = f"{tmp}/out"
        out_path = convert_tokenizer_pickle_to_json(src, out_dir, probe_texts=["abc ab", "ab c"])
        loaded = HFTokenizer.from_file(out_path)

        assert loaded.encode("abc ab", add_special_tokens=False).ids == encoding.encode("abc ab")
