"""Convert a legacy tiktoken tokenizer into Hugging Face tokenizer format."""

from __future__ import annotations

import argparse
import os
import pickle
from collections.abc import Iterable
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from tiktoken._educational import bpe_encode
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE

from tinygpt.metrics import compute_token_bytes
from tinygpt.tokenizer import HuggingFaceTokenizer

DEFAULT_PROBES = [
    "Hello world!",
    "The quick brown fox jumps over 13 lazy dogs.\n",
    "Math: 2 + 2 = 4, 17 + 5 = 22.",
    "Unicode: Привет, 你好, مرحبا.",
    "Whitespace:\n  indented line\n\nlast line.",
    "Punctuation: ()[]{}.,!?-_'\"",
]

def _bytes_to_unicode() -> dict[int, str]:
    """Return the GPT-2 byte-to-unicode map used by ByteLevel tokenization."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs], strict=True))


def _token_bytes_to_string(token_bytes: bytes, byte_map: dict[int, str]) -> str:
    return "".join(byte_map[b] for b in token_bytes)


def _recover_merges(mergeable_ranks: dict[bytes, int]) -> list[tuple[bytes, bytes]]:
    """Recover the ordered BPE merge list from a tiktoken mergeable-ranks table."""
    rank_to_bytes = {rank: token_bytes for token_bytes, rank in mergeable_ranks.items()}
    known_ranks: dict[bytes, int] = {}
    merges: list[tuple[bytes, bytes]] = []

    for rank in range(len(rank_to_bytes)):
        token_bytes = rank_to_bytes[rank]
        if len(token_bytes) == 1:
            known_ranks[token_bytes] = rank
            continue

        parts = bpe_encode(known_ranks, token_bytes, visualise=None)
        if len(parts) != 2:
            raise ValueError(
                f"Could not recover a binary merge for rank {rank} / token {token_bytes!r}; got {len(parts)} parts"
            )
        left = rank_to_bytes[parts[0]]
        right = rank_to_bytes[parts[1]]
        merges.append((left, right))
        known_ranks[token_bytes] = rank

    return merges


def _resolve_tokenizer_pickle_path(source: str) -> str:
    if os.path.isfile(source):
        return source
    if os.path.isdir(source):
        tokenizer_pkl = os.path.join(source, "tokenizer.pkl")
        if os.path.exists(tokenizer_pkl):
            return tokenizer_pkl
        raise FileNotFoundError(f"Could not find tokenizer.pkl in {source}")
    return hf_hub_download(repo_id=source, filename="tokenizer.pkl")


def convert_tiktoken_encoding_to_hf(
    encoding: Any,
    *,
    additional_special_tokens: dict[str, int] | None = None,
) -> Tokenizer:
    """Convert a tiktoken Encoding into an equivalent HuggingFace Tokenizer."""
    mergeable_ranks: dict[bytes, int] = encoding._mergeable_ranks
    special_tokens: dict[str, int] = dict(encoding._special_tokens)
    if additional_special_tokens:
        special_tokens.update(additional_special_tokens)

    byte_map = _bytes_to_unicode()
    merges_bytes = _recover_merges(mergeable_ranks)

    vocab = {
        _token_bytes_to_string(token_bytes, byte_map): rank
        for token_bytes, rank in sorted(mergeable_ranks.items(), key=lambda item: item[1])
    }
    vocab.update(special_tokens)
    merges = [
        (_token_bytes_to_string(left, byte_map), _token_bytes_to_string(right, byte_map))
        for left, right in merges_bytes
    ]

    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, byte_fallback=True, fuse_unk=False))
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(pattern=Regex(encoding._pat_str), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = None
    if special_tokens:
        tokenizer.add_special_tokens(list(special_tokens))
    return tokenizer


def convert_tokenizer_pickle_to_json(
    tokenizer_pkl_path: str,
    out_dir: str,
    *,
    probe_texts: Iterable[str] | None = None,
) -> str:
    """Convert a legacy tiktoken `tokenizer.pkl` into `tokenizer.json`."""
    with open(tokenizer_pkl_path, "rb") as f:
        encoding = pickle.load(f)
    hf_tokenizer = convert_tiktoken_encoding_to_hf(encoding)

    if probe_texts is not None:
        for text in probe_texts:
            source_ids = encoding.encode_ordinary(text)
            target_ids = hf_tokenizer.encode(text, add_special_tokens=False).ids
            if source_ids != target_ids:
                raise ValueError(f"Converted tokenizer does not match source encoding on a probe string: {text!r}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tokenizer.json")
    hf_tokenizer.save(out_path, pretty=True)
    return out_path


def _save_token_bytes(tokenizer_dir: str) -> str:
    tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)
    token_bytes = compute_token_bytes(tokenizer)
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    torch.save(token_bytes, token_bytes_path)
    return token_bytes_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a legacy tokenizer into Hugging Face format")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Local directory, Hub repo, or tokenizer.pkl file",
    )
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for Hugging Face artifacts")
    parser.add_argument("--skip-verify", action="store_true", help="Skip encode-equivalence checks on probe strings")
    args = parser.parse_args()

    tokenizer_pkl = _resolve_tokenizer_pickle_path(args.input)
    out_path = convert_tokenizer_pickle_to_json(
        tokenizer_pkl,
        args.out_dir,
        probe_texts=None if args.skip_verify else DEFAULT_PROBES,
    )
    print(f"Saved converted tokenizer to {out_path}")

    token_bytes_path = _save_token_bytes(args.out_dir)
    print(f"Saved token_bytes to {token_bytes_path}")


if __name__ == "__main__":
    main()
