"""
Convert nanochat-style checkpoints and tokenizers into Hugging Face format.

Usage:
    python -m scripts.convert --input karpathy/nanochat-d34 --out-dir out/teacher_hf
    python -m scripts.convert --input path/to/legacy_dir --out-dir out/model_hf
    python -m scripts.convert --input path/to/tokenizer.pkl --out-dir out/tokenizer_hf --skip-model
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
from collections.abc import Iterable
from typing import Any

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import save_file as safe_save_file
from tiktoken._educational import bpe_encode
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_NAME

from tinygpt.tokenizer import HuggingFaceTokenizer

DEFAULT_PROBES = [
    "Hello world!",
    "The quick brown fox jumps over 13 lazy dogs.\n",
    "Math: 2 + 2 = 4, 17 + 5 = 22.",
    "Unicode: Привет, 你好, مرحبا.",
    "Whitespace:\n  indented line\n\nlast line.",
    "Punctuation: ()[]{}.,!?-_'\"",
]

TRAINER_STATE_NAME = "trainer_state.json"
_META_RE = re.compile(r"meta_(\d+)\.json$")


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


def _patch_legacy_config_keys(model_config_kwargs: dict[str, Any]) -> None:
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"


def _patch_legacy_weights(model_data: dict[str, torch.Tensor], n_layer: int) -> None:
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)


def _find_latest_step(snapshot_dir: str) -> int:
    meta_paths = glob.glob(os.path.join(snapshot_dir, "meta_*.json"))
    if not meta_paths:
        raise FileNotFoundError(f"No meta_*.json files found in {snapshot_dir}")
    steps = [int(match.group(1)) for path in meta_paths if (match := _META_RE.search(os.path.basename(path)))]
    if not steps:
        raise FileNotFoundError(f"No valid meta_*.json files found in {snapshot_dir}")
    return max(steps)


def _resolve_legacy_model_paths(source: str, step: int | None = None) -> tuple[str, str, int]:
    if os.path.isdir(source):
        resolved_step = _find_latest_step(source) if step is None else step
        meta_path = os.path.join(source, f"meta_{resolved_step}.json")
        model_path = os.path.join(source, f"model_{resolved_step}.pt")
        if not os.path.exists(meta_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find legacy model files for step {resolved_step} in {source}")
        return model_path, meta_path, resolved_step

    snapshot_dir = snapshot_download(repo_id=source, allow_patterns=["meta_*.json"])
    resolved_step = _find_latest_step(snapshot_dir) if step is None else step
    meta_path = hf_hub_download(repo_id=source, filename=f"meta_{resolved_step}.json")
    model_path = hf_hub_download(repo_id=source, filename=f"model_{resolved_step}.pt")
    return model_path, meta_path, resolved_step


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
    """Convert a nanochat/tiktoken `tokenizer.pkl` into `tokenizer.json`."""
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


def convert_legacy_model_to_hf(
    source: str,
    out_dir: str,
    *,
    step: int | None = None,
) -> str:
    """Convert a nanochat `model_*.pt` + `meta_*.json` checkpoint into HF model files."""
    model_path, meta_path, resolved_step = _resolve_legacy_model_paths(source, step=step)
    model_data = torch.load(model_path, map_location="cpu", weights_only=True)
    with open(meta_path, encoding="utf-8") as f:
        meta_data = json.load(f)

    config_dict = dict(meta_data["model_config"])
    _patch_legacy_config_keys(config_dict)
    _patch_legacy_weights(model_data, int(config_dict["n_layer"]))

    state_dict = {
        key.removeprefix("_orig_mod."): value.contiguous()
        for key, value in model_data.items()
    }

    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, CONFIG_NAME)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    weights_path = os.path.join(out_dir, SAFE_WEIGHTS_NAME)
    safe_save_file(state_dict, weights_path, metadata={"format": "pt"})

    trainer_state = {
        "global_step": int(meta_data.get("step", resolved_step)),
    }
    trainer_state_path = os.path.join(out_dir, TRAINER_STATE_NAME)
    with open(trainer_state_path, "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)

    return weights_path


def _save_token_bytes(tokenizer_dir: str) -> str:
    tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_bytes_list = []
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32)
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    torch.save(token_bytes, token_bytes_path)
    return token_bytes_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy nanochat artifacts into Hugging Face format")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Legacy source: local directory, Hub repo, or tokenizer.pkl file",
    )
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for Hugging Face artifacts")
    parser.add_argument("--step", type=int, default=None, help="Specific legacy checkpoint step to convert")
    parser.add_argument("--skip-model", action="store_true", help="Skip model conversion")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer conversion")
    parser.add_argument("--skip-verify", action="store_true", help="Skip encode-equivalence checks on probe strings")
    args = parser.parse_args()

    source_is_tokenizer_file = os.path.isfile(args.input) and args.input.endswith(".pkl")

    if not args.skip_model and not source_is_tokenizer_file:
        weights_path = convert_legacy_model_to_hf(args.input, args.out_dir, step=args.step)
        print(f"Saved converted model weights to {weights_path}")

    if not args.skip_tokenizer:
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
