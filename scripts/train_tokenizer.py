"""
Train a BPE tokenizer in GPT-4 style.

Streams text from an HF dataset (or a local .txt file) and trains a BPE
tokenizer using HuggingFace `tokenizers`.  The result is saved as
out/tokenizer/tokenizer.json.

Also writes out/tokenizer/token_bytes.pt: a 1-D int32 tensor mapping each
token id to its UTF-8 byte length (0 for special tokens) — used by the
bits-per-byte evaluator.

Usage:
    # Train on HF dataset (default: HuggingFaceFW/fineweb)
    python -m scripts.train_tokenizer

    # Train on local text file
    python -m scripts.train_tokenizer --txt data/shakespeare.txt

    # Smaller vocab for testing
    python -m scripts.train_tokenizer --vocab-size 8192 --max-chars 10000000
"""

import argparse
import os
import time

import torch

from tinygpt.tokenizer import HuggingFaceTokenizer

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument(
    "--dataset", type=str, default="HuggingFaceFW/fineweb", help="HF dataset identifier (ignored if --txt is given)"
)
parser.add_argument("--split", type=str, default="train", help="Dataset split to train on")
parser.add_argument("--text-field", type=str, default="text", help="Column name containing document text")
parser.add_argument("--txt", type=str, default="", help="Local .txt file to train on (one document per line)")
parser.add_argument("--max-chars", type=int, default=2_000_000_000, help="Maximum characters to train on (default: 2B)")
parser.add_argument("--doc-cap", type=int, default=10_000, help="Maximum characters per document (default: 10k)")
parser.add_argument("--vocab-size", type=int, default=32768, help="Vocabulary size (default: 32768 = 2^15)")
parser.add_argument(
    "--out-dir", type=str, default="out/tokenizer", help="Output directory for tokenizer.json and token_bytes.pt"
)
args = parser.parse_args()

print(f"vocab_size:  {args.vocab_size:,}")
print(f"max_chars:   {args.max_chars:,}")
print(f"doc_cap:     {args.doc_cap:,}")


# ---------------------------------------------------------------------------
# Text iterator
# ---------------------------------------------------------------------------


def text_iterator():
    nchars = 0
    if args.txt:
        print(f"Reading from local file: {args.txt}")
        with open(args.txt, encoding="utf-8") as f:
            for line in f:
                doc = line.strip()
                if not doc:
                    continue
                if len(doc) > args.doc_cap:
                    doc = doc[: args.doc_cap]
                nchars += len(doc)
                yield doc
                if nchars >= args.max_chars:
                    return
    else:
        from datasets import load_dataset  # noqa: PLC0415

        print(f"Streaming from HF dataset: {args.dataset} / {args.split}")
        ds = load_dataset(args.dataset, split=args.split, streaming=True, trust_remote_code=True)
        for row in ds:
            doc = row.get(args.text_field, row.get("content", ""))
            if not doc:
                continue
            if len(doc) > args.doc_cap:
                doc = doc[: args.doc_cap]
            nchars += len(doc)
            yield doc
            if nchars >= args.max_chars:
                return


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

print("Training tokenizer...")
t0 = time.time()
tokenizer = HuggingFaceTokenizer.train_from_iterator(text_iterator(), args.vocab_size)
t1 = time.time()
print(f"Training time: {t1 - t0:.2f}s")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

os.makedirs(args.out_dir, exist_ok=True)
tokenizer.save(args.out_dir)

# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

test_text = "Hello world! This is a test.\nNumbers: 123, 4567\nUnicode: 你好 🌍"
encoded = tokenizer.encode(test_text)
assert tokenizer.decode(encoded) == test_text, "Encode/decode round-trip failed!"
print(f"Sanity check passed: {len(encoded)} tokens for {len(test_text)} chars")

# ---------------------------------------------------------------------------
# Build and save token_bytes tensor
# ---------------------------------------------------------------------------

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
token_bytes_path = os.path.join(args.out_dir, "token_bytes.pt")
torch.save(token_bytes, token_bytes_path)
print(f"Saved token_bytes to {token_bytes_path}")
print(f"Tokenizer vocab size: {vocab_size:,}")
