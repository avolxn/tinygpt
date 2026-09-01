"""Download a Hugging Face teacher and save the distillation format."""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinygpt.metrics import compute_token_bytes
from tinygpt.tokenizer import SPECIAL_TOKENS, HuggingFaceTokenizer

parser = argparse.ArgumentParser(description="Prepare a Hugging Face teacher for distillation")
parser.add_argument("--model", required=True, help="Hugging Face model ID")
parser.add_argument("--out-dir", required=True, help="Output directory for the converted teacher")
parser.add_argument("--revision", default=None, help="Optional Hugging Face revision")
parser.add_argument("--trust-remote-code", action="store_true")
args = parser.parse_args()

print(f"Downloading teacher from {args.model}")
tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
    args.model,
    revision=args.revision,
    trust_remote_code=args.trust_remote_code,
    use_fast=True,
)
if not tokenizer.is_fast:
    raise RuntimeError("Teacher tokenizer must be a fast Hugging Face tokenizer")

tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    revision=args.revision,
    trust_remote_code=args.trust_remote_code,
    torch_dtype="auto",
)
if model.get_input_embeddings().num_embeddings != len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))

os.makedirs(args.out_dir, exist_ok=True)
model.save_pretrained(args.out_dir, safe_serialization=True)
tokenizer.save_pretrained(args.out_dir)

tokenizer_path = os.path.join(args.out_dir, "tokenizer.json")
if not os.path.isfile(tokenizer_path):
    raise RuntimeError("Fast teacher tokenizer did not produce tokenizer.json")

wrapped_tokenizer = HuggingFaceTokenizer.from_directory(args.out_dir)
torch.save(compute_token_bytes(wrapped_tokenizer), os.path.join(args.out_dir, "token_bytes.pt"))

print(f"Saved teacher model and tokenizer to {args.out_dir}")
print(f"Teacher vocabulary size: {len(tokenizer):,}")
