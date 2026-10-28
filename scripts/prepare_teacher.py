"""Download a Hugging Face teacher and save the distillation format."""

import argparse
import os

from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinygpt.tokenizer import CHAT_TEMPLATE, SPECIAL_TOKENS

NANOCHAT_MODEL = "karpathy/nanochat-d32"
NANOCHAT_REVISION = "22a40d181f09ab94fdc071f286b41ca5c965466a"

parser = argparse.ArgumentParser(description="Prepare a Hugging Face teacher for distillation")
parser.add_argument("--model", required=True, help="Hugging Face model ID")
parser.add_argument("--out-dir", required=True, help="Output directory for the converted teacher")
parser.add_argument("--revision", default=None, help="Optional Hugging Face revision")
parser.add_argument("--trust-remote-code", action="store_true")
args = parser.parse_args()

if args.model == NANOCHAT_MODEL and args.revision is None:
    args.revision = NANOCHAT_REVISION
    print(f"Using converted Transformers revision {args.revision} for {NANOCHAT_MODEL}")
args.revision = model_info(args.model, revision=args.revision).sha

print(f"Downloading teacher from {args.model}")
tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    revision=args.revision,
    trust_remote_code=args.trust_remote_code,
    use_fast=True,
)
if not tokenizer.is_fast:
    raise RuntimeError("Teacher tokenizer must be a fast Hugging Face tokenizer")

if missing := set(SPECIAL_TOKENS) - tokenizer.get_vocab().keys():
    raise ValueError(f"Teacher tokenizer is missing nanochat tokens: {sorted(missing)}")
tokenizer.chat_template = CHAT_TEMPLATE
tokenizer.bos_token = SPECIAL_TOKENS[0]
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    revision=args.revision,
    trust_remote_code=args.trust_remote_code,
    dtype="auto",
)
if model.get_input_embeddings().num_embeddings != len(tokenizer):
    raise ValueError("Teacher vocabulary must match its tokenizer without resizing")
model.config.teacher_source = {"model": args.model, "revision": args.revision}

os.makedirs(args.out_dir, exist_ok=True)
model.save_pretrained(args.out_dir, safe_serialization=True)
tokenizer.save_pretrained(args.out_dir)

tokenizer_path = os.path.join(args.out_dir, "tokenizer.json")
if not os.path.isfile(tokenizer_path):
    raise RuntimeError("Fast teacher tokenizer did not produce tokenizer.json")

print(f"Saved teacher model and tokenizer to {args.out_dir}")
print(f"Teacher vocabulary size: {len(tokenizer):,}")
