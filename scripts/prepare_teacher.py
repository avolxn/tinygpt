"""Download an unchanged Hugging Face teacher snapshot for training."""

import argparse
import json
from pathlib import Path

from huggingface_hub import model_info, snapshot_download
from transformers import AutoConfig, AutoTokenizer

parser = argparse.ArgumentParser(description="Download a Hugging Face teacher")
parser.add_argument("--model", required=True, help="Hugging Face model ID")
parser.add_argument("--out-dir", required=True, help="Local snapshot directory")
parser.add_argument("--revision", default=None, help="Optional Hugging Face revision")
args = parser.parse_args()
source_file = Path(args.out_dir, "teacher_source.json")
if Path(args.out_dir, "config.json").exists():
    source = json.loads(source_file.read_text()) if source_file.exists() else {}
    if source.get("model") != args.model:
        raise ValueError("Use a separate teacher directory for a different model to avoid mixing snapshots")
revision = model_info(args.model, revision=args.revision).sha
if source_file.exists() and json.loads(source_file.read_text()).get("revision") != revision:
    raise ValueError("Use a separate teacher directory for a different revision")
source_file.parent.mkdir(parents=True, exist_ok=True)
source_file.write_text(json.dumps({"model": args.model, "revision": revision}) + "\n")
snapshot_download(
    args.model,
    revision=revision,
    local_dir=args.out_dir,
    allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.tiktoken", "merges.txt", "vocab.*"],
)
tokenizer = AutoTokenizer.from_pretrained(args.out_dir, use_fast=True)
config = AutoConfig.from_pretrained(args.out_dir)
if not tokenizer.is_fast or not tokenizer.chat_template:
    raise ValueError("Teacher requires a fast tokenizer with a chat template")
if tokenizer.eos_token_id is None:
    raise ValueError("Teacher tokenizer must define EOS")
if config.vocab_size < len(tokenizer):
    raise ValueError("Teacher vocabulary must cover its tokenizer without resizing")
print(f"Downloaded {args.model}@{revision} to {args.out_dir}")
