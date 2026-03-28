"""
Evaluate a trained tinygpt model.

Supported modes (comma-separated via --eval):
  bpb     : bits-per-byte on train/val splits
  sample  : sample from the model

Usage:
    python -m scripts.evaluate --checkpoint out/checkpoints/d12
    python -m scripts.evaluate --checkpoint out/checkpoints/d12 --eval bpb
    torchrun --nproc_per_node=4 -m scripts.evaluate --checkpoint out/checkpoints/d12
"""

import argparse
import torch

from tinygpt.runtime import autodetect_device_type, compute_init, compute_cleanup, print0
from tinygpt.checkpoint import build_model_from_checkpoint
from tinygpt.tokenizer import HuggingFaceTokenizer
from tinygpt.dataloader import tokenizing_distributed_data_loader_bestfit
from tinygpt.metrics import evaluate_bpb, compute_token_bytes
from tinygpt.engine import Engine

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate tinygpt model")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint directory")
parser.add_argument("--tokenizer-dir", type=str, default="out/tokenizer")
parser.add_argument("--eval", type=str, default="bpb,sample",
                    help="Comma-separated modes: bpb,sample")
parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb")
parser.add_argument("--text-field", type=str, default="text")
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--split-tokens", type=int, default=40 * 524288)
parser.add_argument("--device-type", type=str, default="")
args = parser.parse_args()

eval_modes = {m.strip() for m in args.eval.split(",")}

# ---------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
_, rank, _, world_size, device = compute_init(device_type)

# ---------------------------------------------------------------------------
model, meta = build_model_from_checkpoint(args.checkpoint, device, phase="eval")
tokenizer = HuggingFaceTokenizer.from_directory(args.tokenizer_dir)
token_bytes = compute_token_bytes(tokenizer, device=device)
sequence_len = meta["model_config"]["sequence_len"]

print0(f"Loaded model from {args.checkpoint} (step {meta.get('step', '?')})")
print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

# ---------------------------------------------------------------------------
# Sampling
if "sample" in eval_modes and rank == 0:
    print0("\n" + "=" * 70)
    print0("Samples")
    print0("=" * 70)
    engine = Engine(model, tokenizer)
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
    ]
    for prompt in prompts:
        tokens = tokenizer(prompt, prepend="<|bos|>")
        sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=20, temperature=0)
        print0(f"{tokenizer.decode(sample[0])}")

# ---------------------------------------------------------------------------
# BPB
if "bpb" in eval_modes:
    print0("\n" + "=" * 70)
    print0("BPB Evaluation")
    print0("=" * 70)
    tokens_per_step = args.device_batch_size * sequence_len * world_size
    steps = max(1, args.split_tokens // tokens_per_step)

    for split in ("train", "val"):
        loader = tokenizing_distributed_data_loader_bestfit(
            tokenizer,
            args.device_batch_size,
            sequence_len,
            dataset_name=args.dataset,
            split=split,
            device=device,
            text_field=args.text_field,
        )
        bpb = evaluate_bpb(model, loader, steps, token_bytes)
        print0(f"{split} bpb: {bpb:.6f}")

compute_cleanup()
