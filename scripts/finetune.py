"""
Supervised fine-tuning (SFT) of a pre-trained tinygpt model.

Loss is computed on assistant tokens only (mask = 1).

Usage:
    python -m scripts.finetune --checkpoint out/checkpoints/d12
    torchrun --nproc_per_node=8 -m scripts.finetune --checkpoint out/checkpoints/d12
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
from functools import partial

import torch
import wandb

from tinygpt.attention import fa2_available
from tinygpt.checkpoint import build_model_from_checkpoint, get_checkpoint_dir, save_checkpoint
from tinygpt.dataloader import sft_data_loader
from tinygpt.gpt import Block
from tinygpt.metrics import compute_token_bytes
from tinygpt.optimizer import make_optimizer
from tinygpt.runtime import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_dtype,
    compute_dtype_reason,
    compute_init,
    make_fsdp_mixed_precision,
    print0,
)
from tinygpt.scheduler import step_scheduler
from tinygpt.tokenizer import HuggingFaceTokenizer

parser = argparse.ArgumentParser(description="Supervised fine-tuning")
parser.add_argument("--checkpoint", type=str, required=True, help="Pre-trained checkpoint directory to start from")
parser.add_argument("--tokenizer-dir", type=str, default="out/tokenizer")
# Logging
parser.add_argument("--run", type=str, default="dummy")
# Runtime
parser.add_argument("--device-type", type=str, default="")
parser.add_argument(
    "--sharding-strategy", type=str, default="FULL_SHARD", choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"]
)
# Training horizon
parser.add_argument("--num-iterations", type=int, default=1000)
# Batch sizes
parser.add_argument("--device-batch-size", type=int, default=8)
parser.add_argument("--max-seq-len", type=int, default=None, help="Sequence length (default: inherit from checkpoint)")
# Optimizer
parser.add_argument("--matrix-lr", type=float, default=0.0003)
parser.add_argument("--embedding-lr", type=float, default=0.003)
parser.add_argument("--scalar-lr", type=float, default=0.03)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--grad-clip", type=float, default=1.0)
# LR schedule
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--warmdown-ratio", type=float, default=0.5)
parser.add_argument("--final-lr-frac", type=float, default=0.0)
# Evaluation
parser.add_argument("--eval-every", type=int, default=200)
# Output
parser.add_argument("--out-dir", type=str, default="out")
parser.add_argument("--run-name", type=str, default="sft")
# Task mixture (uses MMLU + GSM8K by default)
parser.add_argument("--tasks", type=str, default="mmlu,gsm8k", help="Comma-separated task names: mmlu,gsm8k")
args = parser.parse_args()
user_config = vars(args).copy()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
is_dist, rank, local_rank, world_size, device = compute_init(device_type)
master_process = rank == 0

print0(f"compute_dtype: {compute_dtype} ({compute_dtype_reason})")
if not fa2_available:
    print0("WARNING: FA2 not available, using SDPA fallback.")

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="tinygpt-sft", name=args.run, config=user_config)

# Load model + tokenizer
model, meta = build_model_from_checkpoint(args.checkpoint, device, phase="train")
tokenizer = HuggingFaceTokenizer.from_directory(args.tokenizer_dir)
sequence_len = args.max_seq_len or meta["model_config"]["sequence_len"]
token_bytes = compute_token_bytes(tokenizer, device=device)

# FSDP wrap
if device_type == "cuda" and is_dist:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: PLC0415
    from torch.distributed.fsdp import ShardingStrategy  # noqa: PLC0415
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy  # noqa: PLC0415

    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    model = FSDP(
        model,
        sharding_strategy=strategy_map[args.sharding_strategy],
        mixed_precision=make_fsdp_mixed_precision(compute_dtype),
        auto_wrap_policy=wrap_policy,
        device_id=local_rank,
    )

# Task mixture
task_names = {t.strip() for t in args.tasks.split(",")}
task_list = []
if "mmlu" in task_names:
    from tasks.mmlu import MMLU  # noqa: PLC0415

    task_list.append(MMLU(subset="all", split="auxiliary_train"))
if "gsm8k" in task_names:
    from tasks.gsm8k import GSM8K  # noqa: PLC0415

    task_list.append(GSM8K(subset="main", split="train"))

if not task_list:
    raise ValueError(f"No valid tasks found in: {args.tasks}")

from tasks.base import TaskMixture  # noqa: E402,PLC0415

task = TaskMixture(task_list)
print0(f"Task mixture: {len(task)} examples from {task_names}")

# Optimizer
optimizer = make_optimizer(
    model,
    matrix_lr=args.matrix_lr,
    embedding_lr=args.embedding_lr,
    scalar_lr=args.scalar_lr,
    weight_decay=args.weight_decay,
)

# Training loop
train_loader = sft_data_loader(tokenizer, task, args.device_batch_size, sequence_len, device)

checkpoint_dir = get_checkpoint_dir(args.out_dir, args.run_name)
model.train()

smooth_loss = 0.0

for step in range(args.num_iterations):
    last_step = step == args.num_iterations - 1

    # Eval
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        # Quick loss on a few SFT batches
        eval_loader = sft_data_loader(tokenizer, task, args.device_batch_size, sequence_len, device)
        with torch.no_grad():
            losses = [model(*next(eval_loader)).item() for _ in range(10)]
        sft_val_loss = sum(losses) / len(losses)
        print0(f"Step {step:05d} | SFT val loss: {sft_val_loss:.4f}")
        wandb_run.log({"step": step, "val/sft_loss": sft_val_loss})
        model.train()

    # Checkpoint
    if last_step or (args.eval_every > 0 and step > 0 and step % args.eval_every == 0):
        save_checkpoint(
            checkpoint_dir,
            model,
            optimizer,
            {"step": step, "model_config": meta["model_config"], "user_config": user_config},
            rank=rank,
        )

    if last_step:
        break

    x, y = next(train_loader)
    loss = model(x, y)
    loss.backward()

    lrm = step_scheduler(
        optimizer, step, args.num_iterations, args.warmup_steps, args.warmdown_ratio, args.final_lr_frac
    )
    if args.grad_clip > 0:
        if hasattr(model, "clip_grad_norm_"):
            model.clip_grad_norm_(args.grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    optimizer.step()
    model.zero_grad(set_to_none=True)

    ema = 0.9
    smooth_loss = ema * smooth_loss + (1 - ema) * loss.item()
    debiased = smooth_loss / (1 - ema ** (step + 1))
    if step % 50 == 0:
        print0(f"Step {step:05d}/{args.num_iterations} | loss: {debiased:.4f} | lrm: {lrm:.3f}")
        wandb_run.log({"step": step, "train/sft_loss": debiased})

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

wandb_run.finish()
compute_cleanup()
