"""
Distill a student model against a teacher via HuggingFace Trainer.

Loss is computed on assistant tokens only (mask = 1) and combines
supervised CE with online KL distillation from a teacher model.

Usage:
    python -m scripts.distill --checkpoint out/checkpoints/d12 --teacher-model out/teacher_hf
    torchrun --nproc_per_node=8 -m scripts.distill --checkpoint out/checkpoints/d12 --teacher-model out/teacher_hf
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from functools import partial

import torch
from tasks.base import TaskMixture
from tasks.customjson import CustomJSON
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import TrainingArguments

from tinygpt.attention import flash_attn_available
from tinygpt.checkpoint import build_model_from_checkpoint, get_checkpoint_dir
from tinygpt.dataloader import sft_data_loader
from tinygpt.distillation import load_teacher_model, validate_teacher_tokenizer_compatibility
from tinygpt.model import Block
from tinygpt.tokenizer import HuggingFaceTokenizer
from tinygpt.train import TinyGPTTrainer
from tinygpt.utils import (
    autodetect_device_type,
    compute_cleanup,
    compute_dtype,
    compute_dtype_reason,
    compute_init,
    make_fsdp_mixed_precision,
    print0,
)

parser = argparse.ArgumentParser(description="Student-teacher distillation")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Student model directory or Trainer output directory",
)
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
parser.add_argument("--max-seq-len", type=int, default=None, help="Sequence length (default: from checkpoint)")
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
parser.add_argument("--run-name", type=str, default="distill")
# Distillation
parser.add_argument(
    "--teacher-model",
    type=str,
    required=True,
    help="Local directory containing the teacher model in Hugging Face format",
)
parser.add_argument(
    "--teacher-tokenizer",
    type=str,
    default="",
    help="Local directory containing the teacher tokenizer (defaults to --teacher-model)",
)
parser.add_argument(
    "--teacher-device",
    type=str,
    default="same",
    choices=["same", "cuda", "cpu", "mps"],
    help="Device for the teacher model; 'same' reuses the student device",
)
parser.add_argument("--distill-alpha", type=float, default=0.5, help="Teacher KL weight in [0, 1]")
parser.add_argument("--distill-temperature", type=float, default=1.0, help="Temperature for KL distillation")
# Task mixture
parser.add_argument(
    "--tasks",
    type=str,
    default="smoltalk,mmlu,gsm8k",
    help="Comma-separated task groups: smoltalk,mmlu,gsm8k,identity",
)
parser.add_argument(
    "--identity-conversations",
    type=str,
    default="",
    help="Path to identity_conversations.jsonl (used when 'identity' in --tasks)",
)
parser.add_argument("--mmlu-epochs", type=int, default=3)
parser.add_argument("--gsm8k-epochs", type=int, default=4)
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
is_dist, rank, local_rank, world_size, device = compute_init(device_type)
master_process = rank == 0

print0(f"compute_dtype: {compute_dtype} ({compute_dtype_reason})")
if not flash_attn_available:
    print0("WARNING: FA2 not available, using SDPA fallback.")

if not 0.0 <= args.distill_alpha <= 1.0:
    raise ValueError(f"--distill-alpha must be in [0, 1], got {args.distill_alpha}")
if args.distill_temperature <= 0:
    raise ValueError(f"--distill-temperature must be > 0, got {args.distill_temperature}")

model, meta = build_model_from_checkpoint(args.checkpoint, device, phase="train")
tokenizer = HuggingFaceTokenizer.from_directory(args.tokenizer_dir)
sequence_len = args.max_seq_len or meta["model_config"]["sequence_len"]


def resolve_teacher_device() -> torch.device:
    if args.teacher_device == "same":
        return device
    if args.teacher_device == "cuda":
        if device_type != "cuda":
            raise ValueError("--teacher-device=cuda requires CUDA")
        return torch.device("cuda", local_rank) if is_dist else torch.device("cuda")
    return torch.device(args.teacher_device)


teacher_device = resolve_teacher_device()
print0(f"Loading teacher model from {args.teacher_model} on {teacher_device}")
teacher_model, teacher_meta = load_teacher_model(args.teacher_model, device=teacher_device)
teacher_tokenizer_ref = args.teacher_tokenizer or args.teacher_model
teacher_tokenizer = HuggingFaceTokenizer.from_directory(teacher_tokenizer_ref)
validate_teacher_tokenizer_compatibility(tokenizer, teacher_tokenizer)
print0(f"Loaded teacher with vocab size {teacher_meta['model_config']['vocab_size']:,}")

if device_type == "cuda" and is_dist:
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

task_names = {t.strip() for t in args.tasks.split(",")}
task_list = []

if "smoltalk" in task_names:
    task_list.append(SmolTalk(split="train"))

if "identity" in task_names and args.identity_conversations:
    task_list += [CustomJSON(filepath=args.identity_conversations)] * 2

if "mmlu" in task_names:
    task_list += [MMLU(subset="all", split="auxiliary_train")] * args.mmlu_epochs

if "gsm8k" in task_names:
    task_list += [GSM8K(subset="main", split="train")] * args.gsm8k_epochs

if not task_list:
    raise ValueError(f"No valid tasks found in: {args.tasks}")

task = TaskMixture(task_list)
print0(f"Task mixture: {len(task)} examples from {task_names}")

train_loader = sft_data_loader(tokenizer, task, args.device_batch_size, sequence_len, device)


def eval_fn(eval_model: torch.nn.Module, step: int) -> dict[str, float]:
    """Evaluate distillation/SFT loss on 10 batches."""
    eval_loader = sft_data_loader(tokenizer, task, args.device_batch_size, sequence_len, device)
    losses = []
    for _ in range(10):
        x, y = next(eval_loader)
        loss = eval_model(x, y)
        losses.append(loss.item())
    sft_val_loss = sum(losses) / len(losses)
    print0(f"Step {step:05d} | distill val loss: {sft_val_loss:.4f}")
    return {"distill_loss": sft_val_loss}


checkpoint_dir = get_checkpoint_dir(args.out_dir, args.run_name)

training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    max_steps=args.num_iterations,
    per_device_train_batch_size=args.device_batch_size,
    gradient_accumulation_steps=1,
    warmup_steps=args.warmup_steps,
    weight_decay=args.weight_decay,
    max_grad_norm=args.grad_clip,
    logging_steps=50,
    eval_strategy="steps" if args.eval_every > 0 else "no",
    eval_steps=args.eval_every if args.eval_every > 0 else None,
    save_strategy="steps",
    save_steps=args.eval_every if args.eval_every > 0 else args.num_iterations,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    report_to=["wandb"] if args.run != "dummy" and master_process else [],
    run_name=args.run if args.run != "dummy" else None,
    label_names=["labels"],
    fsdp="",
    no_cuda=(device_type != "cuda"),
    bf16=(compute_dtype == torch.bfloat16 and device_type == "cuda"),
    fp16=False,
    prediction_loss_only=True,
    disable_tqdm=not master_process,
)

trainer = TinyGPTTrainer(
    model=model,
    args=training_args,
    matrix_lr=args.matrix_lr,
    embedding_lr=args.embedding_lr,
    scalar_lr=args.scalar_lr,
    warmdown_ratio=args.warmdown_ratio,
    final_lr_frac=args.final_lr_frac,
    train_loader=train_loader,
    eval_fn=eval_fn if args.eval_every > 0 else None,
    teacher_model=teacher_model,
    distill_alpha=args.distill_alpha,
    distill_temperature=args.distill_temperature,
)

trainer.train()
compute_cleanup()
