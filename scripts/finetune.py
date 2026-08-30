"""
Supervised fine-tuning (SFT) via HuggingFace Trainer.

Loss is computed on assistant tokens only (mask = 1).

Usage:
    python -m scripts.finetune --checkpoint data/pretrain_checkpoints/from_scratch
    torchrun --nproc_per_node=8 -m scripts.finetune --checkpoint data/pretrain_checkpoints/from_scratch
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import math
from functools import partial

import torch
from tasks.base import TaskMixture
from tasks.sft import build_sft_task_lists
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from tinygpt.attention import flash_attn_available
from tinygpt.checkpoint import (
    build_checkpoint_metadata,
    build_model_from_checkpoint,
    get_checkpoint_dir,
    resolve_trainer_checkpoint,
)
from tinygpt.config import RuntimeConfig, add_runtime_arguments
from tinygpt.dataloader import sft_data_loader
from tinygpt.distributed import (
    compute_cleanup,
    compute_init,
    get_dist_info,
    make_fsdp_mixed_precision,
    print0,
)
from tinygpt.model import Block
from tinygpt.tokenizer import HuggingFaceTokenizer
from tinygpt.tracking import require_wandb_auth
from tinygpt.train import RunMetadataCallback, TinyGPTTrainer, build_training_arguments, resolve_training_value
from tinygpt.utils import autodetect_device_type, compute_dtype, compute_dtype_reason

parser = argparse.ArgumentParser(description="Supervised fine-tuning")
add_runtime_arguments(parser)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Pre-trained model directory or Trainer output directory",
)
parser.add_argument("--resume-from", type=str, default="", help="SFT model or full Trainer checkpoint to resume")
parser.add_argument(
    "--sharding-strategy", type=str, default="FULL_SHARD", choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"]
)
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="Optimization steps (-1 = approximate one epoch)")
# Batch sizes (default: inherit from pretrain metadata when available)
parser.add_argument("--device-batch-size", type=int, default=None)
parser.add_argument("--total-batch-size", type=int, default=None, help="Total batch size in tokens")
parser.add_argument("--max-seq-len", type=int, default=None, help="Sequence length (default: from checkpoint)")
# Optimizer
parser.add_argument("--matrix-lr", type=float, default=None)
parser.add_argument("--lm-head-lr", type=float, default=None)
parser.add_argument("--embedding-lr", type=float, default=None)
parser.add_argument("--scalar-lr", type=float, default=None)
parser.add_argument("--init-lr-frac", type=float, default=0.8)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--grad-clip", type=float, default=1.0)
# LR schedule
parser.add_argument("--warmup-ratio", type=float, default=0.0)
parser.add_argument("--warmdown-ratio", type=float, default=0.5)
parser.add_argument("--final-lr-frac", type=float, default=0.0)
# Evaluation
parser.add_argument("--eval-every", type=int, default=200)
parser.add_argument("--eval-tokens", type=int, default=40 * 524288)
# Task mixture
parser.add_argument(
    "--tasks",
    type=str,
    default="smoltalk,mmlu,gsm8k,identity,spelling",
    help="Comma-separated task groups: smoltalk,mmlu,gsm8k,identity,spelling",
)
parser.add_argument(
    "--identity-conversations",
    type=str,
    default="data/identity_conversations.jsonl",
    help="Path to identity_conversations.jsonl",
)
parser.add_argument("--mmlu-epochs", type=int, default=3)
parser.add_argument("--gsm8k-epochs", type=int, default=4)


args = parser.parse_args()
runtime_config = RuntimeConfig.from_namespace(args)

dist_requested, preflight_rank, _, _ = get_dist_info()
if preflight_rank == 0:
    require_wandb_auth(interactive=not dist_requested)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
is_dist, rank, local_rank, world_size, device = compute_init(device_type, seed=runtime_config.seed)
master_process = rank == 0

print0(f"compute_dtype: {compute_dtype} ({compute_dtype_reason})")
if not flash_attn_available:
    print0("WARNING: FA2 not available, using SDPA fallback.")

model_ref = args.resume_from or args.checkpoint
model, meta = build_model_from_checkpoint(model_ref, device, phase="train")
resume_checkpoint = resolve_trainer_checkpoint(model_ref) if args.resume_from else None
if args.resume_from and resume_checkpoint is None:
    print0("No complete Trainer state found; continuing from weights only.")
tokenizer = HuggingFaceTokenizer.from_directory(args.tokenizer_dir)
sequence_len = args.max_seq_len or meta["model_config"]["sequence_len"]
pretrain_user_config = meta.get("user_config", {})


args.device_batch_size = int(
    resolve_training_value(
        "device_batch_size",
        args.device_batch_size,
        32,
        meta.get("device_batch_size"),
        pretrain_user_config.get("device_batch_size"),
    )
)
args.total_batch_size = int(
    resolve_training_value(
        "total_batch_size",
        args.total_batch_size,
        524288,
        meta.get("total_batch_size"),
        pretrain_user_config.get("total_batch_size"),
    )
)
args.matrix_lr = float(resolve_training_value("matrix_lr", args.matrix_lr, 0.02, pretrain_user_config.get("matrix_lr")))
args.lm_head_lr = float(
    resolve_training_value(
        "lm_head_lr",
        args.lm_head_lr,
        0.004,
        pretrain_user_config.get("lm_head_lr"),
        pretrain_user_config.get("unembedding_lr"),
    )
)
args.embedding_lr = float(
    resolve_training_value("embedding_lr", args.embedding_lr, 0.3, pretrain_user_config.get("embedding_lr"))
)
args.scalar_lr = float(resolve_training_value("scalar_lr", args.scalar_lr, 0.5, pretrain_user_config.get("scalar_lr")))

args.matrix_lr *= args.init_lr_frac
args.lm_head_lr *= args.init_lr_frac
args.embedding_lr *= args.init_lr_frac
args.scalar_lr *= args.init_lr_frac

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

task_names = {t.strip() for t in args.tasks.split(",") if t.strip()}
task_list, val_task_list = build_sft_task_lists(
    task_names,
    identity_conversations=args.identity_conversations,
    mmlu_epochs=args.mmlu_epochs,
    gsm8k_epochs=args.gsm8k_epochs,
)

if not task_list:
    raise ValueError(f"No valid tasks found in: {args.tasks}")

task = TaskMixture(task_list)
print0(f"Task mixture: {len(task):,} examples from {task_names}")

val_task = TaskMixture(val_task_list) if args.eval_every > 0 else None
train_loader = sft_data_loader(tokenizer, task, args.device_batch_size, sequence_len, device)

tokens_per_fwdbwd = args.device_batch_size * sequence_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * world_size
if args.total_batch_size % world_tokens_per_fwdbwd != 0:
    raise ValueError(
        f"total_batch_size {args.total_batch_size} must be divisible by "
        f"device_batch_size*seq_len*world_size = {world_tokens_per_fwdbwd}"
    )
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {sequence_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

if args.num_iterations > 0:
    num_iterations = args.num_iterations
else:
    approx_rows_per_step = max(1, args.device_batch_size * world_size * grad_accum_steps)
    num_iterations = max(1, math.ceil(len(task) / approx_rows_per_step))
    print0(f"Approximated one-epoch num_iterations={num_iterations:,} from {len(task):,} mixed conversations")

warmup_steps = round(args.warmup_ratio * num_iterations)
eval_steps = max(1, args.eval_tokens // args.total_batch_size)


def eval_fn(eval_model: torch.nn.Module, step: int) -> dict[str, float]:
    """Evaluate SFT loss on the held-out validation mixture."""
    assert val_task is not None
    eval_loader = sft_data_loader(tokenizer, val_task, args.device_batch_size, sequence_len, device)
    losses = []
    for _ in range(eval_steps):
        x, y = next(eval_loader)
        loss = eval_model(x, y)
        losses.append(loss.item())
    sft_val_loss = sum(losses) / len(losses)
    print0(f"Step {step:05d} | SFT val loss: {sft_val_loss:.4f}")
    return {"sft_loss": sft_val_loss}


run_name = runtime_config.run_name if runtime_config.run_name else f"d{meta['model_config']['n_layer']}"
runtime_config = runtime_config.with_run_name(run_name)
checkpoint_dir = get_checkpoint_dir(runtime_config.out_dir, runtime_config.run_name, phase="sft")
wandb_run_name = runtime_config.run if runtime_config.run else runtime_config.run_name
os.environ.setdefault("WANDB_PROJECT", "tinygpt")

training_args = build_training_arguments(
    output_dir=checkpoint_dir,
    max_steps=num_iterations,
    device_batch_size=args.device_batch_size,
    grad_accum_steps=grad_accum_steps,
    warmup_steps=warmup_steps,
    weight_decay=args.weight_decay,
    grad_clip=args.grad_clip,
    logging_steps=50,
    eval_every=args.eval_every,
    save_steps=args.eval_every if args.eval_every > 0 else num_iterations,
    run_name=wandb_run_name,
    report_to=["wandb"] if master_process else [],
    device_type=device_type,
    compute_dtype=compute_dtype,
    disable_tqdm=not master_process,
)

checkpoint_metadata = build_checkpoint_metadata(
    phase="sft",
    args=args,
    runtime_config=runtime_config,
    device_batch_size=args.device_batch_size,
    max_seq_len=sequence_len,
    total_batch_size=args.total_batch_size,
    grad_accum_steps=grad_accum_steps,
    num_iterations=num_iterations,
)

trainer = TinyGPTTrainer(
    model=model,
    args=training_args,
    callbacks=[RunMetadataCallback(checkpoint_metadata)],
    eval_dataset=[0] if args.eval_every > 0 else None,
    matrix_lr=args.matrix_lr,
    embedding_lr=args.embedding_lr,
    scalar_lr=args.scalar_lr,
    lm_head_lr=args.lm_head_lr,
    warmdown_ratio=args.warmdown_ratio,
    final_lr_frac=args.final_lr_frac,
    train_loader=train_loader,
    eval_fn=eval_fn if args.eval_every > 0 else None,
    tokenizer_dir=args.tokenizer_dir,
    checkpoint_metadata=checkpoint_metadata,
)

trainer.train(resume_from_checkpoint=resume_checkpoint)
compute_cleanup()
