"""
Pretrain a GPT model with FSDP.

Single GPU:
    python -m scripts.pretrain

Multi-GPU (e.g. 8 GPUs):
    torchrun --nproc_per_node=8 -m scripts.pretrain

Small CPU/MPS test:
    python -m scripts.pretrain --depth 4 --max-seq-len 512 \
        --device-batch-size 1 --total-batch-size 512 \
        --num-iterations 20 --eval-every -1 \
        --dataset "" --txt data/shakespeare.txt
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import time
from dataclasses import asdict
from functools import partial

import torch
import wandb

from tinygpt.attention import fa2_available, use_fa2
from tinygpt.checkpoint import get_checkpoint_dir, save_checkpoint
from tinygpt.config import make_config
from tinygpt.dataloader import tokenizing_distributed_data_loader_bestfit
from tinygpt.engine import Engine
from tinygpt.gpt import GPT, Block
from tinygpt.metrics import compute_token_bytes, evaluate_bpb
from tinygpt.optimizer import make_optimizer
from tinygpt.runtime import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_dtype,
    compute_dtype_reason,
    compute_init,
    get_peak_flops,
    make_fsdp_mixed_precision,
    print0,
)
from tinygpt.scheduler import step_scheduler
from tinygpt.tokenizer import HuggingFaceTokenizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Pretrain tinygpt")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Distributed
parser.add_argument(
    "--sharding-strategy",
    type=str,
    default="FULL_SHARD",
    choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
    help="FSDP sharding strategy (FULL_SHARD=ZeRO-3, SHARD_GRAD_OP=ZeRO-2)",
)
# Model architecture
parser.add_argument("--depth", type=int, default=20)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--window-pattern", type=str, default="SSSL")
# Data
parser.add_argument(
    "--dataset", type=str, default="HuggingFaceFW/fineweb", help="HF dataset identifier (empty = use --txt)"
)
parser.add_argument("--txt", type=str, default="", help="Local .txt file to train on (overrides --dataset)")
parser.add_argument("--text-field", type=str, default="text")
parser.add_argument("--tokenizer-dir", type=str, default="out/tokenizer")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1)
parser.add_argument("--target-param-data-ratio", type=float, default=12)
# Batch sizes
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--total-batch-size", type=int, default=-1)
# Optimizer
parser.add_argument("--matrix-lr", type=float, default=0.001)
parser.add_argument("--embedding-lr", type=float, default=0.01)
parser.add_argument("--scalar-lr", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--grad-clip", type=float, default=1.0)
# LR schedule
parser.add_argument("--warmup-steps", type=int, default=40)
parser.add_argument("--warmdown-ratio", type=float, default=0.65)
parser.add_argument("--final-lr-frac", type=float, default=0.05)
# Resume
parser.add_argument("--resume-from", type=str, default="", help="Checkpoint directory to resume from")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--eval-tokens", type=int, default=80 * 524288)
parser.add_argument("--sample-every", type=int, default=2000)
parser.add_argument("--save-every", type=int, default=-1)
# Output
parser.add_argument("--out-dir", type=str, default="out")
parser.add_argument(
    "--run-name", type=str, default="", help="Subdirectory name under out/checkpoints/ (default: d<depth>)"
)
args = parser.parse_args()
user_config = vars(args).copy()

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
is_dist, rank, local_rank, world_size, device = compute_init(device_type)
master_process = rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_name)
    print0(f"GPU: {gpu_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")

print0(f"compute_dtype: {compute_dtype} ({compute_dtype_reason})")

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="tinygpt", name=args.run, config=user_config)

# Flash attention status
if use_fa2:
    print0("Using Flash Attention 2 (Ampere+ GPU detected).")
else:
    print0("!" * 70)
    if fa2_available and compute_dtype != torch.bfloat16:
        print0(f"WARNING: FA2 only supports bf16, compute_dtype={compute_dtype}. Using SDPA.")
    else:
        print0("WARNING: Flash Attention 2 not available, using PyTorch SDPA fallback.")
    print0("!" * 70)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
if args.txt:
    # Tiny dataset path: inline train from txt
    print0(f"Loading tokenizer from {args.tokenizer_dir}")
else:
    print0(f"Loading tokenizer from {args.tokenizer_dir}")

tokenizer = HuggingFaceTokenizer.from_directory(args.tokenizer_dir)
vocab_size = tokenizer.get_vocab_size()
token_bytes = compute_token_bytes(tokenizer, device=device)
print0(f"Vocab size: {vocab_size:,}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
config = make_config(
    args.depth,
    aspect_ratio=args.aspect_ratio,
    head_dim=args.head_dim,
    vocab_size=vocab_size,
    sequence_len=args.max_seq_len,
    window_pattern=args.window_pattern,
)
model_config_kwargs = asdict(config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

# ---------------------------------------------------------------------------
# FSDP wrap (CUDA only)
# ---------------------------------------------------------------------------
if device_type == "cuda" and is_dist:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

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
    print0(f"FSDP enabled with sharding strategy: {args.sharding_strategy}")

# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------
param_counts = model.num_scaling_params() if hasattr(model, "num_scaling_params") else {}
if param_counts:
    print0("Parameter counts:")
    for k, v in param_counts.items():
        print0(f"  {k:<24}: {v:,}")
num_params = sum(p.numel() for p in model.parameters())
num_flops_per_token = model.estimate_flops() if hasattr(model, "estimate_flops") else 0
print0(f"Total params: {num_params:,}")
print0(f"Estimated FLOPs/token: {num_flops_per_token:e}")

# ---------------------------------------------------------------------------
# Training horizon
# ---------------------------------------------------------------------------
if args.num_iterations > 0:
    num_iterations = args.num_iterations
elif param_counts and args.target_param_data_ratio > 0:
    scaling_params = param_counts.get("transformer_matrices", 0) + param_counts.get("lm_head", 0)
    target_tokens = int(args.target_param_data_ratio * scaling_params)
    total_batch = args.total_batch_size if args.total_batch_size > 0 else 524288
    num_iterations = max(1, target_tokens // total_batch)
else:
    num_iterations = 1000

# Batch size
if args.total_batch_size > 0:
    total_batch_size = args.total_batch_size
else:
    total_batch_size = args.device_batch_size * args.max_seq_len * world_size

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens = tokens_per_fwdbwd * world_size
if total_batch_size % world_tokens != 0:
    raise ValueError(
        f"total_batch_size {total_batch_size} must be divisible by "
        f"world_tokens {world_tokens} = device_batch_size*seq_len*world_size"
    )
grad_accum_steps = total_batch_size // world_tokens

print0(f"num_iterations: {num_iterations:,}")
print0(f"total_batch_size: {total_batch_size:,}")
print0(f"grad_accum_steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Optimizer + resume
# ---------------------------------------------------------------------------
optimizer = make_optimizer(
    model,
    matrix_lr=args.matrix_lr,
    embedding_lr=args.embedding_lr,
    scalar_lr=args.scalar_lr,
    weight_decay=args.weight_decay,
)

step = 0
val_bpb = None
min_val_bpb = float("inf")
smooth_train_loss = 0.0
total_training_time = 0.0

if args.resume_from:
    from tinygpt.checkpoint import load_checkpoint  # noqa: PLC0415

    print0(f"Resuming from {args.resume_from}")
    meta = load_checkpoint(args.resume_from, model, optimizer, device, rank=rank)
    step = meta.get("step", 0)
    val_bpb = meta.get("val_bpb")
    loop_state = meta.get("loop_state", {})
    min_val_bpb = loop_state.get("min_val_bpb", float("inf"))
    smooth_train_loss = loop_state.get("smooth_train_loss", 0.0)
    total_training_time = loop_state.get("total_training_time", 0.0)
    print0(f"Resumed at step {step}")

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def make_loader(split: str):
    if args.txt:
        # Wrap local txt file as a trivial HF-like streaming source
        # We'll use the tokenizing loader but point at our custom iterator
        # For simplicity, fall back to the bestfit loader with the txt file
        # exposed as a local HF dataset:
        dataset = _TxtDataset(args.txt)
        return _txt_loader(tokenizer, dataset, args.device_batch_size, args.max_seq_len, device)
    return tokenizing_distributed_data_loader_bestfit(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        dataset_name=args.dataset,
        split=split,
        device=device,
        text_field=args.text_field,
    )


class _TxtDataset:
    """Minimal wrapper so local txt files work like HF streaming datasets."""

    def __init__(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]

    def __iter__(self):
        while True:
            yield from self.lines


def _txt_loader(tokenizer, dataset, B, T, device):
    """Minimal bestfit loader backed by a local text iterator."""
    bos = tokenizer.get_bos_token_id()
    row_capacity = T + 1
    doc_buffer = []
    data_iter = iter(dataset)

    def refill():
        for _ in range(128):
            try:
                doc_buffer.append(tokenizer.encode(next(data_iter), prepend=bos))
            except StopIteration:
                break

    use_cuda = str(device) == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T :].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T :].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < 200:
                    refill()
                    if not doc_buffer:
                        break
                if not doc_buffer:
                    break
                remaining = row_capacity - pos
                best_idx = max(
                    (i for i in range(len(doc_buffer)) if len(doc_buffer[i]) <= remaining),
                    key=lambda i: len(doc_buffer[i]),
                    default=-1,
                )
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    dl = len(doc)
                    row_buffer[row_idx, pos : pos + dl] = torch.tensor(doc, dtype=torch.long)
                    pos += dl
                else:
                    si = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(si)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets


train_loader = make_loader("train")
x, y = next(train_loader)

# ---------------------------------------------------------------------------
# Checkpoint dir
# ---------------------------------------------------------------------------
run_name = args.run_name if args.run_name else f"d{args.depth}"
checkpoint_dir = get_checkpoint_dir(args.out_dir, run_name)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
model.train()

while True:
    last_step = step == num_iterations

    # -----------------------------------------------------------------------
    # Eval
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = make_loader("val")
        eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * world_size))
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | val bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({"step": step, "val/bpb": val_bpb, "total_time": total_training_time})
        model.train()

    # Sample
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        engine = Engine(model, tokenizer)
        prompts = ["The capital of France is", "The chemical symbol of gold is"]
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # Checkpoint
    if last_step or (step > 0 and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            model,
            optimizer,
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=rank,
        )

    if last_step:
        break

    # -----------------------------------------------------------------------
    # Training step
    synchronize()
    t0 = time.time()
    for _ in range(grad_accum_steps):
        loss = model(x, y)
        train_loss_detach = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)

    lrm = step_scheduler(optimizer, step, num_iterations, args.warmup_steps, args.warmdown_ratio, args.final_lr_frac)

    if args.grad_clip > 0:
        if hasattr(model, "clip_grad_norm_"):
            model.clip_grad_norm_(args.grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss_detach.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))
    if step > 10:
        total_training_time += dt
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * world_size)
    pct = 100 * step / num_iterations

    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct:.1f}%) | "
        f"loss: {debiased:.4f} | lrm: {lrm:.3f} | "
        f"dt: {dt * 1000:.1f}ms | tok/s: {tok_per_sec:,} | mfu: {mfu:.1f}%"
    )
    if step % 100 == 0:
        wandb_run.log(
            {
                "step": step,
                "train/loss": debiased,
                "train/lrm": lrm,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
            }
        )

    first_step = step == 0 or (
        args.resume_from and step == int(args.resume_from.split("_")[-1]) if args.resume_from else False
    )
    step += 1
    if step == 1:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.1f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
wandb_run.finish()
compute_cleanup()
