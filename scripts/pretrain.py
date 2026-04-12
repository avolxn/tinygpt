"""
Pretrain a GPT model via HuggingFace Trainer + FSDP.

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
import json
from dataclasses import asdict
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import TrainingArguments

from tinygpt.attention import flash_attn_available, flash_attn_backend, use_flash_attn
from tinygpt.checkpoint import build_model_from_checkpoint, get_checkpoint_dir, resolve_model_directory
from tinygpt.config import make_config
from tinygpt.dataloader import tokenizing_distributed_data_loader_bestfit
from tinygpt.distributed import (
    compute_cleanup,
    compute_init,
    make_fsdp_mixed_precision,
    print0,
)
from tinygpt.metrics import compute_token_bytes, evaluate_bpb
from tinygpt.model import GPT, Block
from tinygpt.tokenizer import HuggingFaceTokenizer
from tinygpt.train import SamplerCallback, TinyGPTTrainer
from tinygpt.utils import autodetect_device_type, compute_dtype, compute_dtype_reason, get_peak_flops

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
    help="FSDP sharding strategy",
)
# Model architecture
parser.add_argument("--depth", type=int, default=20)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--window-pattern", type=str, default="SSSL")
# Data
parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb")
parser.add_argument("--txt", type=str, default="", help="Local .txt file (overrides --dataset)")
parser.add_argument("--text-field", type=str, default="text")
parser.add_argument("--tokenizer-dir", type=str, default="out/tokenizer")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1)
parser.add_argument("--target-param-data-ratio", type=float, default=12)
# Batch sizes
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--total-batch-size", type=int, default=-1)
# Optimizer (MuonAdamW: Muon for matrix params, AdamW for embeddings/scalars)
parser.add_argument("--matrix-lr", type=float, default=0.02, help="LR for transformer matrix weights (Muon)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="LR for embedding parameters (AdamW)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="LR for scalar/1-D parameters (AdamW)")
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--muon-momentum", type=float, default=0.95, help="Momentum for Muon optimizer")
parser.add_argument("--muon-ns-steps", type=int, default=5, help="Newton-Schulz iterations for Muon")
# LR schedule
parser.add_argument("--warmup-steps", type=int, default=40)
parser.add_argument("--warmdown-ratio", type=float, default=0.65)
parser.add_argument("--final-lr-frac", type=float, default=0.05)
# Resume
parser.add_argument("--resume-from", type=str, default="", help="Model directory or Trainer output directory to resume from")
# Evaluation / sampling
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--eval-tokens", type=int, default=80 * 524288)
parser.add_argument("--sample-every", type=int, default=2000)
parser.add_argument("--save-every", type=int, default=-1)
# Output
parser.add_argument("--out-dir", type=str, default="out")
parser.add_argument("--run-name", type=str, default="")
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
is_dist, rank, local_rank, world_size, device = compute_init(device_type)
master_process = rank == 0

if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_name)
    print0(f"GPU: {gpu_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")

print0(f"compute_dtype: {compute_dtype} ({compute_dtype_reason})")

if use_flash_attn:
    print0(f"Using {flash_attn_backend}.")
else:
    print0("!" * 70)
    if flash_attn_available and compute_dtype != torch.bfloat16:
        print0(f"WARNING: flash-attn available but requires bf16, compute_dtype={compute_dtype}. Using SDPA.")
    else:
        print0("WARNING: flash-attn not available (install flash-attn-4 / kernels / flash-attn). Using SDPA fallback.")
    print0("!" * 70)

tokenizer = HuggingFaceTokenizer.from_directory(args.tokenizer_dir)
vocab_size = tokenizer.get_vocab_size()
token_bytes = compute_token_bytes(tokenizer, device=device)
print0(f"Vocab size: {vocab_size:,}")

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

start_step = 0
if args.resume_from:
    print0(f"Resuming model weights from {args.resume_from}")
    resolved_resume_dir = resolve_model_directory(args.resume_from)
    print0(f"Resolved resume checkpoint: {resolved_resume_dir}")
    model, resume_meta = build_model_from_checkpoint(resolved_resume_dir, device, phase="train")
    start_step = int(resume_meta.get("step", 0))
    print0(f"Resumed at step {start_step}")


if device_type == "cuda" and is_dist:
    if args.sharding_strategy != "NO_SHARD":
        print0("!" * 70)
        print0(f"WARNING: sharding_strategy={args.sharding_strategy} shards parameters along")
        print0("         the first dimension. Muon's Newton-Schulz orthogonalization")
        print0("         requires full matrices — results will be INCORRECT with sharding.")
        print0("         Use --sharding-strategy NO_SHARD for correct Muon behavior.")
        print0("!" * 70)
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

param_counts = model.num_scaling_params() if hasattr(model, "num_scaling_params") else {}
if param_counts:
    print0("Parameter counts:")
    for k, v in param_counts.items():
        print0(f"  {k:<24}: {v:,}")
num_params = sum(p.numel() for p in model.parameters())
print0(f"Total params: {num_params:,}")

if args.num_iterations > 0:
    num_iterations = args.num_iterations
elif param_counts and args.target_param_data_ratio > 0:
    scaling_params = param_counts.get("transformer_matrices", 0) + param_counts.get("lm_head", 0)
    target_tokens = int(args.target_param_data_ratio * scaling_params)
    total_batch = args.total_batch_size if args.total_batch_size > 0 else 524288
    num_iterations = max(1, target_tokens // total_batch)
else:
    num_iterations = 1000

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

run_name = args.run_name if args.run_name else f"d{args.depth}"
checkpoint_dir = get_checkpoint_dir(args.out_dir, run_name, phase="pretrain")


def make_loader(split: str):
    if args.txt:
        return _txt_loader(tokenizer, args.txt, args.device_batch_size, args.max_seq_len, device)
    return tokenizing_distributed_data_loader_bestfit(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        dataset_name=args.dataset,
        split=split,
        device=device,
        text_field=args.text_field,
    )


def _txt_loader(tok, path: str, B: int, T: int, dev):
    """Minimal bestfit loader backed by a local text file."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    bos = tok.get_bos_token_id()
    row_capacity = T + 1
    doc_buffer: list[list[int]] = []

    def refill() -> None:
        for ln in lines:
            doc_buffer.append(tok.encode(ln, prepend=bos))

    use_cuda = str(dev) == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=dev)
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
eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * world_size))


def eval_fn(eval_model: torch.nn.Module, step: int) -> dict[str, float]:
    """Evaluate bits-per-byte on the validation split."""
    eval_loader = make_loader("val")
    bpb = evaluate_bpb(eval_model, eval_loader, eval_steps, token_bytes)
    print0(f"Step {step:05d} | val bpb: {bpb:.6f}")
    return {"bpb": bpb}


training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    max_steps=num_iterations,
    per_device_train_batch_size=args.device_batch_size,
    gradient_accumulation_steps=grad_accum_steps,
    warmup_steps=args.warmup_steps,
    weight_decay=args.weight_decay,
    max_grad_norm=args.grad_clip,
    logging_steps=100,
    eval_strategy="steps" if args.eval_every > 0 else "no",
    eval_steps=args.eval_every if args.eval_every > 0 else None,
    save_strategy="steps",
    save_steps=args.save_every if args.save_every > 0 else num_iterations,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    report_to=["wandb"] if args.run != "dummy" and master_process else [],
    run_name=args.run if args.run != "dummy" else None,
    label_names=["labels"],
    fsdp="",  # We pre-wrap with FSDP above
    no_cuda=(device_type != "cuda"),
    bf16=(compute_dtype == torch.bfloat16 and device_type == "cuda"),
    fp16=False,
    prediction_loss_only=True,
    disable_tqdm=not master_process,
)

callbacks = [
    SamplerCallback(
        tokenizer=tokenizer,
        device=device,
        sample_every=args.sample_every,
        master_process=master_process,
    ),
]

trainer = TinyGPTTrainer(
    model=model,
    args=training_args,
    callbacks=callbacks,
    matrix_lr=args.matrix_lr,
    embedding_lr=args.embedding_lr,
    scalar_lr=args.scalar_lr,
    muon_momentum=args.muon_momentum,
    muon_ns_steps=args.muon_ns_steps,
    warmdown_ratio=args.warmdown_ratio,
    final_lr_frac=args.final_lr_frac,
    train_loader=train_loader,
    eval_fn=eval_fn if args.eval_every > 0 else None,
)

if start_step > 0:
    trainer.state.global_step = start_step

trainer.train()
compute_cleanup()
