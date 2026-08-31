"""
Pretrain a Llama model via Transformers Trainer + native PyTorch FSDP.

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

import torch
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from tinygpt.checkpoint import (
    build_checkpoint_metadata,
    build_model_from_checkpoint,
    get_checkpoint_dir,
    resolve_model_directory,
    resolve_trainer_checkpoint,
)
from tinygpt.config import (
    RuntimeConfig,
    add_runtime_arguments,
    compute_scaled_total_batch_size,
    compute_scaled_weight_decay,
    make_config,
)
from tinygpt.dataloader import CLIMBMIX_DATASET, text_data_loader, tokenizing_distributed_data_loader_bestfit
from tinygpt.distributed import (
    compute_cleanup,
    compute_init,
    get_dist_info,
    print0,
    wrap_fsdp,
)
from tinygpt.metrics import compute_token_bytes, evaluate_bpb
from tinygpt.tokenizer import HuggingFaceTokenizer
from tinygpt.tracking import require_mlflow_tracking
from tinygpt.train import MlflowMetadataCallback, TinyGPTTrainer, build_training_arguments
from tinygpt.utils import autodetect_device_type, compute_dtype, compute_dtype_reason, get_peak_flops

parser = argparse.ArgumentParser(description="Pretrain tinygpt")
add_runtime_arguments(parser)
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
# Data
parser.add_argument("--dataset", type=str, default=CLIMBMIX_DATASET)
parser.add_argument("--txt", type=str, default="", help="Local .txt file (overrides --dataset)")
parser.add_argument("--text-field", type=str, default="text")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1)
parser.add_argument("--target-param-data-ratio", type=float, default=12)
# Batch sizes
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--total-batch-size", type=int, default=-1)
# Optimizer (MuonAdamW: Muon for matrix params, AdamW for embeddings/scalars)
parser.add_argument("--matrix-lr", type=float, default=0.02, help="LR for transformer matrix weights (Muon)")
parser.add_argument("--lm-head-lr", type=float, default=0.008, help="LR for lm_head / unembedding parameters (AdamW)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="LR for embedding parameters (AdamW)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="LR for scalar/1-D parameters (AdamW)")
parser.add_argument("--weight-decay", type=float, default=0.28)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--muon-momentum", type=float, default=0.95, help="Momentum for Muon optimizer")
parser.add_argument("--muon-ns-steps", type=int, default=5, help="Newton-Schulz iterations for Muon")
# LR schedule
parser.add_argument("--warmup-steps", type=int, default=40)
parser.add_argument("--warmdown-ratio", type=float, default=0.65)
parser.add_argument("--final-lr-frac", type=float, default=0.05)
# Resume
parser.add_argument(
    "--resume-from", type=str, default="", help="Model directory or Trainer output directory to resume from"
)
# Evaluation / sampling
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--eval-tokens", type=int, default=80 * 524288)
parser.add_argument("--save-every", type=int, default=-1)


args = parser.parse_args()
runtime_config = RuntimeConfig.from_namespace(args)


def scaling_param_counts(model: LlamaForCausalLM) -> dict[str, int]:
    """Bucket native Llama parameters for the scaling-law calculation."""
    counts = {
        "wte": model.get_input_embeddings().weight.numel(),
        "value_embeds": 0,
        "lm_head": model.lm_head.weight.numel(),
        "transformer_matrices": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("model.layers.") and parameter.dim() >= 2
        ),
        "scalars": sum(parameter.numel() for parameter in model.parameters() if parameter.dim() < 2),
    }
    counts["total"] = sum(counts.values())
    return counts

dist_requested, preflight_rank, _, _ = get_dist_info()
if preflight_rank == 0:
    require_mlflow_tracking()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
is_dist, rank, local_rank, world_size, device = compute_init(device_type, seed=runtime_config.seed)
master_process = rank == 0

if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_name)
    print0(f"GPU: {gpu_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")

print0(f"compute_dtype: {compute_dtype} ({compute_dtype_reason})")

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
)
model_config_kwargs = config.to_dict()
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")

with torch.device("meta"):
    model = LlamaForCausalLM(config)
model.to_empty(device=device)
model.apply(model._init_weights)  # type: ignore[attr-defined]

start_step = 0
resume_checkpoint = None
if args.resume_from:
    print0(f"Resuming model weights from {args.resume_from}")
    resolved_resume_dir = resolve_model_directory(args.resume_from)
    print0(f"Resolved resume checkpoint: {resolved_resume_dir}")
    model, resume_meta = build_model_from_checkpoint(resolved_resume_dir, device, phase="train")
    start_step = int(resume_meta.get("step", 0))
    print0(f"Resumed at step {start_step}")
    resume_checkpoint = resolve_trainer_checkpoint(resolved_resume_dir)
    if resume_checkpoint is None:
        print0("No complete Trainer state found; continuing from weights only.")

if device_type == "cuda" and is_dist:
    if args.sharding_strategy != "NO_SHARD":
        print0("!" * 70)
        print0(f"WARNING: sharding_strategy={args.sharding_strategy} shards parameters along")
        print0("         the first dimension. Muon's Newton-Schulz orthogonalization")
        print0("         requires full matrices — results will be INCORRECT with sharding.")
        print0("         Use --sharding-strategy NO_SHARD for correct Muon behavior.")
        print0("!" * 70)
    model = wrap_fsdp(
        model,
        device_type=device_type,
        is_dist=is_dist,
        sharding_strategy=args.sharding_strategy,
        compute_dtype_override=compute_dtype,
        local_rank=local_rank,
        transformer_layer_cls=LlamaDecoderLayer,
    )
    print0(f"FSDP enabled with sharding strategy: {args.sharding_strategy}")

param_counts = scaling_param_counts(model)
if param_counts:
    print0("Parameter counts:")
    for k, v in param_counts.items():
        print0(f"  {k:<24}: {v:,}")
num_params = sum(p.numel() for p in model.parameters())
print0(f"Total params: {num_params:,}")

scaling_params = (
    param_counts.get("transformer_matrices", 0) + param_counts.get("lm_head", 0) if param_counts else num_params
)
d12_config = make_config(
    12,
    aspect_ratio=args.aspect_ratio,
    head_dim=args.head_dim,
    vocab_size=vocab_size,
    sequence_len=args.max_seq_len,
)
with torch.device("meta"):
    d12_model = LlamaForCausalLM(d12_config)
d12_counts = scaling_param_counts(d12_model)
d12_scaling_params = d12_counts["transformer_matrices"] + d12_counts["lm_head"]
target_tokens = int(args.target_param_data_ratio * scaling_params)
d12_target_tokens = args.target_param_data_ratio * d12_scaling_params
total_batch_size = compute_scaled_total_batch_size(
    scaling_params=scaling_params,
    d12_scaling_params=d12_scaling_params,
    target_param_data_ratio=args.target_param_data_ratio,
    requested_total_batch_size=args.total_batch_size,
)
weight_decay = compute_scaled_weight_decay(
    base_weight_decay=args.weight_decay,
    total_batch_size=total_batch_size,
    target_tokens=target_tokens,
    d12_target_tokens=d12_target_tokens,
)

if args.num_iterations > 0:
    num_iterations = args.num_iterations
elif args.target_param_data_ratio > 0:
    num_iterations = max(1, target_tokens // total_batch_size)
else:
    num_iterations = 1000

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
print0(f"weight_decay: {weight_decay:.6f}")

run_name = runtime_config.run_name if runtime_config.run_name else f"d{args.depth}"
runtime_config = runtime_config.with_run_name(run_name)
checkpoint_dir = get_checkpoint_dir(runtime_config.out_dir, runtime_config.run_name, phase="pretrain")
mlflow_run_name = runtime_config.run if runtime_config.run else runtime_config.run_name


def make_loader(split: str):
    if args.txt:
        return text_data_loader(tokenizer, args.txt, args.device_batch_size, args.max_seq_len, device)
    return tokenizing_distributed_data_loader_bestfit(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        dataset_name=args.dataset,
        split=split,
        device=device,
        text_field=args.text_field,
    )


train_loader = make_loader("train")
eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * world_size))


def eval_fn(eval_model: torch.nn.Module, step: int) -> dict[str, float]:
    """Evaluate bits-per-byte on the validation split."""
    eval_loader = make_loader("val")
    bpb = evaluate_bpb(eval_model, eval_loader, eval_steps, token_bytes)
    print0(f"Step {step:05d} | val bpb: {bpb:.6f}")
    return {"bpb": bpb}


training_args = build_training_arguments(
    output_dir=checkpoint_dir,
    max_steps=num_iterations,
    device_batch_size=args.device_batch_size,
    grad_accum_steps=grad_accum_steps,
    warmup_steps=args.warmup_steps,
    weight_decay=weight_decay,
    grad_clip=args.grad_clip,
    logging_steps=100,
    eval_every=args.eval_every,
    save_steps=args.save_every if args.save_every > 0 else num_iterations,
    run_name=mlflow_run_name,
    report_to=["mlflow"] if master_process else [],
    device_type=device_type,
    compute_dtype=compute_dtype,
    disable_tqdm=not master_process,
)

checkpoint_metadata = build_checkpoint_metadata(
    phase="pretrain",
    args=args,
    runtime_config=runtime_config,
    device_batch_size=args.device_batch_size,
    max_seq_len=args.max_seq_len,
    total_batch_size=total_batch_size,
    grad_accum_steps=grad_accum_steps,
    num_iterations=num_iterations,
)
callbacks = [
    MlflowMetadataCallback(checkpoint_metadata),
]

trainer = TinyGPTTrainer(
    model=model,
    args=training_args,
    callbacks=callbacks,
    eval_dataset=[0] if args.eval_every > 0 else None,
    matrix_lr=args.matrix_lr,
    embedding_lr=args.embedding_lr,
    scalar_lr=args.scalar_lr,
    lm_head_lr=args.lm_head_lr,
    muon_momentum=args.muon_momentum,
    muon_ns_steps=args.muon_ns_steps,
    warmdown_ratio=args.warmdown_ratio,
    final_lr_frac=args.final_lr_frac,
    train_loader=train_loader,
    eval_fn=eval_fn if args.eval_every > 0 else None,
    tokenizer_dir=args.tokenizer_dir,
    checkpoint_metadata=checkpoint_metadata,
)

trainer.train(resume_from_checkpoint=resume_checkpoint)
compute_cleanup()
