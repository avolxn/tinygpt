"""
Reinforcement learning on GSM8K via policy gradient.

Simplified GRPO-like approach:
1. No reference model or KL regularization
2. On-policy sampling (no PPO ratio/clip)
3. Token-level DAPO-style advantage normalization (r - mu)
4. Simple linear learning rate schedule

Single GPU:
    python -m scripts.rl

Multi-GPU (e.g. 8 GPUs):
    torchrun --nproc_per_node=8 -m scripts.rl
"""

import argparse
import itertools
import json
import os
from dataclasses import asdict
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
import wandb
from tasks.gsm8k import GSM8K

from tinygpt.checkpoint import get_checkpoint_dir, load_checkpoint, save_checkpoint
from tinygpt.config import GPTConfig, make_config
from tinygpt.engine import Engine
from tinygpt.model import GPT, Block
from tinygpt.optimizer import make_optimizer
from tinygpt.tokenizer import HuggingFaceTokenizer
from tinygpt.utils import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_dtype,
    compute_init,
    get_peak_flops,
    make_fsdp_mixed_precision,
    print0,
)

parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K")
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
# Model loading
parser.add_argument("--model-dir", type=str, default="out/checkpoints/d20", help="Model checkpoint directory")
# Data
parser.add_argument("--tokenizer-dir", type=str, default="out/tokenizer")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs over GSM8K training set")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="Max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="Total examples per optimization step")
parser.add_argument("--num-samples", type=int, default=16, help="Number of samples per example")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--matrix-lr", type=float, default=0.0005, help="Learning rate for matrix parameters")
parser.add_argument("--embedding-lr", type=float, default=0.005, help="Learning rate for embedding parameters")
parser.add_argument("--scalar-lr", type=float, default=0.05, help="Learning rate for scalar parameters")
parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
parser.add_argument("--init-lr-frac", type=float, default=0.1, help="Initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="Evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="Number of examples for pass@k evaluation")
parser.add_argument("--save-every", type=int, default=60, help="Save checkpoint every N steps")
# Output
parser.add_argument("--out-dir", type=str, default="out")
parser.add_argument("--run-name", type=str, default="rl", help="Subdirectory name under out/checkpoints/")
args = parser.parse_args()
user_config = vars(args).copy()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
is_dist, rank, local_rank, world_size, device = compute_init(device_type)
master_process = rank == 0


def _noop() -> None:
    pass


synchronize: Any = torch.cuda.synchronize if device_type == "cuda" else _noop

if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_name)
    print0(f"GPU: {gpu_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="tinygpt-rl", name=args.run, config=user_config)

# Load tokenizer
print0(f"Loading tokenizer from {args.tokenizer_dir}")
tokenizer = HuggingFaceTokenizer.from_directory(args.tokenizer_dir)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Load model from checkpoint
print0(f"Loading model from {args.model_dir}")
meta = {}
if os.path.exists(args.model_dir):
    meta_path = os.path.join(args.model_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

model_config_kwargs = meta.get("model_config", {})
if not model_config_kwargs:
    # Fallback to default config
    config = make_config(20, vocab_size=vocab_size)
    model_config_kwargs = asdict(config)
else:
    config = GPTConfig(**model_config_kwargs)

print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")

with torch.device("meta"):
    model = GPT(GPTConfig(**model_config_kwargs))
model.to_empty(device=device)
model.init_weights()

# Load checkpoint weights
if os.path.exists(args.model_dir):
    try:
        load_checkpoint(args.model_dir, model, optimizer=None, device=device, rank=rank)
        print0(f"✅ Loaded model from {args.model_dir}")
    except FileNotFoundError:
        print0(f"⚠️  Checkpoint not found in {args.model_dir}, starting from init weights")

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

# Setup
engine = Engine(model, tokenizer)
train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")

# Optimizer
optimizer = make_optimizer(
    model,
    matrix_lr=args.matrix_lr,
    embedding_lr=args.embedding_lr,
    scalar_lr=args.scalar_lr,
    weight_decay=args.weight_decay,
)

# Set initial LR as fraction of base LR
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]


@torch.no_grad()
def get_batch() -> Any:
    """Generator that yields rollout batches for policy gradient training."""
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = range(rank, len(train_task), world_size)

    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples
        model.eval()
        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed,
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate rewards
        rewards_list: list[float] = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text)
            rewards_list.append(reward)

        # Pad sequences to same length
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [
            seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences
        ]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]

        # Convert to tensors
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)

        # Generate inputs/targets
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1

        rewards = torch.tensor(rewards_list, dtype=torch.float, device=device)

        # Calculate advantages: rewards - mean_reward
        mu = rewards.mean()
        advantages = rewards - mu

        yield generated_token_sequences, inputs, targets, rewards, advantages


def run_gsm8k_eval(
    task: GSM8K,
    max_examples: int | None = None,
    num_samples: int = 1,
    max_completion_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 50,
) -> Any:
    """Evaluate pass@k on GSM8K."""
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(rank, max_examples, world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        assert num_samples <= args.device_batch_size
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})

        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record


# Training loop
assert args.examples_per_step % world_size == 0
examples_per_rank = args.examples_per_step // world_size

batch_iterator = get_batch()
for step in range(num_steps):
    # Evaluation
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device)
        records_iter = run_gsm8k_eval(
            val_task,
            num_samples=args.device_batch_size,
            max_examples=args.eval_examples,
            temperature=1.0,
        )
        records = list(records_iter)
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if is_dist:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item()
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
        wandb_run.log({"step": step, **log_passk})

    # Training on rollouts
    rewards_list: list[float] = []
    sequence_lengths: list[int] = []
    for example_step in range(examples_per_rank):
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)

        model.train()
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size

        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]

            # Calculate log probabilities
            logp = -model(inputs, targets, loss_reduction="none").view_as(inputs)

            # Policy gradient objective
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()

            # Normalize by number of valid tokens and number of passes/examples
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)

            # Loss to minimize
            loss = -pg_obj
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            print0(
                f"Step {step}/{num_steps} | Example {example_step} | "
                f"Pass {pass_idx} | loss: {loss.item():.6f} | "
                f"Avg reward: {rewards.mean().item():.4f}"
            )

        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # Logging and step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)

    if is_dist:
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()

    print0(f"Step {step}/{num_steps} | Avg reward: {mean_reward:.4f} | Avg seq len: {mean_sequence_length:.2f}")
    wandb_run.log({"step": step, "reward": mean_reward, "sequence_length": mean_sequence_length})

    # Update parameters with learning rate schedule
    lrm = 1.0 - step / num_steps  # Linear rampdown
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({"step": step, "lrm": lrm})

    # Save checkpoint
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        checkpoint_dir = get_checkpoint_dir(args.out_dir, args.run_name)
        model_config_kwargs = asdict(model.module.config) if hasattr(model, "module") else asdict(model.config)
        save_checkpoint(
            checkpoint_dir,
            model,
            optimizer,
            {
                "step": step,
                "model_config": model_config_kwargs,
            },
            rank=rank,
        )
        print0(f"✅ Saved checkpoint to {checkpoint_dir}")

wandb_run.finish()
compute_cleanup()
