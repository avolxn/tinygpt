"""
Evaluate a trained tinygpt model.

Supported modes (comma-separated via --eval):
  bpb     : bits-per-byte on train/val splits of a text dataset
  sample  : unconditional text samples from the model
  chat    : task accuracy on chat benchmarks (categorical + generative)

Usage:
    python -m scripts.evaluate_model --checkpoint data/pretrain_checkpoints/from_scratch
    python -m scripts.evaluate_model --checkpoint data/pretrain_checkpoints/from_scratch --eval chat --tasks MMLU
    torchrun --nproc_per_node=4 -m scripts.evaluate_model --checkpoint data/pretrain_checkpoints/from_scratch --eval chat
"""

import argparse
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU

from tinygpt.checkpoint import build_model_from_checkpoint
from tinygpt.dataloader import tokenizing_distributed_data_loader_bestfit
from tinygpt.distributed import compute_cleanup, compute_init, get_dist_info, print0
from tinygpt.inference import Engine
from tinygpt.metrics import compute_token_bytes, evaluate_bpb
from tinygpt.tokenizer import HuggingFaceTokenizer
from tinygpt.utils import autodetect_device_type

parser = argparse.ArgumentParser(description="Evaluate tinygpt model")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to a model directory or Trainer output directory",
)
parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizer")
parser.add_argument("--eval", type=str, default="bpb,sample", help="Comma-separated modes: bpb,sample,chat")
parser.add_argument("--tasks", type=str, default="", help="Tasks for chat eval, pipe-separated. Default = all.")
parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb")
parser.add_argument("--text-field", type=str, default="text")
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--split-tokens", type=int, default=40 * 524288)
parser.add_argument("--num-samples", type=int, default=1, help="Samples per problem for generative eval")
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--max-problems", type=int, default=None, help="Cap number of problems per task")
parser.add_argument("--device-type", type=str, default="")
args = parser.parse_args()

eval_modes = {m.strip() for m in args.eval.split(",")}

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
_, rank, _, world_size, device = compute_init(device_type)
is_dist, ddp_rank, _, ddp_world_size = get_dist_info()

model, meta = build_model_from_checkpoint(args.checkpoint, device, phase="eval")
tokenizer = HuggingFaceTokenizer.from_directory(args.tokenizer_dir)
token_bytes = compute_token_bytes(tokenizer, device=device)
sequence_len = meta["model_config"]["sequence_len"]

print0(f"Loaded model from {args.checkpoint} (step {meta.get('step', '?')})")
print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

# -----------------------------------------------------------------------------
# Sampling

if "sample" in eval_modes and rank == 0:
    print0("\n" + "=" * 70)
    print0("Samples")
    print0("=" * 70)
    engine = Engine(model, tokenizer)
    for prompt in [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
    ]:
        tokens = tokenizer(prompt, prepend="<|bos|>")
        sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=20, temperature=0)
        print0(tokenizer.decode(sample[0]))

# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Chat eval (categorical + generative)


def run_generative_eval(
    task_object: Any,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    max_problems: int | None = None,
) -> float:
    """Evaluate a generative task by sampling completions and checking any-pass.

    Args:
        task_object: Task instance with eval_type == 'generative'.
        num_samples: Number of completions to sample per problem.
        max_new_tokens: Maximum tokens to generate per sample.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        max_problems: Cap on problems to evaluate; None means all.

    Returns:
        Pass rate (fraction of problems where any sample is correct).
    """
    engine = Engine(model, tokenizer)
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_passed, total = 0, 0

    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        prefix_len = len(encoded_prompt)
        completions = [tokenizer.decode(r[prefix_len:]) for r in results]
        passed = any(task_object.evaluate(conversation, c) for c in completions)
        total += 1
        num_passed += int(passed)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100 * num_passed / total:.2f}%)", end="", flush=True)

    print()

    if is_dist:
        passed_t = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_t = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(passed_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        num_passed = int(passed_t.item())
        total = int(total_t.item())

    acc = num_passed / total
    print0(f"Final: {num_passed}/{total} ({100 * acc:.2f}%)")
    return acc


def run_categorical_eval(task_object: Any, batch_size: int, max_problems: int | None = None) -> float:
    """Evaluate a categorical task using argmax over answer-letter logits.

    Args:
        task_object: Task instance with eval_type == 'categorical'.
        batch_size: Number of problems to evaluate in parallel.
        max_problems: Cap on problems to evaluate; None means all.

    Returns:
        Accuracy (fraction of problems answered correctly).
    """
    bos = tokenizer.get_bos_token_id()
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_batches = -(-num_problems // batch_size)  # ceil_div
    letter_cache: dict[str, int] = {}
    num_passed, total = 0, 0

    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conv) for conv in conversations]
        max_len = max(len(ids) for ids in prompt_ids)
        answer_positions = [len(ids) - 1 for ids in prompt_ids]
        padded = [ids + [bos] * (max_len - len(ids)) for ids in prompt_ids]
        input_t = torch.tensor(padded, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_t)  # (B, T, V)

        for idx, conversation in enumerate(conversations):
            letters: list[str] = conversation["letters"]
            letter_ids: list[int] = []
            for letter in letters:
                if letter not in letter_cache:
                    encoded = tokenizer.encode(letter)
                    assert len(encoded) == 1, f"Letter {letter!r} must be a single token"
                    letter_cache[letter] = encoded[0]
                letter_ids.append(letter_cache[letter])
            focus = logits[idx, answer_positions[idx], letter_ids]
            predicted = letters[int(focus.argmax(dim=-1).item())]
            num_passed += int(task_object.evaluate(conversation, predicted))
            total += 1

    if is_dist:
        passed_t = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_t = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(passed_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        num_passed = int(passed_t.item())
        total = int(total_t.item())

    acc = num_passed / total
    print0(f"Final: {num_passed}/{total} ({100 * acc:.2f}%)")
    return acc


if "chat" in eval_modes:
    ALL_TASKS: dict[str, Any] = {
        "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
        "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
        "MMLU": partial(MMLU, subset="all", split="test"),
        "GSM8K": partial(GSM8K, subset="main", split="test"),
        "HumanEval": HumanEval,
    }
    BASELINES: dict[str, float] = {
        "ARC-Easy": 0.25,
        "ARC-Challenge": 0.25,
        "MMLU": 0.25,
        "GSM8K": 0.0,
        "HumanEval": 0.0,
    }

    task_names = list(ALL_TASKS.keys()) if not args.tasks else args.tasks.split("|")

    print0("\n" + "=" * 70)
    print0("Chat Evaluation")
    print0("=" * 70)

    results: dict[str, float] = {}
    for task_name in task_names:
        print0(f"\n--- {task_name} ---")
        task_obj = ALL_TASKS[task_name]()
        if task_obj.eval_type == "generative":
            acc = run_generative_eval(
                task_obj,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_problems=args.max_problems,
            )
        else:
            acc = run_categorical_eval(task_obj, batch_size=args.device_batch_size, max_problems=args.max_problems)
        results[task_name] = acc
        print0(f"{task_name}: {100 * acc:.2f}%")

    # ChatCORE metric (mean-centered accuracy over baseline chance levels)
    if all(t in results for t in ALL_TASKS):
        centered = [(results[t] - BASELINES[t]) / (1.0 - BASELINES[t]) for t in ALL_TASKS]
        chatcore = sum(centered) / len(centered)
        print0(f"\nChatCORE: {100 * chatcore:.2f}%")

compute_cleanup()
