"""
Evaluate a trained tinygpt model.

Supported modes (comma-separated via --eval):
  bpb     : bits-per-byte on train/val splits of a text dataset
  sample  : unconditional text samples from the model
  chat    : task accuracy on chat benchmarks (categorical + generative)

Usage:
    python -m scripts.evaluate_model --eval sample --vllm-model data/distill_checkpoints/student
    python -m scripts.evaluate_model --eval bpb --checkpoint data/distill_checkpoints/student
    python -m scripts.evaluate_model --eval chat --vllm-model data/distill_checkpoints/student --tasks MMLU
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

from tinygpt.checkpoint import build_model_from_checkpoint, resolve_model_directory
from tinygpt.dataloader import CLIMBMIX_DATASET, streaming_data_loader
from tinygpt.distributed import compute_cleanup, compute_init, get_dist_info, print0
from tinygpt.inference import VLLM
from tinygpt.metrics import compute_token_bytes, evaluate_bpb
from tinygpt.tokenizer import HuggingFaceTokenizer
from tinygpt.utils import autodetect_device_type

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tinygpt model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to a model directory or Trainer output directory",
    )
    parser.add_argument("--eval", type=str, default="bpb,sample", help="Comma-separated modes: bpb,sample,chat")
    parser.add_argument("--tasks", type=str, default="", help="Tasks for chat eval, pipe-separated. Default = all.")
    parser.add_argument("--dataset", type=str, default=CLIMBMIX_DATASET)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--split-tokens", type=int, default=40 * 524288)
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per problem for generative eval")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-problems", type=int, default=None, help="Cap number of problems per task")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument(
        "--vllm-model", type=str, default="", help="Prepared model directory or Trainer checkpoint for sample/chat eval"
    )
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")


    args = parser.parse_args()

    eval_modes = {m.strip() for m in args.eval.split(",")}
    needs_local_model = "bpb" in eval_modes
    needs_vllm = bool(eval_modes & {"sample", "chat"})
    if needs_vllm and not args.vllm_model:
        parser.error("--vllm-model is required for sample/chat evaluation")
    if needs_local_model and not args.checkpoint:
        parser.error("--checkpoint is required for bpb evaluation")
    tokenizer_ref = args.checkpoint or args.vllm_model
    if not tokenizer_ref:
        parser.error("--checkpoint or --vllm-model is required to load the tokenizer")
    tokenizer_dir = resolve_model_directory(tokenizer_ref)
    vllm_model_dir = resolve_model_directory(args.vllm_model) if needs_vllm else ""

    tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)
    vllm = (
        VLLM(
            vllm_model_dir,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
        )
        if needs_vllm
        else None
    )

    if needs_local_model:
        device_type = autodetect_device_type() if args.device_type == "" else args.device_type
        init_info = compute_init(device_type)
        rank = init_info[1]
        world_size = init_info[3]
        device = init_info[4]
        dist_info = get_dist_info()
        is_dist = dist_info[0]
        ddp_rank = dist_info[1]
        ddp_world_size = dist_info[3]
        model, meta = build_model_from_checkpoint(args.checkpoint, device, phase="eval")
        token_bytes = compute_token_bytes(tokenizer, device=device)
        sequence_len = meta["model_config"]["max_position_embeddings"]
    else:
        rank, world_size, device = 0, 1, torch.device("cpu")
        is_dist, ddp_rank, ddp_world_size = False, 0, 1

    if needs_local_model:
        print0(f"Loaded model from {args.checkpoint} (step {meta.get('step', '?')})")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # -----------------------------------------------------------------------------
    # Sampling

    if "sample" in eval_modes and rank == 0:
        assert vllm is not None
        print0("\n" + "=" * 70)
        print0("Samples")
        print0("=" * 70)
        for prompt in [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
        ]:
            tokens = tokenizer.encode(prompt, prepend="<|bos|>")
            sample = vllm.generate(
                tokenizer.decode(tokens),
                max_tokens=20,
                temperature=0,
                top_k=args.top_k,
                stop=["<|assistant_end|>", "<|bos|>"],
            )
            print0(sample)

    # -----------------------------------------------------------------------------
    # BPB

    if "bpb" in eval_modes:
        print0("\n" + "=" * 70)
        print0("BPB Evaluation")
        print0("=" * 70)
        tokens_per_step = args.device_batch_size * sequence_len * world_size
        steps = max(1, args.split_tokens // tokens_per_step)
        for split in ("train", "val"):
            loader = streaming_data_loader(
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
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_passed, total = 0, 0
    assert vllm is not None

    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)
        completions = vllm.generate_batch(
            [tokenizer.decode(encoded_prompt)] * num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop=["<|assistant_end|>", "<|bos|>"],
        )
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
    assert vllm is not None
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_batches = -(-num_problems // batch_size)  # ceil_div
    num_passed, total = 0, 0

    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conv) for conv in conversations]
        completions = vllm.generate_batch(
            [tokenizer.decode(ids) for ids in prompt_ids],
            max_tokens=1,
            temperature=0.0,
        )

        for conversation, completion in zip(conversations, completions, strict=True):
            letters: list[str] = conversation["letters"]
            normalized = completion.strip()
            predicted = next((letter for letter in letters if normalized.startswith(letter)), normalized[:1])
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


if __name__ == "__main__":
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
