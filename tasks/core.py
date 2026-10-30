"""DCLM CORE evaluation with nanochat's prompts, bundle, and normalization.

Reference: karpathy/nanochat core_eval.py and scripts/base_eval.py. Multiple-choice
and schema tasks compare mean continuation losses; language-modeling tasks
require every continuation token to match its greedy prediction.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import torch
import torch.distributed as dist
import yaml  # type: ignore[import-untyped]
from filelock import FileLock
from jinja2 import Template

from tinygpt.distributed import get_dist_info, print0
from tinygpt.tokenizer import HuggingFaceTokenizer

BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def prepare_bundle(directory: str = "data") -> Path:
    """Download the reference evaluation bundle once, sharing it across ranks."""
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bundle = root / "eval_bundle"
    with FileLock(str(root / "eval_bundle.lock")):
        if not (bundle / "core.yaml").exists():
            archive = root / "eval_bundle.zip"
            torch.hub.download_url_to_file(BUNDLE_URL, str(archive))
            with ZipFile(archive) as zipped:
                if any(not (root / name).resolve().is_relative_to(root) for name in zipped.namelist()):
                    raise ValueError("Evaluation archive contains an invalid path")
                zipped.extractall(root)
    return bundle


def render_prompts(item: dict[str, Any], task: dict[str, Any], examples: list[dict[str, Any]]) -> list[str]:
    """Render nanochat's exact few-shot templates for one CORE example."""
    kind = task["icl_task_type"]
    delimiter = task.get("continuation_delimiter", " ")
    if kind == "multiple_choice":
        template = Template(
            "{%- for example in examples -%}{{ example.query }}{{ delimiter }}{{ example.choices[example.gold] }}\n\n{% endfor -%}{{ item.query }}{{ delimiter }}{{ choice }}"
        )
        return [template.render(examples=examples, delimiter=delimiter, item=item, choice=c) for c in item["choices"]]
    if kind == "schema":
        template = Template(
            "{%- for example in examples -%}{{ example.context_options[example.gold] }}{{ delimiter }}{{ example.continuation }}\n\n{% endfor -%}{{ context }}{{ delimiter }}{{ item.continuation }}"
        )
        return [
            template.render(examples=examples, delimiter=delimiter, item=item, context=c)
            for c in item["context_options"]
        ]
    if kind == "language_modeling":
        template = Template(
            "{%- for example in examples -%}{{ example.context | trim }}{{ delimiter }}{{ example.continuation }}\n\n{% endfor -%}{{ item.context | trim }}{{ delimiter }}{% if include %}{{ item.continuation }}{% endif %}"
        )
        return [
            template.render(examples=examples, delimiter=delimiter, item=item, include=include).strip()
            for include in (False, True)
        ]
    raise ValueError(f"Unknown CORE task type: {kind}")


def tokenize_prompts(
    prompts: list[str], kind: str, tokenizer: HuggingFaceTokenizer
) -> tuple[list[list[int]], list[int]]:
    """Locate continuation boundaries using the reference common-prefix/suffix rules."""
    tokens = tokenizer.encode(prompts, prepend=tokenizer.get_bos_token_id())
    if kind == "language_modeling":
        prefix, complete = tokens
        if prefix != complete[: len(prefix)] or len(prefix) >= len(complete):
            raise ValueError("CORE continuation is not a nonempty token-aligned suffix")
        return [complete], [len(prefix)]
    shortest = min(map(len, tokens))
    indices = range(shortest) if kind == "multiple_choice" else range(-1, -shortest - 1, -1)
    common = 0
    for position in indices:
        if any(sequence[position] != tokens[0][position] for sequence in tokens):
            break
        common += 1
    starts = [common] * len(tokens) if kind == "multiple_choice" else [len(sequence) - common for sequence in tokens]
    return tokens, starts


@torch.no_grad()
def evaluate_core(
    model: Any,
    tokenizer: HuggingFaceTokenizer,
    device: torch.device,
    *,
    max_per_task: int = -1,
    directory: str = "data",
) -> dict[str, float]:
    """Evaluate the reference CORE task suite, including mean-centered aggregate.

    Args:
        model: Student or FSDP-wrapped student returning Transformers logits.
        tokenizer: The student's unchanged teacher tokenizer.
        device: Model input device.
        max_per_task: Deterministic cap per task; negative evaluates every example.
        directory: Cache location for the original evaluation bundle.

    Returns:
        Per-task accuracy and the mean-centered ``core`` aggregate.
    """
    bundle = prepare_bundle(directory)
    with (bundle / "core.yaml").open() as stream:
        tasks = yaml.safe_load(stream)["icl_tasks"]
    with (bundle / "eval_meta_data.csv").open() as stream:
        baselines = {row["Eval Task"]: float(row["Random baseline"]) / 100 for row in csv.DictReader(stream)}
    _, rank, _, world = get_dist_info()
    results: dict[str, float] = {}
    centered = []
    max_length = model.config.max_position_embeddings
    for task in tasks:
        with (bundle / "eval_data" / task["dataset_uri"]).open() as stream:
            data = [json.loads(line) for line in stream]
        random.Random(1337).shuffle(data)
        count = min(len(data), max_per_task) if max_per_task > 0 else len(data)
        correct = torch.zeros((), dtype=torch.float32, device=device)
        # Equal forward counts keep FSDP collectives aligned for uneven task lengths.
        for batch in range((count + world - 1) // world):
            index = batch * world + rank
            real = index < count
            index = index if real else 0
            rng = random.Random(1234 + index)
            pool = [i for i in range(len(data)) if i != index]
            examples = [data[i] for i in rng.sample(pool, task["num_fewshot"][0])]
            prompts = render_prompts(data[index], task, examples)
            sequences, starts = tokenize_prompts(prompts, task["icl_task_type"], tokenizer)
            cropped = [max(0, len(sequence) - max_length) for sequence in sequences]
            starts = [start - crop for start, crop in zip(starts, cropped, strict=True)]
            sequences = [sequence[crop:] for sequence, crop in zip(sequences, cropped, strict=True)]
            if any(start < 1 or start >= len(sequence) for start, sequence in zip(starts, sequences, strict=True)):
                raise ValueError("CORE continuation does not fit the model context")
            inputs = torch.full(
                (len(sequences), max(map(len, sequences))),
                tokenizer.get_bos_token_id(),
                device=device,
                dtype=torch.long,
            )
            for i, sequence in enumerate(sequences):
                inputs[i, : len(sequence)] = torch.tensor(sequence, device=device)
            logits = model(input_ids=inputs).logits
            if task["icl_task_type"] == "language_modeling":
                end = len(sequences[0])
                passed = bool(torch.equal(logits[0, starts[0] - 1 : end - 1].argmax(-1), inputs[0, starts[0] : end]))
            else:
                losses = [
                    torch.nn.functional.cross_entropy(
                        logits[i, start - 1 : len(sequence) - 1], inputs[i, start : len(sequence)]
                    ).item()
                    for i, (start, sequence) in enumerate(zip(starts, sequences, strict=True))
                ]
                passed = losses.index(min(losses)) == data[index]["gold"]
            correct += int(real and passed)
        if world > 1:
            dist.all_reduce(correct)
        accuracy = correct.item() / count
        label = task["label"]
        results[label] = accuracy
        centered.append((accuracy - baselines[label]) / (1 - baselines[label]))
        print0(f"{label}: {accuracy:.4f}")
    results["core"] = sum(centered) / len(centered)
    return results
