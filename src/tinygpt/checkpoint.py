"""
Model serialization utilities for tinygpt.

Runtime model directories use a Hugging Face style layout:
- `config.json`
- `model.safetensors`

Trainer outputs may also contain `checkpoint-*` subdirectories with the same
layout plus optimizer/scheduler state managed by `transformers.Trainer`.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, cast

import torch
from transformers import LlamaForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_NAME

from tinygpt.config import RuntimeConfig
from tinygpt.tokenizer import SPECIAL_TOKENS

logger = logging.getLogger(__name__)
TRAINER_STATE_NAME = "trainer_state.json"
METADATA_NAME = "tinygpt_metadata.json"
TOKENIZER_FILES = ("tokenizer.json", "token_bytes.pt")


def get_checkpoint_dir(out_dir: str, run_name: str, phase: str = "pretrain") -> str:
    """Return the Trainer output directory for a named run."""
    return os.path.join(out_dir, f"{phase}_checkpoints", run_name)


def build_checkpoint_metadata(
    *,
    phase: str,
    args: Any,
    runtime_config: RuntimeConfig,
    **derived_values: Any,
) -> dict[str, Any]:
    """Build reproducibility metadata shared by all training phases."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        git_revision = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = "unknown"

    return {
        "phase": phase,
        "user_config": vars(args).copy(),
        "runtime_config": runtime_config.as_dict(),
        "derived_values": derived_values,
        "git_revision": git_revision,
        "command": sys.argv.copy(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }


def _has_model_files(model_dir: str) -> bool:
    config_path = os.path.join(model_dir, CONFIG_NAME)
    return os.path.exists(config_path) and os.path.exists(os.path.join(model_dir, SAFE_WEIGHTS_NAME))


def resolve_model_directory(model_ref: str) -> str:
    """Resolve a local model directory or Trainer output directory."""
    if _has_model_files(model_ref):
        return model_ref
    last_checkpoint = get_last_checkpoint(model_ref)  # type: ignore[no-untyped-call]
    if last_checkpoint is not None:
        return cast(str, last_checkpoint)
    raise FileNotFoundError(
        f"Could not find {CONFIG_NAME} and {SAFE_WEIGHTS_NAME}, or a checkpoint-* directory, in {model_ref}"
    )


def resolve_trainer_checkpoint(model_ref: str) -> str | None:
    """Return a full Trainer checkpoint path when optimizer state is present."""
    model_dir = resolve_model_directory(model_ref)
    required_files = (TRAINER_STATE_NAME, "optimizer.pt", "scheduler.pt")
    if all(os.path.exists(os.path.join(model_dir, filename)) for filename in required_files):
        return model_dir
    return None


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def _load_optional_json(path: str) -> dict[str, Any]:
    return _load_json(path) if os.path.exists(path) else {}


def _sanitize_state_dict_for_save(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    return {key.removeprefix("_orig_mod."): value.detach().cpu().contiguous() for key, value in state_dict.items()}


def save_model_checkpoint(
    output_dir: str,
    model: torch.nn.Module,
    metadata: dict[str, Any] | None = None,
    tokenizer_dir: str | None = None,
) -> None:
    """Save a model with the standard Transformers serialization API."""
    os.makedirs(output_dir, exist_ok=True)
    inner: Any = model.module if hasattr(model, "module") else model
    inner.save_pretrained(
        output_dir,
        state_dict=_sanitize_state_dict_for_save(model),
        safe_serialization=True,
    )
    logger.info("Saved Transformers model to: %s", output_dir)

    if metadata is not None:
        metadata_path = os.path.join(output_dir, METADATA_NAME)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata to: %s", metadata_path)

    if tokenizer_dir is not None:
        for filename in TOKENIZER_FILES:
            source = os.path.join(tokenizer_dir, filename)
            if os.path.exists(source):
                shutil.copy2(source, os.path.join(output_dir, filename))
        tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "bos_token": SPECIAL_TOKENS[0],
                    "additional_special_tokens": SPECIAL_TOKENS[1:],
                },
                f,
                indent=2,
            )


def build_model_from_checkpoint(
    model_ref: str,
    device: torch.device,
    phase: str = "eval",
) -> tuple[LlamaForCausalLM, dict[str, Any]]:
    """Load a native Llama model from a directory or Trainer output."""
    model_dir = resolve_model_directory(model_ref)
    config_dict = _load_json(os.path.join(model_dir, CONFIG_NAME))
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.to(device)

    if phase == "eval":
        model.eval()
    else:
        model.train()

    metadata = _load_optional_json(os.path.join(model_dir, METADATA_NAME))
    metadata.update(_load_optional_json(os.path.join(model_dir, TRAINER_STATE_NAME)))
    if "global_step" in metadata and "step" not in metadata:
        metadata["step"] = metadata["global_step"]
    metadata["model_config"] = config_dict
    metadata["model_dir"] = model_dir
    return model, metadata
