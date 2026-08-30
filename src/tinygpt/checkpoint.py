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
from dataclasses import asdict
from typing import Any, cast

import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_NAME

from tinygpt.config import GPTConfig, RuntimeConfig
from tinygpt.model import GPT

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


def _weights_path(model_dir: str) -> str:
    weights_path = os.path.join(model_dir, SAFE_WEIGHTS_NAME)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find {SAFE_WEIGHTS_NAME} in {model_dir}")
    return weights_path


def _sanitize_state_dict_for_save(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    return {key.removeprefix("_orig_mod."): value.detach().cpu().contiguous() for key, value in state_dict.items()}


def _load_state_dict(weights_path: str, device: torch.device) -> dict[str, torch.Tensor]:
    load_device = str(device) if device.type == "cuda" else "cpu"
    state_dict = safe_load_file(weights_path, device=load_device)
    if device.type in {"cpu", "mps"}:
        state_dict = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in state_dict.items()}
    if device.type == "mps":
        state_dict = {k: v.to(device) for k, v in state_dict.items()}
    return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}


def save_model_checkpoint(
    output_dir: str,
    model: torch.nn.Module,
    metadata: dict[str, Any] | None = None,
    tokenizer_dir: str | None = None,
) -> None:
    """Save model weights and config in a Hugging Face style directory."""
    os.makedirs(output_dir, exist_ok=True)
    inner: Any = model.module if hasattr(model, "module") else model
    config_dict: dict[str, Any] = asdict(inner.config) if hasattr(inner, "config") else {}

    config_path = os.path.join(output_dir, CONFIG_NAME)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("Saved config to: %s", config_path)

    weights_path = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
    safe_save_file(_sanitize_state_dict_for_save(model), weights_path, metadata={"format": "pt"})
    logger.info("Saved model weights to: %s", weights_path)

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


def build_model_from_checkpoint(
    model_ref: str,
    device: torch.device,
    phase: str = "eval",
) -> tuple[GPT, dict[str, Any]]:
    """Instantiate a GPT model from a model directory or Trainer output directory."""
    model_dir = resolve_model_directory(model_ref)
    config_dict = _load_json(os.path.join(model_dir, CONFIG_NAME))
    config = GPTConfig(**config_dict)

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(_load_state_dict(_weights_path(model_dir), device), strict=True, assign=True)

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
