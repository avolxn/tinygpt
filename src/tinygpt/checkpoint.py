"""Model serialization utilities for tinygpt.

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
import subprocess
import sys
from typing import Any, cast

import torch
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from tinygpt.config import RuntimeConfig
from tinygpt.model import TinyGPTForCausalLM
from tinygpt.utils import compute_dtype

logger = logging.getLogger(__name__)
TRAINER_STATE_NAME = "trainer_state.json"
METADATA_NAME = "tinygpt_metadata.json"


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
        "teacher_source": _load_optional_json(os.path.join(getattr(args, "teacher_model", ""), "teacher_source.json")),
    }


def _has_model_files(model_dir: str) -> bool:
    config_path = os.path.join(model_dir, CONFIG_NAME)
    return os.path.exists(config_path) and any(
        os.path.exists(os.path.join(model_dir, name)) for name in (SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME)
    )


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
    has_optimizer = any(
        os.path.exists(os.path.join(model_dir, filename)) for filename in ("optimizer.pt", "optimizer.bin")
    )
    required_files = (TRAINER_STATE_NAME, "scheduler.pt")
    if has_optimizer and all(os.path.exists(os.path.join(model_dir, filename)) for filename in required_files):
        return model_dir
    return None


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def _load_optional_json(path: str) -> dict[str, Any]:
    return _load_json(path) if os.path.exists(path) else {}


def save_checkpoint_assets(
    output_dir: str,
    metadata: dict[str, Any] | None = None,
    tokenizer_dir: str | None = None,
) -> None:
    """Write project metadata alongside a checkpoint saved by Transformers."""
    if metadata is not None:
        metadata_path = os.path.join(output_dir, METADATA_NAME)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata to: %s", metadata_path)

    if tokenizer_dir is not None:
        AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True).save_pretrained(output_dir)


def build_model_from_checkpoint(
    model_ref: str,
    device: torch.device,
    phase: str = "eval",
) -> tuple[TinyGPTForCausalLM, dict[str, Any]]:
    """Load the nanochat student from a directory or Trainer checkpoint."""
    model_dir = resolve_model_directory(model_ref)
    config_dict = _load_json(os.path.join(model_dir, CONFIG_NAME))
    backend = "flash_attention_3" if device.type == "cuda" and compute_dtype == torch.bfloat16 else "sdpa"
    model: Any = TinyGPTForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation=backend)
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
    return cast(TinyGPTForCausalLM, model), metadata
