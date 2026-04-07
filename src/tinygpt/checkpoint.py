"""
Checkpoint save/load utilities (mirrors nanochat/checkpoint_manager.py).

Model weights:   model_{step:06d}.pt   (torch.save state dict, rank 0)
Metadata:        meta_{step:06d}.json  (model_config + training args, rank 0)
Optimizer state: optim_{step:06d}_rank{rank}.pt  (every rank saves its own)
"""

import glob
import json
import logging
import os
import re
from typing import Any

import torch

from tinygpt.config import GPTConfig
from tinygpt.model import GPT

logger = logging.getLogger(__name__)

_CKPT_RE = re.compile(r"model_(\d+)\.pt$")


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model_data: dict[str, Any],
    optimizer_data: dict[str, Any] | None,
    meta_data: dict[str, Any],
    rank: int = 0,
) -> None:
    """Save checkpoint in nanochat format.

    Args:
        checkpoint_dir: Directory where checkpoint files will be written.
        step: Current training step; used to name the files.
        model_data: Model state dict (saved by rank 0 only).
        optimizer_data: Optimizer state dict for this rank; None skips saving.
        meta_data: Metadata dict (model_config, training args, etc.).
        rank: This process's rank.
    """
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model to: {model_path}")
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer to: {optimizer_path}")


def load_checkpoint(
    checkpoint_dir: str,
    step: int,
    device: torch.device,
    load_optimizer: bool = False,
    rank: int = 0,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """Load model weights and optionally optimizer state from a checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        step: Checkpoint step to load.
        device: Target device for tensor loading.
        load_optimizer: Whether to load the optimizer state for this rank.
        rank: This process's rank (selects the per-rank optimizer shard file).

    Returns:
        A (model_data, optimizer_data, meta_data) tuple.
    """
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device, weights_only=True)
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def find_last_step(checkpoint_dir: str) -> int:
    """Find the highest step number among saved model checkpoints.

    Args:
        checkpoint_dir: Directory containing model_*.pt files.

    Returns:
        The highest step number found.

    Raises:
        FileNotFoundError: If no checkpoints are found.
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    steps = [int(m.group(1)) for f in checkpoint_files if (m := _CKPT_RE.search(os.path.basename(f)))]
    if not steps:
        raise FileNotFoundError(f"No valid checkpoints found in {checkpoint_dir}")
    return max(steps)


def build_model_from_checkpoint(
    checkpoint_dir: str,
    device: torch.device,
    phase: str = "eval",
    step: int | None = None,
) -> tuple[GPT, dict[str, Any]]:
    """Instantiate a GPT model from a saved checkpoint.

    Args:
        checkpoint_dir: Directory containing the checkpoint files.
        device: Device on which to place the model.
        phase: Either 'eval' (model.eval()) or 'train' (model.train()).
        step: Checkpoint step to load; defaults to the latest step found.

    Returns:
        A (model, meta_data) tuple.

    Raises:
        FileNotFoundError: If no checkpoints are found in checkpoint_dir.
    """
    if step is None:
        step = find_last_step(checkpoint_dir)
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    config = GPTConfig(**meta_data["model_config"])
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    if phase == "eval":
        model.eval()
    else:
        model.train()
    return model, meta_data
