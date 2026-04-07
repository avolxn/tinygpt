"""
Checkpoint save/load in nanochat format.

Model weights:     model_{step:06d}.pt    (torch.save state dict)
Metadata:          meta_{step:06d}.json   (model_config + training args)
Optimizer state:   optim_{step:06d}_rank{rank}.pt

FSDP-aware: model state dict and optimizer state gathered to rank 0 before saving.
"""

import glob
import json
import logging
import os
from typing import Any

import torch
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from tinygpt.config import GPTConfig
from tinygpt.model import GPT

logger = logging.getLogger(__name__)


def is_fsdp(model: torch.nn.Module) -> bool:
    """Check if a model is wrapped with FSDP.

    Args:
        model: The PyTorch module to check.

    Returns:
        True if model is an instance of FullyShardedDataParallel.
    """
    return isinstance(model, FSDP)


def fsdp_full_state_dict_ctx(model: torch.nn.Module) -> Any:
    """Return a context manager that gathers a full state dict to rank 0.

    Args:
        model: FSDP-wrapped model.

    Returns:
        A context manager that sets the model's state dict type to FULL_STATE_DICT
        with CPU offload and rank-0-only gathering.
    """
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    return FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg)


def get_checkpoint_dir(output_dir: str, run_name: str) -> str:
    """Return the checkpoint subdirectory for a given run.

    Args:
        output_dir: Root output directory.
        run_name: Name of the training run.

    Returns:
        Path to the checkpoint directory for this run.
    """
    return os.path.join(output_dir, "checkpoints", run_name)


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
    return max(int(os.path.basename(f).split("_")[1].split(".")[0]) for f in checkpoint_files)


def save_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_args: dict[str, Any],
    rank: int = 0,
) -> None:
    """Save checkpoint in nanochat format.

    Saves:
      - model_{step:06d}.pt        — model state dict (rank 0 only)
      - meta_{step:06d}.json       — training_args including model_config (rank 0 only)
      - optim_{step:06d}_rank0.pt  — optimizer state dict (rank 0 only)

    For FSDP models, state dicts are gathered to rank 0 before saving.

    Args:
        checkpoint_dir: Directory where checkpoint files will be written.
        model: FSDP-wrapped or plain GPT instance.
        optimizer: Optimizer instance.
        training_args: Dict with 'step', 'model_config', and other metadata.
        rank: This process's rank.
    """
    step: int = training_args["step"]

    if is_fsdp(model):
        with fsdp_full_state_dict_ctx(model):
            state_dict = model.state_dict()
        optim_state = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
    else:
        state_dict = model.state_dict()
        optim_state = optimizer.state_dict()

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(state_dict, model_path)
        logger.info(f"Saved model to: {model_path}")

        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(training_args, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")

        optim_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank0.pt")
        torch.save(optim_state, optim_path)
        logger.info(f"Saved optimizer to: {optim_path}")


def load_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    rank: int = 0,
    step: int | None = None,
) -> dict[str, Any]:
    """Load model weights and optionally optimizer state into existing objects.

    For FSDP models, the full state dict is loaded on rank 0 and scattered to all ranks.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        model: FSDP-wrapped or plain model to load weights into.
        optimizer: Optimizer to restore state into; None skips optimizer loading.
        device: Target device for tensor loading.
        rank: This process's rank.
        step: Checkpoint step to load; defaults to the latest step found.

    Returns:
        Dict with loaded training_args (step, model_config, etc.).
    """
    if step is None:
        step = find_last_step(checkpoint_dir)

    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    state_dict: dict[str, torch.Tensor] = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    if is_fsdp(model):
        with fsdp_full_state_dict_ctx(model):
            model.load_state_dict(state_dict, strict=True, assign=True)
    else:
        model.load_state_dict(state_dict, strict=True, assign=True)

    if optimizer is not None:
        optim_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank0.pt")
        try:
            if is_fsdp(model):
                full_optim = torch.load(optim_path, map_location="cpu", weights_only=False) if rank == 0 else None
                sharded_optim = FSDP.scatter_full_optim_state_dict(full_optim, model)
                optimizer.load_state_dict(sharded_optim)
            else:
                optim_state = torch.load(optim_path, map_location=device, weights_only=False)
                optimizer.load_state_dict(optim_state)
        except FileNotFoundError:
            logger.warning(f"Optimizer checkpoint not found at {optim_path}, skipping")

    training_args: dict[str, Any] = {}
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            training_args = json.load(f)

    return training_args


def build_model_from_checkpoint(
    checkpoint_dir: str,
    device: torch.device,
    phase: str = "eval",
    step: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Instantiate a GPT model from a saved checkpoint.

    Args:
        checkpoint_dir: Directory containing the checkpoint files.
        device: Device on which to place the model.
        phase: Either 'eval' (model.eval()) or 'train' (model.train()).
        step: Checkpoint step to load; defaults to the latest step found.

    Returns:
        A (model, training_args) tuple with the model placed on device and
        training_args dict loaded from checkpoint metadata.
    """
    if step is None:
        step = find_last_step(checkpoint_dir)

    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No metadata found in {checkpoint_dir} for step {step}")
    with open(meta_path, encoding="utf-8") as f:
        training_args = json.load(f)

    config_kwargs: dict[str, Any] = training_args.get("model_config", {})
    config = GPTConfig(**config_kwargs)

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    load_checkpoint(checkpoint_dir, model, optimizer=None, device=device, step=step)

    if phase == "eval":
        model.eval()
    else:
        model.train()

    return model, training_args
