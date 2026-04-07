"""
Checkpoint save/load in HuggingFace format.

Model weights: model.safetensors (via safetensors library, no pickle).
Config: config.json (model architecture).
Training metadata: training_args.json (step, learning rate, etc).
Optimizer state: optimizer_state.pt (via torch.save, isolated from weights).

FSDP-aware: state dicts gathered to rank 0 before saving.
"""

import json
import logging
import os
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from tinygpt.config import GPTConfig
from tinygpt.model import GPT

logger = logging.getLogger(__name__)

model_filename = "model.safetensors"
config_filename = "config.json"
optim_filename = "optimizer_state.pt"
training_args_filename = "training_args.json"


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


def save_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_args: dict[str, Any],
    rank: int = 0,
) -> None:
    """Save checkpoint in HuggingFace format.

    For FSDP models, state dicts are gathered to rank 0 only before saving.
    For non-FSDP models, only rank 0 writes (consistent behaviour in DDP-less runs).

    Args:
        checkpoint_dir: Directory where checkpoint files will be written.
        model: FSDP-wrapped or plain GPT instance.
        optimizer: AdamW optimizer.
        training_args: Dict with 'step', 'model_config', and other metadata.
        rank: This process's rank.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if is_fsdp(model):
        with fsdp_full_state_dict_ctx(model):
            state_dict = model.state_dict()
        optim_state = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
    else:
        state_dict = model.state_dict()
        optim_state = optimizer.state_dict()

    if rank == 0:
        # Save model weights (safetensors format)
        model_path = os.path.join(checkpoint_dir, model_filename)
        state_dict_cpu = {k: v.contiguous().cpu() for k, v in state_dict.items()}
        save_file(state_dict_cpu, model_path)
        logger.info(f"Saved model to: {model_path}")

        # Save config.json (HuggingFace convention)
        config_path = os.path.join(checkpoint_dir, config_filename)
        model_config = training_args.get("model_config", {})
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"Saved config to: {config_path}")

        # Save training arguments / metadata
        args_path = os.path.join(checkpoint_dir, training_args_filename)
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(training_args, f, indent=2)
        logger.info(f"Saved training args to: {args_path}")

        # Save optimizer state (torch format)
        optim_path = os.path.join(checkpoint_dir, optim_filename)
        torch.save(optim_state, optim_path)
        logger.info(f"Saved optimizer to: {optim_path}")


def load_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    rank: int = 0,
) -> dict[str, Any]:
    """Load model weights and optionally optimizer state into existing objects.

    For FSDP models, the full state dict is loaded on rank 0 and then
    scattered to all ranks.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        model: FSDP-wrapped or plain model to load weights into.
        optimizer: Optimizer to restore state into; None skips optimizer loading.
        device: Target device for tensor loading.
        rank: This process's rank.

    Returns:
        Dict with loaded training_args (step, model_config, etc.).
    """
    # Load training args
    training_args: dict[str, Any] = {}
    args_path = os.path.join(checkpoint_dir, training_args_filename)
    if os.path.exists(args_path):
        with open(args_path, encoding="utf-8") as f:
            training_args = json.load(f)

    # Load model weights
    model_path = os.path.join(checkpoint_dir, model_filename)
    state_dict = load_file(model_path, device=str(device))
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    if is_fsdp(model):
        with fsdp_full_state_dict_ctx(model):
            model.load_state_dict(state_dict, strict=True, assign=True)
    else:
        model.load_state_dict(state_dict, strict=True, assign=True)

    # Load optimizer state
    if optimizer is not None:
        optim_path = os.path.join(checkpoint_dir, optim_filename)
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

    return training_args


def build_model_from_checkpoint(
    checkpoint_dir: str,
    device: torch.device,
    phase: str = "eval",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Instantiate a GPT model from a saved checkpoint.

    Args:
        checkpoint_dir: Directory containing the checkpoint files.
        device: Device on which to place the model.
        phase: Either 'eval' (model.eval()) or 'train' (model.train()).

    Returns:
        A (model, training_args) tuple with the model placed on device and
        training_args dict loaded from checkpoint metadata.
    """
    # Load config
    config_path = os.path.join(checkpoint_dir, config_filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config found in {checkpoint_dir}")
    with open(config_path, encoding="utf-8") as f:
        config_kwargs = json.load(f)

    config = GPTConfig(**config_kwargs)

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    training_args = load_checkpoint(checkpoint_dir, model, optimizer=None, device=device)

    if phase == "eval":
        model.eval()
    else:
        model.train()

    return model, training_args
