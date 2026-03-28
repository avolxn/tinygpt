"""
FSDP-aware checkpoint save/load.

Model weights are stored with safetensors (no pickle).
Optimizer state is stored with torch.save (still pickle, but isolated from weights).
Meta-data is stored as plain JSON.

Directory layout per checkpoint:
    <checkpoint_dir>/
        model.safetensors   — full model state dict (rank 0 only)
        optimizer.pt        — full optimizer state dict (rank 0 only)
        meta.json           — JSON metadata
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
optim_filename = "optimizer.pt"
meta_filename = "meta.json"


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


def save_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    meta: dict[str, Any],
    rank: int = 0,
) -> None:
    """Save model, optimizer and meta to checkpoint_dir.

    For FSDP models, state dicts are gathered to rank 0 only before saving.
    For non-FSDP models, only rank 0 writes (consistent behaviour in DDP-less runs).

    Args:
        checkpoint_dir: Directory where checkpoint files will be written.
        model: FSDP-wrapped or plain GPT instance.
        optimizer: AdamW optimizer.
        meta: JSON-serialisable metadata dict.
        rank: This process's rank.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if is_fsdp(model):
        with fsdp_full_state_dict_ctx(model):
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()

    if is_fsdp(model):
        optim_state = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
    else:
        optim_state = optimizer.state_dict()

    if rank == 0:
        model_path = os.path.join(checkpoint_dir, model_filename)
        # safetensors requires contiguous fp32/bf16 tensors
        state_dict_cpu = {k: v.contiguous().cpu() for k, v in state_dict.items()}
        save_file(state_dict_cpu, model_path)
        logger.info(f"Saved model to: {model_path}")

        optim_path = os.path.join(checkpoint_dir, optim_filename)
        torch.save(optim_state, optim_path)
        logger.info(f"Saved optimizer to: {optim_path}")

        meta_path = os.path.join(checkpoint_dir, meta_filename)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved meta to: {meta_path}")


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
        The meta dict loaded from meta.json.
    """
    meta_path = os.path.join(checkpoint_dir, meta_filename)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    model_path = os.path.join(checkpoint_dir, model_filename)
    state_dict = load_file(model_path, device=str(device))
    # Strip torch.compile prefix if present
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    if is_fsdp(model):
        with fsdp_full_state_dict_ctx(model):
            model.load_state_dict(state_dict, strict=True, assign=True)
    else:
        model.load_state_dict(state_dict, strict=True, assign=True)

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

    return meta  # type: ignore[no-any-return]


def get_checkpoint_dir(output_dir: str, run_name: str) -> str:
    """Return the checkpoint subdirectory for a given run.

    Args:
        output_dir: Root output directory.
        run_name: Name of the training run.

    Returns:
        Path to the checkpoint directory for this run.
    """
    return os.path.join(output_dir, "checkpoints", run_name)


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
        A (model, meta) tuple with the model placed on device and meta.json
        loaded as a dict.
    """
    meta_path = os.path.join(checkpoint_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    config_kwargs = meta["model_config"]
    config = GPTConfig(**config_kwargs)

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    load_checkpoint(checkpoint_dir, model, optimizer=None, device=device)

    if phase == "eval":
        model.eval()
    else:
        model.train()

    return model, meta
