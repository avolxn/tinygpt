"""
Process group setup, rank-0 logging, and FSDP helpers for multi-GPU training.

Training scripts call compute_init / compute_cleanup; FSDP scripts use make_fsdp_mixed_precision.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecision

from tinygpt.utils import compute_dtype

logger = logging.getLogger(__name__)


def is_distributed_requested() -> bool:
    """Check whether the process was launched by torchrun.

    Returns:
        True if RANK, LOCAL_RANK and WORLD_SIZE are all set in the environment.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_distributed_initialized() -> bool:
    """Check whether the distributed process group has been initialized.

    Returns:
        True if torch.distributed is available and the process group is active.
    """
    return dist.is_available() and dist.is_initialized()


def get_dist_info() -> tuple[bool, int, int, int]:
    """Read distributed process info from environment variables.

    Returns:
        A (is_dist, rank, local_rank, world_size) tuple. When not distributed,
        returns (False, 0, 0, 1).
    """
    if is_distributed_requested():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def print0(s: str = "", **kwargs: Any) -> None:
    """Print a message only on rank 0.

    Args:
        s: The string to print.
        **kwargs: Additional keyword arguments forwarded to the built-in print.
    """
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(s, **kwargs)


def compute_init(device_type: str = "cuda") -> tuple[bool, int, int, int, torch.device]:
    """Initialize compute environment: seeds, precision, and distributed process group.

    For FSDP, this sets up the process group; FSDP wrapping happens in the training script.

    Args:
        device_type: One of 'cuda', 'mps', or 'cpu'.

    Returns:
        A (is_dist, rank, local_rank, world_size, device) tuple.

    Raises:
        ValueError: If device_type is not one of the allowed values.
        RuntimeError: If the requested device is not available.
    """
    if device_type not in ("cuda", "mps", "cpu"):
        raise ValueError(f"Invalid device type: {device_type}")
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch is not configured for CUDA")
    if device_type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch is not configured for MPS")

    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
        torch.set_float32_matmul_precision("high")

    is_dist, rank, local_rank, world_size = get_dist_info()
    if is_dist and device_type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)

    if rank == 0:
        logger.info("Distributed world size: %s", world_size)

    return is_dist, rank, local_rank, world_size, device


def compute_cleanup() -> None:
    """Companion to compute_init; destroys the process group if initialized."""
    if is_distributed_initialized():
        dist.destroy_process_group()


def make_fsdp_mixed_precision(override: torch.dtype | None = None) -> Any:
    """Return a MixedPrecision config suitable for FSDP wrapping.

    Args:
        override: dtype to use; if None, the module-level compute_dtype is used.

    Returns:
        A MixedPrecision object with param, reduce, and buffer dtypes all set.
    """
    dtype = override if override is not None else compute_dtype
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )
