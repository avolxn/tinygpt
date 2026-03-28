"""
Hardware and process setup utilities for tinygpt.

Covers: device detection, compute_dtype, FSDP mixed precision, distributed init,
print0, MFU utilities. Nothing about the model.
"""

import logging
import os
import re
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecision

# ---------------------------------------------------------------------------
# Compute dtype
# ---------------------------------------------------------------------------

dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def detect_compute_dtype() -> tuple[torch.dtype, str]:
    env = os.environ.get("TINYGPT_DTYPE")
    if env is not None:
        return dtype_map[env], f"set via TINYGPT_DTYPE={env}"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        return torch.float32, (
            f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
        )
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"


compute_dtype, compute_dtype_reason = detect_compute_dtype()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == "INFO":
            message = re.sub(r"(\d+\.?\d*\s*(?:GB|MB|%|docs))", rf"{self.BOLD}\1{self.RESET}", message)
        return message


def setup_default_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


setup_default_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def is_distributed_requested() -> bool:
    """True if launched by torchrun (env vars present)."""
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_distributed_initialized() -> bool:
    """True if torch.distributed is available and process group is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_dist_info() -> tuple[bool, int, int, int]:
    """Return (is_dist, rank, local_rank, world_size)."""
    if is_distributed_requested():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def print0(s: str = "", **kwargs: Any) -> None:
    """Print only on rank 0."""
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(s, **kwargs)


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device where *model* parameters reside."""
    if hasattr(model, "get_device"):
        return model.get_device()  # type: ignore[operator, no-any-return]
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------


def autodetect_device_type() -> str:
    """Return 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type


def compute_init(device_type: str = "cuda") -> tuple[bool, int, int, int, torch.device]:
    """
    Initialize compute environment: seeds, precision, distributed process group.

    Returns (is_dist, rank, local_rank, world_size, device).
    For FSDP, this sets up the process group; FSDP wrapping happens in the training script.
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
        logger.info(f"Distributed world size: {world_size}")

    return is_dist, rank, local_rank, world_size, device


def compute_cleanup() -> None:
    """Companion to compute_init; destroys the process group if initialized."""
    if is_distributed_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# FSDP mixed precision
# ---------------------------------------------------------------------------


def make_fsdp_mixed_precision(override: torch.dtype | None = None) -> Any:
    """
    Return a MixedPrecision config suitable for FSDP wrapping.

    Uses compute_dtype by default.
    """
    dtype = override if override is not None else compute_dtype
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )


# ---------------------------------------------------------------------------
# MFU / peak flops table
# ---------------------------------------------------------------------------


class DummyWandb:
    """Drop-in wandb replacement that silently ignores all calls."""

    def log(self, *args: object, **kwargs: object) -> None:
        pass

    def finish(self) -> None:
        pass


# BF16 peak flops for known GPUs; first matching entry wins (most specific first)
peak_flops_table: tuple[tuple[list[str], float], ...] = (
    (["gb200"], 2.5e15),
    (["grace blackwell"], 2.5e15),
    (["b200"], 2.25e15),
    (["b100"], 1.8e15),
    (["h200", "nvl"], 836e12),
    (["h200", "pcie"], 836e12),
    (["h200"], 989e12),
    (["h100", "nvl"], 835e12),
    (["h100", "pcie"], 756e12),
    (["h100"], 989e12),
    (["h800", "nvl"], 989e12),
    (["h800"], 756e12),
    (["a100"], 312e12),
    (["a800"], 312e12),
    (["a40"], 149.7e12),
    (["a30"], 165e12),
    (["l40s"], 362e12),
    (["l40-s"], 362e12),
    (["l40 s"], 362e12),
    (["l4"], 121e12),
    (["mi355"], 2.5e15),
    (["mi325"], 1.3074e15),
    (["mi300x"], 1.3074e15),
    (["mi300a"], 980.6e12),
    (["mi250x"], 383e12),
    (["mi250"], 362.1e12),
    (["5090"], 209.5e12),
    (["4090"], 165.2e12),
    (["3090"], 71e12),
)


def get_peak_flops(device_name: str) -> float:
    """Return BF16 peak TFLOPS for a known GPU. Returns inf for unknown GPUs."""
    name = device_name.lower()
    for patterns, flops in peak_flops_table:
        if all(p in name for p in patterns):
            return flops
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float("inf")
