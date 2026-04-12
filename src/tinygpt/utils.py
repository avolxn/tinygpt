"""
Hardware helpers, logging, cache I/O, and MFU tables for tinygpt.

Distributed training setup lives in tinygpt.distributed.
"""

import logging
import os
import re
import urllib.request

import torch
from filelock import FileLock

dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def detect_compute_dtype() -> tuple[torch.dtype, str]:
    """Detect the best compute dtype for the current hardware.

    Returns:
        A (dtype, reason) tuple where reason is a human-readable explanation.
    """
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
        """Format a log record with ANSI color codes.

        Args:
            record: The log record to format.

        Returns:
            The formatted log message string with color codes applied.
        """
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == "INFO":
            message = re.sub(r"(\d+\.?\d*\s*(?:GB|MB|%|docs))", rf"{self.BOLD}\1{self.RESET}", message)
        return message


def setup_default_logging() -> None:
    """Configure the root logger with a colored StreamHandler."""
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


setup_default_logging()
logger = logging.getLogger(__name__)


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device where model parameters reside.

    Args:
        model: The PyTorch module to inspect.

    Returns:
        The device of the first parameter, or the result of model.get_device()
        if available.
    """
    if hasattr(model, "get_device"):
        return model.get_device()  # type: ignore[operator, no-any-return]
    return next(model.parameters()).device


def autodetect_device_type() -> str:
    """Detect the best available compute device.

    Returns:
        One of 'cuda', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"Autodetected device type: {device_type}")
    return device_type


class DummyWandb:
    """Drop-in wandb replacement that silently ignores all calls."""

    def log(self, *args: object, **kwargs: object) -> None:
        """Silently ignore log calls."""
        ...

    def finish(self) -> None:
        """Silently ignore finish calls."""
        ...


peak_flops_table: tuple[tuple[frozenset[str], float], ...] = (
    (frozenset({"gb200"}), 2.5e15),
    (frozenset({"grace blackwell"}), 2.5e15),
    (frozenset({"b200"}), 2.25e15),
    (frozenset({"b100"}), 1.8e15),
    (frozenset({"h200", "nvl"}), 836e12),
    (frozenset({"h200", "pcie"}), 836e12),
    (frozenset({"h200"}), 989e12),
    (frozenset({"h100", "nvl"}), 835e12),
    (frozenset({"h100", "pcie"}), 756e12),
    (frozenset({"h100"}), 989e12),
    (frozenset({"h800", "nvl"}), 989e12),
    (frozenset({"h800"}), 756e12),
    (frozenset({"a100"}), 312e12),
    (frozenset({"a800"}), 312e12),
    (frozenset({"a40"}), 149.7e12),
    (frozenset({"a30"}), 165e12),
    (frozenset({"l40s"}), 362e12),
    (frozenset({"l40-s"}), 362e12),
    (frozenset({"l40 s"}), 362e12),
    (frozenset({"l4"}), 121e12),
    (frozenset({"mi355"}), 2.5e15),
    (frozenset({"mi325"}), 1.3074e15),
    (frozenset({"mi300x"}), 1.3074e15),
    (frozenset({"mi300a"}), 980.6e12),
    (frozenset({"mi250x"}), 383e12),
    (frozenset({"mi250"}), 362.1e12),
    (frozenset({"5090"}), 209.5e12),
    (frozenset({"4090"}), 165.2e12),
    (frozenset({"3090"}), 71e12),
)


def get_peak_flops(device_name: str) -> float:
    """Return the BF16 peak FLOP/s for a known GPU model.

    Args:
        device_name: GPU model string as returned by torch.cuda.get_device_name().

    Returns:
        Peak BF16 FLOP/s as a float. Returns inf for unrecognised GPU names.
    """
    name = device_name.lower()
    for patterns, flops in peak_flops_table:
        if all(p in name for p in patterns):
            return flops
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float("inf")


def get_cache_dir() -> str:
    """Return the tinygpt cache directory, creating it if necessary.

    Returns:
        Absolute path to ~/.cache/tinygpt.
    """
    path = os.path.join(os.path.expanduser("~"), ".cache", "tinygpt")
    os.makedirs(path, exist_ok=True)
    return path


def download_file_with_lock(url: str, filename: str) -> str:
    """Download a file from a URL into the tinygpt cache directory.

    Safe for concurrent callers: uses a lock file so only one process
    downloads at a time; all others wait and reuse the cached copy.

    Args:
        url: URL to download from.
        filename: Local filename to save as (relative to the cache dir).

    Returns:
        Absolute path to the downloaded file.
    """
    cache_dir = get_cache_dir()
    file_path = os.path.join(cache_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        if os.path.exists(file_path):
            return file_path
        print(f"Downloading {url} ...")
        with urllib.request.urlopen(url) as response:
            content = response.read()
        with open(file_path, "wb") as f:
            f.write(content)
        print(f"Saved to {file_path}")

    return file_path
