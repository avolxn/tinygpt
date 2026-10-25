"""
Hardware helpers, logging, cache I/O, and MFU tables for tinygpt.

Distributed training setup lives in tinygpt.distributed.
"""

import logging
import os
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
        try:
            return dtype_map[env], f"set via TINYGPT_DTYPE={env}"
        except KeyError as exc:
            raise ValueError(f"Unsupported TINYGPT_DTYPE={env!r}; choose from {sorted(dtype_map)}") from exc
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        return torch.float32, (
            f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
        )
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"


compute_dtype, compute_dtype_reason = detect_compute_dtype()


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
    if os.path.basename(filename) != filename:
        raise ValueError(f"filename must be a plain file name, got {filename!r}")
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
