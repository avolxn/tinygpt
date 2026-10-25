"""Hardware and compute dtype selection for tinygpt.

Distributed training setup lives in tinygpt.distributed.
"""

import os

import torch

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
