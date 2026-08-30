"""Mandatory Weights & Biases authentication helpers."""

from __future__ import annotations

import os

import wandb


class WandbAuthError(RuntimeError):
    """Raised when an online W&B session cannot be authenticated."""


def _has_configured_api_key(timeout: int) -> bool:
    try:
        return bool(wandb.Api(timeout=timeout).api_key)
    except wandb.errors.UsageError:
        return False


def require_wandb_auth(*, interactive: bool, timeout: int = 15) -> None:
    """Require verified online W&B credentials before starting training.

    Interactive mode may prompt for a key. Non-interactive mode is intended
    for orchestrators and distributed workers, where credentials must already
    be supplied through the environment or W&B's credential store.
    """
    mode = os.environ.get("WANDB_MODE", "online").lower()
    if mode != "online":
        raise WandbAuthError(f"Training requires online W&B tracking; WANDB_MODE={mode!r} is not allowed.")

    if not interactive and not _has_configured_api_key(timeout):
        raise WandbAuthError(
            "W&B credentials are required. Set WANDB_API_KEY or run `uv run wandb login --verify` before training."
        )

    try:
        authenticated = wandb.login(verify=True, timeout=timeout)
    except Exception as exc:
        raise WandbAuthError(f"W&B authentication failed: {exc}") from exc
    if not authenticated:
        raise WandbAuthError("W&B authentication failed: the API key was not accepted.")
