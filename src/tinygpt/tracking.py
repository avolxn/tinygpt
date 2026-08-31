"""Mandatory MLflow tracking setup."""

from __future__ import annotations

import os

import mlflow


class MlflowTrackingError(RuntimeError):
    """Raised when the configured MLflow tracking backend is unavailable."""


def require_mlflow_tracking() -> str:
    """Configure and verify the MLflow tracking backend before training."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "tinygpt")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        raise MlflowTrackingError(
            f"MLflow tracking is required but unavailable at {tracking_uri!r}: {exc}"
        ) from exc
    return tracking_uri
