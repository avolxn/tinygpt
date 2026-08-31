"""Verify MLflow tracking before launching training jobs."""

from tinygpt.tracking import require_mlflow_tracking

tracking_uri = require_mlflow_tracking()
print(f"MLflow tracking verified: {tracking_uri}")
