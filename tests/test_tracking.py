"""Tests for mandatory MLflow tracking setup."""

from types import SimpleNamespace

import pytest

import tinygpt.tracking as tracking


def test_require_mlflow_tracking_configures_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: calls.append(("uri", uri)),
        set_experiment=lambda name: calls.append(("experiment", name)),
    )
    monkeypatch.setattr(tracking, "mlflow", fake_mlflow)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "tinygpt-test")

    assert tracking.require_mlflow_tracking() == "http://mlflow:5000"
    assert calls == [("uri", "http://mlflow:5000"), ("experiment", "tinygpt-test")]


def test_require_mlflow_tracking_defaults_to_local_store(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: calls.append(uri),
        set_experiment=lambda name: None,
    )
    monkeypatch.setattr(tracking, "mlflow", fake_mlflow)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)

    assert tracking.require_mlflow_tracking() == "file:./mlruns"
    assert calls == ["file:./mlruns"]


def test_require_mlflow_tracking_wraps_backend_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        set_experiment=lambda name: (_ for _ in ()).throw(ConnectionError("offline")),
    )
    monkeypatch.setattr(tracking, "mlflow", fake_mlflow)

    with pytest.raises(tracking.MlflowTrackingError, match="unavailable"):
        tracking.require_mlflow_tracking()
