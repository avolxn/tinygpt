"""Tests for mandatory W&B authentication."""

from types import SimpleNamespace

import pytest
import wandb

from tinygpt.tracking import WandbAuthError, require_wandb_auth


def test_require_wandb_auth_verifies_configured_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    login_calls: list[dict[str, object]] = []
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(wandb, "Api", lambda **_: SimpleNamespace(api_key="secret"))
    monkeypatch.setattr(wandb, "login", lambda **kwargs: login_calls.append(kwargs) or True)

    require_wandb_auth(interactive=False, timeout=3)

    assert login_calls == [{"verify": True, "timeout": 3}]


@pytest.mark.parametrize("mode", ["offline", "disabled"])
def test_require_wandb_auth_rejects_non_online_modes(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv("WANDB_MODE", mode)

    with pytest.raises(WandbAuthError, match="requires online W&B"):
        require_wandb_auth(interactive=True)


def test_require_wandb_auth_requires_preconfigured_key_for_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(wandb, "Api", lambda **_: SimpleNamespace(api_key=None))

    with pytest.raises(WandbAuthError, match="credentials are required"):
        require_wandb_auth(interactive=False)


def test_require_wandb_auth_allows_interactive_login(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(wandb, "login", lambda **_: True)

    require_wandb_auth(interactive=True)
