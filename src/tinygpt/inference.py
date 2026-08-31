"""Lightweight client for remote vLLM serving."""

import json
import urllib.error
import urllib.request
from typing import Any, cast


class VLLMError(RuntimeError):
    """Raised when the vLLM OpenAI-compatible API returns an error."""


class VLLMClient:
    """Small dependency-free client for a vLLM OpenAI-compatible server."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_k: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text through vLLM's `/v1/completions` endpoint."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        if stop:
            payload["stop"] = stop

        request = urllib.request.Request(
            f"{self.base_url}/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = cast(dict[str, Any], json.load(response))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise VLLMError(f"vLLM request failed ({error.code}): {detail}") from error
        except urllib.error.URLError as error:
            raise VLLMError(f"Could not reach vLLM at {self.base_url}: {error.reason}") from error

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise VLLMError(f"Invalid vLLM response: {body!r}")
        text = choices[0].get("text")
        if not isinstance(text, str):
            raise VLLMError(f"vLLM response has no text choice: {body!r}")
        return text
