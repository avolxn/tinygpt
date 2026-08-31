import io
import json

import pytest

from tinygpt.inference import VLLMClient, VLLMError


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> io.BytesIO:
        return io.BytesIO(json.dumps(self.payload).encode())

    def __exit__(self, *args: object) -> None:
        pass


def test_vllm_client_posts_completion_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def urlopen(request: object, timeout: float) -> _Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response({"choices": [{"text": " answer"}]})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = VLLMClient("http://localhost:8000/v1/", "tinygpt", api_key="secret", timeout=12.0)

    result = client.generate("prompt", max_tokens=8, temperature=0.2, top_k=4, stop=["<|end|>"])

    request = captured["request"]
    assert result == " answer"
    assert captured["timeout"] == 12.0
    assert request.full_url == "http://localhost:8000/v1/completions"
    assert request.get_header("Authorization") == "Bearer secret"
    assert json.loads(request.data) == {
        "model": "tinygpt",
        "prompt": "prompt",
        "max_tokens": 8,
        "temperature": 0.2,
        "top_k": 4,
        "stop": ["<|end|>"],
    }


def test_vllm_client_rejects_invalid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: _Response({"choices": []}))

    with pytest.raises(VLLMError, match="Invalid vLLM response"):
        VLLMClient("http://localhost:8000/v1", "tinygpt").generate(
            "prompt", max_tokens=8, temperature=0.0
        )
