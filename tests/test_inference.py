import sys
from types import ModuleType, SimpleNamespace

import pytest

from tinygpt.inference import VLLM


def test_vllm_uses_offline_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class SamplingParams:
        def __init__(self, **kwargs: object) -> None:
            calls["sampling_params"] = kwargs

    class LLM:
        def __init__(self, **kwargs: object) -> None:
            calls["llm"] = kwargs

        def generate(self, prompts: list[str], params: object, use_tqdm: bool) -> list[object]:
            calls["generate"] = (prompts, params, use_tqdm)
            return [SimpleNamespace(outputs=[SimpleNamespace(text=" answer")])]

    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    vllm_module = sys.modules["vllm"]
    vllm_module.LLM = LLM  # type: ignore[attr-defined]
    vllm_module.SamplingParams = SamplingParams  # type: ignore[attr-defined]

    backend = VLLM("model", tensor_parallel_size=2, trust_remote_code=True)
    result = backend.generate("prompt", max_tokens=8, temperature=0.2, top_k=4, stop=["<|end|>"])

    assert result == " answer"
    assert calls["llm"] == {"model": "model", "tensor_parallel_size": 2, "trust_remote_code": True}
    assert calls["sampling_params"] == {
        "max_tokens": 8,
        "temperature": 0.2,
        "top_k": 4,
        "stop": ["<|end|>"],
    }
    generate_call = calls["generate"]
    assert isinstance(generate_call, tuple)
    assert generate_call[0] == ["prompt"]
    assert generate_call[2] is False


def test_vllm_requires_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "vllm", None)

    with pytest.raises(RuntimeError, match="Install vLLM"):
        VLLM("model")
