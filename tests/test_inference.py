import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

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
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: False)
    vllm_module = sys.modules["vllm"]
    vllm_module.LLM = LLM  # type: ignore[attr-defined]
    vllm_module.SamplingParams = SamplingParams  # type: ignore[attr-defined]

    backend = VLLM("model", tensor_parallel_size=2, trust_remote_code=True)
    result = backend.generate("prompt", max_tokens=8, temperature=0.2, top_k=4, stop=["<|end|>"])

    assert result == " answer"
    assert calls["llm"] == {"model": "model", "tensor_parallel_size": 2, "trust_remote_code": True, "dtype": "float16"}
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="vLLM numerical integration requires an Ampere or newer CUDA GPU",
)
def test_vllm_cached_decoding_matches_training_model(tmp_path) -> None:
    from tests.helpers import make_test_tokenizer
    from vllm import LLM, SamplingParams

    from tinygpt.config import make_config
    from tinygpt.model import TinyGPTForCausalLM

    torch.manual_seed(42)
    tokenizer = make_test_tokenizer(["Hello from TinyGPT. The sky is blue."])
    config = make_config(2, aspect_ratio=32, head_dim=32, vocab_size=tokenizer.get_vocab_size(), sequence_len=256)
    config.eos_token_id = tokenizer.encode_special("<|assistant_end|>")
    reference = TinyGPTForCausalLM(config).eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.03)
        reference.smear_lambda.fill_(0.5)
    reference.save_pretrained(tmp_path)
    tokenizer.save(str(tmp_path))
    # Different prompt lengths exercise per-request smear state and short attention.
    prompts = [torch.randint(9, config.vocab_size, (length,)).tolist() for length in (9, 145)]
    engine = LLM(
        model=str(tmp_path),
        dtype="bfloat16",
        tensor_parallel_size=min(2, torch.cuda.device_count()),
        gpu_memory_utilization=0.3,
        max_model_len=256,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    outputs = engine.generate(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        SamplingParams(max_tokens=4, temperature=0.0, logprobs=1),
        use_tqdm=False,
    )
    reference.cuda()
    with torch.no_grad():
        for prompt, output in zip(prompts, outputs, strict=True):
            completion = output.outputs[0]
            assert completion.token_ids
            for step, token in enumerate(completion.token_ids):
                inputs = torch.tensor([prompt + list(completion.token_ids[:step])], device="cuda")
                log_probs = reference(inputs).logits[0, -1].log_softmax(-1)
                assert abs(completion.logprobs[step][token].logprob - log_probs[token].item()) < 0.08
                assert log_probs[token] >= log_probs.max() - 0.08
