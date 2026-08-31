"""Native vLLM inference integration."""

from typing import Any


class VLLMNative:
    """Thin adapter around vLLM's offline ``LLM`` API."""

    def __init__(
        self,
        model: str,
        *,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as error:
            raise RuntimeError("Install vLLM to use the vllm backend: pip install vllm") from error

        self._llm: Any = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
        )
        self._sampling_params = SamplingParams

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_k: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Generate one completion with vLLM's native offline engine."""
        return self.generate_batch(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop=stop,
        )[0]

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_k: int | None = None,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Generate one completion for each prompt with vLLM's native engine."""
        params: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_k is not None:
            params["top_k"] = top_k
        if stop:
            params["stop"] = stop

        outputs = self._llm.generate(prompts, self._sampling_params(**params), use_tqdm=False)
        texts = [output.outputs[0].text for output in outputs if output.outputs]
        if len(texts) != len(prompts) or not all(isinstance(text, str) for text in texts):
            raise RuntimeError("vLLM returned an incomplete completion batch")
        return texts
