"""vLLM inference integration."""

from typing import Any

import torch

from tinygpt.execution import use_calculator
from tinygpt.tokenizer import HuggingFaceTokenizer


class VLLM:
    """Thin adapter around vLLM's offline ``LLM`` API."""

    def __init__(
        self,
        model: str,
        *,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = False,
    ) -> None:
        if tensor_parallel_size < 1:
            raise ValueError(f"tensor_parallel_size must be positive, got {tensor_parallel_size}")
        try:
            from vllm import LLM, SamplingParams
        except ImportError as error:
            raise RuntimeError("Install vLLM to use the vllm backend: pip install vllm") from error

        self._llm: Any = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        )
        self._sampling_params = SamplingParams

    def generate(
        self,
        prompt: str | list[int],
        *,
        max_tokens: int,
        temperature: float,
        top_k: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Generate one completion with vLLM's offline engine."""
        return self.generate_batch(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop=stop,
        )[0]

    def generate_batch(
        self,
        prompts: list[str | list[int]],
        *,
        max_tokens: int,
        temperature: float,
        top_k: int | None = None,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Generate one completion for each prompt with vLLM's engine."""
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not prompts:
            return []

        params: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_k is not None:
            params["top_k"] = top_k
        if stop:
            params["stop"] = stop

        requests = [{"prompt_token_ids": p} if isinstance(p, list) else p for p in prompts]
        outputs = self._llm.generate(requests, self._sampling_params(**params), use_tqdm=False)
        texts = [output.outputs[0].text for output in outputs if output.outputs]
        if len(texts) != len(prompts) or not all(isinstance(text, str) for text in texts):
            raise RuntimeError("vLLM returned an incomplete completion batch")
        return texts

    def choose_tokens(self, prompts: list[list[int]], candidates: list[list[int]]) -> list[int]:
        """Select the highest-logit answer token among each question's choices.

        Args:
            prompts: Tokenized questions including the assistant prefix.
            candidates: Allowed single-token answers for each prompt.

        Returns:
            The greedy token from the restricted answer vocabulary per question.
        """
        params = [self._sampling_params(max_tokens=1, temperature=0.0, allowed_token_ids=ids) for ids in candidates]
        outputs = self._llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        return [int(result.outputs[0].token_ids[0]) for result in outputs]

    def generate_conversation(
        self,
        prompt: list[int],
        tokenizer: HuggingFaceTokenizer,
        *,
        max_tokens: int,
        temperature: float,
        top_k: int | None = None,
    ) -> str:
        """Generate a chat response with the dataset calculator annotations.

        Args:
            prompt: Tokenized conversation ending with the assistant prefix.
            tokenizer: The unchanged teacher tokenizer.
            max_tokens: Total sampled-token budget across tool continuations.
            temperature: Sampling temperature, with zero selecting greedy decoding.
            top_k: Optional vocabulary sampling cap.

        Returns:
            Assistant text, including calculator markers and supplied outputs.
        """
        generated: list[int] = []
        remaining = max_tokens
        while remaining > 0:
            params = self._sampling_params(
                max_tokens=remaining,
                temperature=temperature,
                top_k=top_k or -1,
                stop=["</calculator>"],
                include_stop_str_in_output=True,
                stop_token_ids=[tokenizer.get_eos_token_id()],
                skip_special_tokens=False,
            )
            response = self._llm.generate([{"prompt_token_ids": prompt + generated}], params, use_tqdm=False)[
                0
            ].outputs[0]
            tokens = list(response.token_ids)
            if not tokens:
                break
            generated.extend(tokens)
            remaining -= len(tokens)
            text = tokenizer.decode(generated)
            if not text.endswith("</calculator>") or "<calculator>" not in text:
                break
            expression = text.rsplit("<calculator>", 1)[1].removesuffix("</calculator>")
            result = use_calculator(expression)
            if result is None:
                break
            forced = tokenizer.encode("<result>" + result + "</result>")[:remaining]
            generated.extend(forced)
            remaining -= len(forced)
        return tokenizer.decode(generated).removesuffix(tokenizer.decode([tokenizer.get_eos_token_id()]))
