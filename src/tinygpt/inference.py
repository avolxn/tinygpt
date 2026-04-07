"""
Inference engine for token generation with utilities.

Taken from nanochat/engine.py as-is (no FSDP dependency — inference is always
single-GPU / single-process).
"""

import signal
import warnings
from collections import deque
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from tinygpt.kvcache import KVCache
from tinygpt.utils import compute_dtype, get_model_device


@contextmanager
def timeout(duration: int, formula: str) -> Generator[None, None, None]:
    """Context manager that raises TimeoutError if the body exceeds a time limit.

    Args:
        duration: Maximum allowed seconds before a TimeoutError is raised.
        formula: Human-readable label included in the TimeoutError message.

    Raises:
        TimeoutError: If the body takes longer than duration seconds.
    """

    def _handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"'{formula}': timed out after {duration}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def eval_with_timeout(formula: str, max_time: int = 3) -> object:
    """Evaluate a Python expression string with a wall-clock time limit.

    Args:
        formula: Python expression to evaluate.
        max_time: Maximum seconds to allow before aborting.

    Returns:
        The result of the expression, or None if evaluation fails or times out.
    """
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        signal.alarm(0)
        return None


def use_calculator(expr: str) -> object:
    """Safely evaluate a Python expression limited to arithmetic or .count() calls.

    Args:
        expr: Expression string to evaluate.

    Returns:
        The numeric result, or None if the expression is unsafe or evaluation fails.
    """
    expr = expr.replace(",", "")
    if all(x in "0123456789*+-/.() " for x in expr):
        if "**" in expr:
            return None
        return eval_with_timeout(expr)
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all(x in allowed for x in expr):
        return None
    dangerous = [
        "__",
        "import",
        "exec",
        "eval",
        "compile",
        "open",
        "file",
        "input",
        "raw_input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    ]
    if any(p in expr.lower() for p in dangerous):
        return None
    if ".count(" not in expr:
        return None
    return eval_with_timeout(expr)


def sample_next_token(
    logits: torch.Tensor,
    rng: torch.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Sample the next token from a logits distribution.

    Args:
        logits: Unnormalised logits of shape (B, vocab_size).
        rng: Random number generator for sampling.
        temperature: Softmax temperature; 0.0 selects the argmax deterministically.
        top_k: If set, restrict sampling to the top-k most likely tokens.

    Returns:
        Selected token ids of shape (B, 1).
    """
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)


@dataclass
class RowState:
    """Per-row generation state for the inference engine."""

    current_tokens: list[int]
    forced_tokens: deque[int] = field(default_factory=deque)
    in_python_block: bool = False
    python_expr_tokens: list[int] = field(default_factory=list)
    completed: bool = False


class Engine:
    """Efficient batched inference engine with KV cache and tool use."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any) -> None:
        """Initialise the engine with a model and tokenizer.

        Args:
            model: GPT model to use for generation.
            tokenizer: HuggingFaceTokenizer used to encode/decode tokens.
        """
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        tokens: list[int],
        num_samples: int = 1,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Stream tokens for num_samples parallel sequences from a shared prompt.

        Performs a single batch-1 prefill then decodes all samples in parallel
        using a shared KV cache.  Forced tool-output tokens are injected when
        a python block is evaluated by use_calculator.

        Args:
            tokens: Prompt token ids shared across all samples.
            num_samples: Number of independent sequences to generate.
            max_tokens: Maximum new tokens to generate; None means model seq_len.
            temperature: Sampling temperature; 0.0 = greedy.
            top_k: Top-k sampling cutoff; None disables top-k.
            seed: RNG seed for reproducibility.

        Yields:
            (token_column, token_masks) at each decode step, where token_column
            is a list of length num_samples and token_masks is 1 for sampled
            tokens and 0 for forced tool-output tokens.

        Raises:
            TypeError: If tokens is not a non-empty list of ints.
            RuntimeError: If the tokenizer is missing required special tokens.
        """
        if not isinstance(tokens, list) or not tokens or not isinstance(tokens[0], int):
            raise TypeError("tokens must be a non-empty list of ints")
        device = get_model_device(self.model)
        dtype = compute_dtype if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        def get_special(s: str) -> int | None:
            return self.tokenizer.encode_special(s)  # type: ignore[no-any-return]

        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        if output_start is None or output_end is None:
            raise RuntimeError("Tokenizer missing output delimiters")

        cfg: Any = self.model.config
        n_kv_head: int = cfg.n_kv_head
        head_dim: int = cfg.n_embd // cfg.n_head
        n_layer: int = cfg.n_layer
        seq_len: int = cfg.sequence_len

        kv_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            num_heads=n_kv_head,
            head_dim=head_dim,
            num_layers=n_layer,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)

        kv_len = (len(tokens) + max_tokens) if max_tokens is not None else seq_len
        kv_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_len,
            device=device,
            dtype=dtype,
            num_heads=n_kv_head,
            head_dim=head_dim,
            num_layers=n_layer,
        )
        kv_decode.prefill(kv_prefill)
        del kv_prefill

        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]
        num_generated = 0

        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(s.completed for s in row_states):
                break

            next_ids = sample_next_token(logits, rng, temperature, top_k)
            sampled = next_ids[:, 0].tolist()

            token_column: list[int] = []
            token_masks: list[int] = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                token = state.forced_tokens.popleft() if is_forced else sampled[i]
                token_column.append(token)
                state.current_tokens.append(token)

                if token in (assistant_end, bos):
                    state.completed = True
                if token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(token)

            yield token_column, token_masks
            num_generated += 1

            ids_next = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids_next, kv_cache=kv_decode)[:, -1, :]

    def generate_batch(
        self,
        tokens: list[int],
        num_samples: int = 1,
        **kwargs: Any,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Generate num_samples completions and return them all at once.

        Args:
            tokens: Prompt token ids.
            num_samples: Number of independent sequences to generate.
            **kwargs: Additional keyword arguments forwarded to generate().

        Returns:
            A (sequences, masks) tuple where sequences is a list of token id
            lists and masks is a corresponding list of 0/1 mask lists.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks, strict=True)):
                if not completed[i]:
                    if token in (assistant_end, bos):
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks
