"""
Inference engine with KV cache.

Taken from nanochat/engine.py as-is (no FSDP dependency — inference is always
single-GPU / single-process).
"""

import signal
import warnings
from collections import deque
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from tinygpt.runtime import compute_dtype, get_model_device

# ---------------------------------------------------------------------------
# Calculator tool helpers (used by the chat engine)
# ---------------------------------------------------------------------------


@contextmanager
def timeout(duration: int, formula: str) -> Generator[None, None, None]:
    def _handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"'{formula}': timed out after {duration}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def eval_with_timeout(formula: str, max_time: int = 3) -> object:
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        signal.alarm(0)
        return None


def use_calculator(expr: str) -> object:
    """Safely evaluate a Python expression (math or .count() only)."""
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


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------


class KVCache:
    """
    Pre-allocated KV cache for flash attention (B, T, H, D) layout.

    Compatible with both FA2 and SDPA backends.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.prev_embedding: torch.Tensor | None = None

    def reset(self) -> None:
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self) -> int:
        return self.cache_seqlens[0].item()  # type: ignore[return-value]

    def get_layer_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens: int) -> None:
        self.cache_seqlens += num_tokens

    def prefill(self, other: "KVCache") -> None:
        """Copy KV state from a batch-1 prefill cache into this decode cache."""
        assert self.get_pos() == 0
        assert self.n_layers == other.n_layers
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@torch.inference_mode()
def sample_next_token(
    logits: torch.Tensor,
    rng: torch.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Sample next token from logits (B, vocab_size). Returns (B, 1)."""
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


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RowState:
    def __init__(self, tokens: list[int]) -> None:
        self.current_tokens = tokens
        self.forced_tokens: deque[int] = deque()
        self.in_python_block = False
        self.python_expr_tokens: list[int] = []
        self.completed = False


class Engine:
    """Efficient batched inference engine with KV cache and tool use."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any) -> None:
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
        """
        Streaming batched generation with single prefill + KV cache decode.

        Yields (token_column, token_masks) at each step:
            token_column: list[int] of length num_samples
            token_masks:  list[int], 1=sampled, 0=forced (tool output)
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int)
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
        assert output_start is not None and output_end is not None, "Tokenizer missing output delimiters"

        cfg: Any = self.model.config
        n_kv_head: int = cfg.n_kv_head
        head_dim: int = cfg.n_embd // cfg.n_head
        n_layer: int = cfg.n_layer
        seq_len: int = cfg.sequence_len

        # 1) Prefill with batch=1
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

        # 2) Clone KV cache for decode
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
        """Non-streaming generation. Returns (sequences, masks)."""
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
