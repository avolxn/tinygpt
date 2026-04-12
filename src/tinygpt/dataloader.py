"""
Distributed data loader using HF `datasets` streaming.

BOS-aligned bestfit packing:
  - Every row starts with BOS token
  - Documents packed with best-fit algorithm to minimise cropping
  - When no document fits, crops a document to fill remaining space exactly
  - 100% utilisation (no padding), ~35% tokens cropped at T=2048
"""

import logging
from collections import deque
from collections.abc import Iterator
from typing import Any

import torch
from datasets import load_dataset

from tinygpt.distributed import get_dist_info

logger = logging.getLogger(__name__)


def document_batches(
    dataset_name: str,
    split: str,
    rank: int,
    world_size: int,
    batch_size: int,
    text_field: str = "text",
) -> Iterator[tuple[list[str], int]]:
    """Infinite iterator that yields batches of document text from HF streaming.

    Args:
        dataset_name: HF dataset identifier, e.g. "HuggingFaceFW/fineweb".
        split: Dataset split, e.g. "train" or "validation".
        rank: This process's rank (0-indexed).
        world_size: Total number of processes.
        batch_size: Number of documents per batch yielded to the tokenizer.
        text_field: Column name containing document text.

    Yields:
        A (texts, epoch) tuple where texts is a list of document strings and
        epoch is the 1-based epoch counter.
    """
    epoch = 1
    while True:
        ds = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)
        for batch in ds.iter(batch_size=batch_size):
            texts = batch.get(text_field, batch.get("content", []))
            if texts:
                yield texts, epoch
        epoch += 1


def tokenizing_distributed_data_loader_bestfit(
    tokenizer: Any,
    B: int,
    T: int,
    dataset_name: str,
    split: str,
    device: torch.device | str = "cuda",
    tokenizer_batch_size: int = 128,
    buffer_size: int = 1000,
    text_field: str = "text",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """BOS-aligned bestfit dataloader backed by HF datasets streaming.

    Args:
        tokenizer: HuggingFaceTokenizer with encode() and get_bos_token_id().
        B: Batch size (number of sequences per step).
        T: Sequence length.
        dataset_name: HF dataset identifier.
        split: "train" or "val" / "validation".
        device: Target device for the output tensors.
        tokenizer_batch_size: Documents to tokenize at once.
        buffer_size: Bestfit document buffer size; larger values improve packing.
        text_field: Column name in the HF dataset containing document text.

    Yields:
        An (inputs, targets) tuple of (B, T) long tensors where targets are
        inputs shifted by one position.
    """
    _, rank, _, world_size = get_dist_info()
    hf_split = "validation" if split == "val" else split

    batches = document_batches(dataset_name, hf_split, rank, world_size, tokenizer_batch_size, text_field)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer: list[list[int]] = []

    row_capacity = T + 1
    use_cuda = str(device) == "cuda"

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T :].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T :].view(B, T)

    def refill_buffer() -> None:
        doc_batch, _ = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        if isinstance(token_lists[0], int):
            token_lists = [token_lists]
        doc_buffer.extend(token_lists)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos : pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets


def sft_data_loader(
    tokenizer: Any,
    task: Any,
    B: int,
    T: int,
    device: torch.device | str = "cuda",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """SFT dataloader that packs tokenized conversations with loss masking.

    Targets are -1 (ignore_index) for non-assistant tokens so the model only
    learns to predict assistant responses.

    Args:
        tokenizer: HuggingFaceTokenizer with render_conversation() and get_bos_token_id().
        task: Task object with __len__ and __getitem__ returning conversation dicts.
        B: Batch size (number of sequences per step).
        T: Sequence length.
        device: Target device for the output tensors.

    Yields:
        An (inputs, targets) tuple of (B, T) long tensors where non-assistant
        target positions are set to -1.
    """
    _, rank, _, world_size = get_dist_info()
    n = len(task)
    indices = list(range(rank, n, world_size))

    bos_token = tokenizer.get_bos_token_id()
    row_capacity = T + 1
    use_cuda = str(device) == "cuda"

    row_buffer = torch.full((B, row_capacity), fill_value=bos_token, dtype=torch.long)
    target_buffer = torch.full((B, row_capacity), fill_value=-1, dtype=torch.long)

    cpu_inputs_buf = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    cpu_targets_buf = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_inputs = torch.empty(B * T, dtype=torch.long, device=device).view(B, T)
    gpu_targets = torch.empty(B * T, dtype=torch.long, device=device).view(B, T)
    cpu_inputs = cpu_inputs_buf[: B * T].view(B, T)
    cpu_targets = cpu_targets_buf[: B * T].view(B, T)

    doc_deque: deque[tuple[list[int], list[int]]] = deque()  # (ids, mask) pairs

    def refill() -> None:
        for idx in indices:
            conv = task[idx % n]
            ids, mask = tokenizer.render_conversation(conv, max_tokens=T + 1)
            doc_deque.append((ids, mask))

    while True:
        for row_idx in range(B):
            pos = 0
            row_buffer[row_idx].fill_(bos_token)
            target_buffer[row_idx].fill_(-1)

            while pos < row_capacity:
                if not doc_deque:
                    refill()

                remaining = row_capacity - pos
                ids, mask = doc_deque[0]

                if len(ids) <= remaining:
                    doc_deque.popleft()
                    length = len(ids)
                    row_buffer[row_idx, pos : pos + length] = torch.tensor(ids, dtype=torch.long)
                    target_buffer[row_idx, pos : pos + length] = torch.tensor(
                        [i if m else -1 for i, m in zip(ids, mask, strict=True)], dtype=torch.long
                    )
                    pos += length
                else:
                    doc_deque[0] = (ids[:remaining], mask[:remaining])
                    length = remaining
                    row_buffer[row_idx, pos : pos + length] = torch.tensor(ids[:remaining], dtype=torch.long)
                    target_buffer[row_idx, pos : pos + length] = torch.tensor(
                        [i if m else -1 for i, m in zip(ids[:remaining], mask[:remaining], strict=True)],
                        dtype=torch.long,
                    )
                    pos += length

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(target_buffer[:, 1:])
        gpu_inputs.copy_(cpu_inputs, non_blocking=use_cuda)
        gpu_targets.copy_(cpu_targets, non_blocking=use_cuda)
        yield gpu_inputs, gpu_targets
