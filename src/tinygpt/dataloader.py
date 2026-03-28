"""
Distributed data loader using HF `datasets` streaming.

BOS-aligned bestfit packing:
  - Every row starts with BOS token
  - Documents packed with best-fit algorithm to minimise cropping
  - When no document fits, crops a document to fill remaining space exactly
  - 100% utilisation (no padding), ~35% tokens cropped at T=2048

Replaces nanochat's pyarrow + manual parquet downloader with:
  datasets.load_dataset(name, streaming=True) → dataset.shard(world_size, rank)
"""

import logging
from collections.abc import Iterator
from typing import Any

import torch
from datasets import load_dataset

from tinygpt.runtime import get_dist_info

logger = logging.getLogger(__name__)


def document_batches(
    dataset_name: str,
    split: str,
    rank: int,
    world_size: int,
    batch_size: int,
    text_field: str = "text",
) -> Iterator[tuple[list[str], int]]:
    """
    Infinite iterator that yields (text_batch, epoch) pairs from HF streaming.

    Args:
        dataset_name: HF dataset identifier, e.g. "HuggingFaceFW/fineweb"
        split: "train" or "validation"
        rank: this process's rank (0-indexed)
        world_size: total number of processes
        batch_size: number of documents per batch yielded to the tokenizer
        text_field: column name containing document text
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
    """
    BOS-aligned bestfit dataloader backed by HF datasets streaming.

    Yields (inputs, targets) where:
      inputs  = (B, T) long tensor of token ids
      targets = (B, T) long tensor of token ids shifted by 1

    Args:
        tokenizer: HuggingFaceTokenizer with encode() and get_bos_token_id()
        B: batch size (number of sequences per step)
        T: sequence length
        dataset_name: HF dataset identifier
        split: "train" or "val" / "validation"
        device: target device for the output tensors
        tokenizer_batch_size: documents to tokenize at once
        buffer_size: bestfit document buffer size (larger = better packing)
        text_field: column name in the HF dataset containing document text
    """
    _, rank, _, world_size = get_dist_info()
    hf_split = "validation" if split == "val" else split

    batches = document_batches(dataset_name, hf_split, rank, world_size, tokenizer_batch_size, text_field)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer: list[list[int]] = []

    row_capacity = T + 1  # T inputs + 1 target requires T+1 tokens per row
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
            # encode returned a flat list (single string) — wrap it
            token_lists = [token_lists]
        doc_buffer.extend(token_lists)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest document that fits entirely (best-fit)
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
                    # Nothing fits — crop shortest document to fill remaining space
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets


# ---------------------------------------------------------------------------
# SFT dataloader (from a Task)
# ---------------------------------------------------------------------------


def sft_data_loader(
    tokenizer: Any,
    task: Any,
    B: int,
    T: int,
    device: torch.device | str = "cuda",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    SFT dataloader that packs tokenized conversations with loss masking.

    Targets are -1 (ignore_index) for non-assistant tokens so the model only
    learns to predict assistant responses.

    Yields (inputs, targets) tensors of shape (B, T).
    """
    _, rank, _, world_size = get_dist_info()
    n = len(task)
    # Each rank processes a disjoint slice of the dataset
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

    doc_buffer: list[tuple[list[int], list[int]]] = []  # (ids, mask) pairs

    def refill() -> None:
        for idx in indices:
            conv = task[idx % n]
            ids, mask = tokenizer.render_conversation(conv, max_tokens=T + 1)
            doc_buffer.append((ids, mask))

    while True:
        for row_idx in range(B):
            pos = 0
            row_buffer[row_idx].fill_(bos_token)
            target_buffer[row_idx].fill_(-1)

            while pos < row_capacity:
                if not doc_buffer:
                    refill()

                remaining = row_capacity - pos
                ids, mask = doc_buffer[0]

                if len(ids) <= remaining:
                    doc_buffer.pop(0)
                    length = len(ids)
                    row_buffer[row_idx, pos : pos + length] = torch.tensor(ids, dtype=torch.long)
                    target_buffer[row_idx, pos : pos + length] = torch.tensor(
                        [i if m else -1 for i, m in zip(ids, mask, strict=True)], dtype=torch.long
                    )
                    pos += length
                else:
                    # Crop
                    doc_buffer[0] = (ids[:remaining], mask[:remaining])
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
