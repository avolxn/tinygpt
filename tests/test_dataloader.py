"""Tests for the dataloader.

- BOS token starts every row
- bestfit packing doesn't drop tokens (all tokens in buffer used).
"""

from itertools import cycle, islice

import pytest
import torch
from datasets import Dataset
from tests.helpers import make_test_tokenizer

from tinygpt.dataloader import _bestfit_batches, conversation_data_loader, document_batches, text_data_loader
from tinygpt.tokenizer import HuggingFaceTokenizer
from tinygpt.train import PreBatchedIterableDataset


def test_accumulated_batches_do_not_alias_loader_buffers():
    inputs = torch.zeros((1, 4), dtype=torch.long)
    targets = torch.zeros_like(inputs)

    def batches():
        for value in range(3):
            inputs.fill_(value)
            targets.fill_(value + 1)
            yield inputs, targets

    accumulated = list(PreBatchedIterableDataset(batches()))
    assert [batch["input_ids"][0, 0].item() for batch in accumulated] == [0, 1, 2]
    assert [batch["labels"][0, 0].item() for batch in accumulated] == [1, 2, 3]


def test_streaming_single_shard_supports_eight_ranks(monkeypatch):
    dataset = Dataset.from_dict({"text": [str(index) for index in range(16)]}).to_iterable_dataset(num_shards=1)
    monkeypatch.setattr("tinygpt.dataloader.load_dataset", lambda *args, **kwargs: dataset)
    partitions = [set(next(document_batches("local", "train", rank, 8, 2))) for rank in range(8)]
    assert set.union(*partitions) == {str(index) for index in range(16)}
    assert sum(map(len, partitions)) == 16


@pytest.fixture(scope="module")
def tokenizer() -> HuggingFaceTokenizer:
    texts = [
        "Hello world! " * 20,
        "Python is great. " * 20,
        "The quick brown fox. " * 20,
        "1 2 3 4 5 6 7 8 9 10. " * 20,
    ]
    return make_test_tokenizer(texts * 20)


@pytest.mark.parametrize("rank,expected,excluded", [(0, "Hello", "Python"), (1, "Python", "Hello")])
def test_local_text_is_sharded(tokenizer, tmp_path, monkeypatch, rank, expected, excluded):
    path = tmp_path / "documents.txt"
    path.write_text("Hello " * 100 + "\n" + "Python " * 100 + "\n", encoding="utf-8")
    monkeypatch.setattr("tinygpt.dataloader.get_dist_info", lambda: (True, rank, rank, 2))
    loader = text_data_loader(tokenizer, str(path), 1, 32, "cpu")
    inputs, _ = next(loader)
    text = tokenizer.decode(inputs[0].tolist())
    assert expected in text
    assert excluded not in text


def make_in_memory_loader(tokenizer, docs, B, T, device="cpu"):
    """Exercise production packing with an in-memory document source."""
    documents = cycle(tokenizer.encode(docs, prepend=tokenizer.get_bos_token_id()))

    def refill_buffer(buffer):
        buffer.extend(islice(documents, 16))
        return bool(buffer)

    return _bestfit_batches(B=B, T=T, device=device, buffer_size=200, refill_buffer=refill_buffer)


def test_bos_starts_every_row(tokenizer: HuggingFaceTokenizer) -> None:
    """Every row in the batch begins with BOS."""
    docs = ["Hello world! " * 10, "Python programming. " * 10, "The quick brown fox. " * 10]
    bos = tokenizer.get_bos_token_id()
    B, T = 4, 32
    loader = make_in_memory_loader(tokenizer, docs, B, T)
    for _ in range(3):
        x, _ = next(loader)
        for row in range(B):
            assert x[row, 0].item() == bos, f"Row {row} does not start with BOS"


def test_output_shapes(tokenizer: HuggingFaceTokenizer) -> None:
    """Inputs and targets have the expected shape."""
    docs = ["Hello world! " * 5]
    B, T = 2, 16
    loader = make_in_memory_loader(tokenizer, docs, B, T)
    x, y = next(loader)
    assert x.shape == (B, T), f"Expected ({B}, {T}), got {x.shape}"
    assert y.shape == (B, T), f"Expected ({B}, {T}), got {y.shape}"


def test_targets_are_shifted_inputs(tokenizer: HuggingFaceTokenizer) -> None:
    """Targets are inputs shifted by 1 position."""
    docs = ["Hello world! " * 5]
    B, T = 2, 32
    loader = make_in_memory_loader(tokenizer, docs, B, T)
    x, y = next(loader)
    torch.testing.assert_close(x[:, 1:], y[:, :-1])


def test_no_out_of_vocab_tokens(tokenizer: HuggingFaceTokenizer) -> None:
    """All token IDs are within [0, vocab_size)."""
    docs = ["Hello world! " * 5]
    B, T = 2, 32
    vocab_size = tokenizer.get_vocab_size()
    loader = make_in_memory_loader(tokenizer, docs, B, T)
    for _ in range(5):
        x, y = next(loader)
        assert (x >= 0).all() and (x < vocab_size).all()
        assert (y >= 0).all() and (y < vocab_size).all()


def test_full_utilisation(tokenizer: HuggingFaceTokenizer) -> None:
    """The loader produces full rows with no padding tokens."""
    docs = ["Hello world! " * 10]
    B, T = 2, 32
    loader = make_in_memory_loader(tokenizer, docs, B, T)
    x, _ = next(loader)
    for row in range(B):
        assert x[row].any(), f"Row {row} is all zeros (unfilled)"


def test_text_data_loader_reads_local_documents(tmp_path, tokenizer: HuggingFaceTokenizer) -> None:
    """The production text loader packs documents from a local file."""
    source = tmp_path / "train.txt"
    source.write_text("Hello world!\nPython is great.\n", encoding="utf-8")

    loader = text_data_loader(tokenizer, str(source), B=2, T=16, device="cpu")
    inputs, targets = next(loader)

    assert inputs.shape == (2, 16)
    assert targets.shape == (2, 16)
    assert (inputs[:, 0] == tokenizer.get_bos_token_id()).all()


def test_conversation_loader_pads_instead_of_cropping() -> None:
    class Tokenizer:
        def get_bos_token_id(self) -> int:
            return 0

        def render_conversation(self, conversation, max_tokens):
            return [1, 2, 3][:max_tokens], [0, 1, 1][:max_tokens]

    task = [{"messages": []}]
    loader = conversation_data_loader(Tokenizer(), task, B=1, T=4, device="cpu")
    inputs, targets = next(loader)

    assert inputs.shape == (1, 4)
    assert targets.shape == (1, 4)
    assert targets.tolist() == [[2, 3, -1, -1]]


def test_text_data_loader_rejects_empty_files(tmp_path, tokenizer: HuggingFaceTokenizer) -> None:
    source = tmp_path / "empty.txt"
    source.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no non-empty lines"):
        next(text_data_loader(tokenizer, str(source), B=1, T=4, device="cpu"))


def test_conversation_loader_skips_truncated_answers() -> None:
    class Tokenizer:
        def get_bos_token_id(self):
            return 0

        def render_conversation(self, conversation, max_tokens):
            return ([1, 2, 3], [0, 0, int(conversation["answer_fits"])])

    tokenizer = Tokenizer()
    task = [{"answer_fits": False}, {"answer_fits": True}]
    _, targets = next(conversation_data_loader(tokenizer, task, B=1, T=4, device="cpu"))
    assert targets.tolist() == [[-1, 3, -1, -1]]
    with pytest.raises(ValueError, match="No assistant targets"):
        next(conversation_data_loader(tokenizer, task[:1], B=1, T=4, device="cpu"))
