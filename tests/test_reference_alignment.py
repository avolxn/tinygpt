import math

import torch

from tinygpt.config import GPTConfig, compute_scaled_total_batch_size, make_config
from tinygpt.dataloader import CLIMBMIX_DATASET, resolve_dataset_source
from tinygpt.model import GPT
from tinygpt.sft_tasks import build_sft_task_lists


def test_climbmix_source_uses_last_shard_for_validation(monkeypatch):
    files = [
        ".gitattributes",
        "README.md",
        "shard_00000.parquet",
        "shard_00001.parquet",
        "shard_00002.parquet",
    ]

    monkeypatch.setattr("tinygpt.dataloader.list_repo_files", lambda *args, **kwargs: files)

    train_name, train_split, train_files = resolve_dataset_source(CLIMBMIX_DATASET, "train")
    val_name, val_split, val_files = resolve_dataset_source(CLIMBMIX_DATASET, "val")

    assert train_name == "parquet"
    assert train_split == "train"
    assert train_files == {
        "train": [
            "hf://datasets/karpathy/climbmix-400b-shuffle/shard_00000.parquet",
            "hf://datasets/karpathy/climbmix-400b-shuffle/shard_00001.parquet",
        ]
    }
    assert val_name == "parquet"
    assert val_split == "train"
    assert val_files == {"train": ["hf://datasets/karpathy/climbmix-400b-shuffle/shard_00002.parquet"]}


def test_all_ignored_targets_return_zero_loss_not_nan():
    config = GPTConfig(sequence_len=8, vocab_size=64, n_layer=1, n_head=1, n_kv_head=1, n_embd=32, window_pattern="L")
    model = GPT(config)
    model.init_weights()
    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.full((2, 4), -1, dtype=torch.long)

    loss = model(idx, targets)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_identity_only_sft_validation_uses_identity_dataset(tmp_path):
    identity_path = tmp_path / "identity.jsonl"
    identity_path.write_text(
        '[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello."}]\n',
        encoding="utf-8",
    )

    train_tasks, val_tasks = build_sft_task_lists(
        {"identity"},
        identity_conversations=str(identity_path),
        mmlu_epochs=3,
        gsm8k_epochs=4,
    )

    assert len(train_tasks) == 2
    assert len(val_tasks) == 1
    assert all(type(task).__name__ == "CustomJSON" for task in train_tasks + val_tasks)


def test_default_sft_validation_matches_reference_core_mix():
    _, val_tasks = build_sft_task_lists(
        {"smoltalk", "mmlu", "gsm8k", "identity", "spelling"},
        identity_conversations="missing.jsonl",
        mmlu_epochs=3,
        gsm8k_epochs=4,
    )

    assert [type(task).__name__ for task in val_tasks] == ["SmolTalk", "MMLU", "GSM8K"]


def test_reference_batch_size_scaling_matches_reference_formula():
    d12 = make_config(12, vocab_size=32768)
    d24 = make_config(24, vocab_size=32768)
    batch = compute_scaled_total_batch_size(
        scaling_params=1_000_000,
        d12_scaling_params=500_000,
        target_param_data_ratio=12,
        requested_total_batch_size=-1,
    )
    expected = 2 ** round(math.log2((2**19) * ((12 * 1_000_000) / (12 * 500_000)) ** 0.383))

    assert batch == expected
    assert d12.n_embd == 768
    assert d24.n_embd == 1536
