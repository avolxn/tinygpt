import math

import pytest
import torch
from tasks.distillation import build_distillation_tasks
from tasks.spellingbee import SimpleSpelling, SpellingBee
from transformers import LlamaForCausalLM, TrainerCallback, TrainingArguments

from tinygpt.config import compute_scaled_total_batch_size, compute_scaled_weight_decay, make_config
from tinygpt.dataloader import CLIMBMIX_DATASET, resolve_dataset_source
from tinygpt.metrics import evaluate_loss
from tinygpt.scheduler import get_lr_multiplier
from tinygpt.train import TinyGPTTrainer, causal_lm_loss


def test_validation_loss_weights_supervised_tokens():
    model = LlamaForCausalLM(make_config(1, aspect_ratio=8, head_dim=8, vocab_size=32, sequence_len=8))
    model.eval()
    inputs = torch.tensor([[1, 2, 3, 4]])
    labels = [torch.tensor([[2, -1, -1, -1]]), torch.tensor([[2, 3, 4, 5]])]
    expected = sum(causal_lm_loss(model, inputs, y, reduction="sum").item() for y in labels) / 5
    actual = evaluate_loss(model, iter((inputs, y) for y in labels), 2)
    assert actual == pytest.approx(expected)
    with pytest.raises(ValueError, match="no supervised tokens"):
        evaluate_loss(model, iter([(inputs, torch.full_like(inputs, -1))]), 1)


def test_evaluation_preserves_mode_and_calls_callbacks(tmp_path):
    class EvaluationCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            assert metrics == {"eval_loss": 1.25}
            control.should_training_stop = True
            return control

    model = LlamaForCausalLM(make_config(1, aspect_ratio=8, head_dim=8, vocab_size=32, sequence_len=8))
    model.eval()
    trainer = TinyGPTTrainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[], disable_tqdm=True),
        matrix_lr=0.01,
        embedding_lr=0.01,
        scalar_lr=0.01,
        train_loader=iter([]),
        eval_fn=lambda model, step: {"loss": 1.25},
        callbacks=[EvaluationCallback()],
    )
    assert trainer.evaluate() == {"eval_loss": 1.25}
    assert not model.training
    assert trainer.control.should_training_stop


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
    config = make_config(1, aspect_ratio=32, head_dim=32, vocab_size=64, sequence_len=8)
    model = LlamaForCausalLM(config)
    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.full((2, 4), -1, dtype=torch.long)

    loss = causal_lm_loss(model, idx, targets)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_identity_only_distillation_validation_uses_identity_dataset(tmp_path):
    identity_path = tmp_path / "identity.jsonl"
    identity_path.write_text(
        '[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello."}]\n',
        encoding="utf-8",
    )

    train_tasks, val_tasks = build_distillation_tasks(
        {"identity"},
        identity_conversations=str(identity_path),
        mmlu_epochs=3,
        gsm8k_epochs=4,
    )

    assert len(train_tasks) == 2
    assert len(val_tasks) == 1
    assert all(type(task).__name__ == "CustomJSON" for task in train_tasks + val_tasks)


def test_default_distillation_validation_matches_reference_core_mix():
    _, val_tasks = build_distillation_tasks(
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
    assert d12.hidden_size == 768
    assert d24.hidden_size == 1536


def test_scaling_rejects_zero_reference_counts():
    with pytest.raises(ValueError, match="positive"):
        compute_scaled_total_batch_size(
            scaling_params=0,
            d12_scaling_params=1,
            target_param_data_ratio=12,
            requested_total_batch_size=-1,
        )
    with pytest.raises(ValueError, match="positive"):
        compute_scaled_weight_decay(
            base_weight_decay=0.1,
            total_batch_size=1,
            target_tokens=0,
            d12_target_tokens=1,
        )


def test_spelling_tasks_reject_non_positive_sizes():
    with pytest.raises(ValueError, match="size must be positive"):
        SimpleSpelling(size=0)
    with pytest.raises(ValueError, match="size must be positive"):
        SpellingBee(size=0)


def test_scheduler_handles_zero_warmup_and_warmdown():
    assert get_lr_multiplier(0, 10, warmup_steps=0, warmdown_ratio=0) == 1.0
    assert get_lr_multiplier(10, 10, warmup_steps=0, warmdown_ratio=0.5, final_lr_frac=0.1) == 0.1

    with pytest.raises(ValueError, match="warmdown_ratio"):
        get_lr_multiplier(0, 10, warmup_steps=0, warmdown_ratio=1.1)
