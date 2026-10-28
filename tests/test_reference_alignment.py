import copy
import math
import sys
from types import SimpleNamespace

import pytest
import torch
from tasks.distillation import build_distillation_tasks
from transformers import TrainerCallback, TrainingArguments

from tinygpt.config import compute_scaled_total_batch_size, compute_scaled_weight_decay, make_config
from tinygpt.dataloader import CLIMBMIX_DATASET, resolve_dataset_source
from tinygpt.metrics import evaluate_loss
from tinygpt.model import TinyGPTForCausalLM
from tinygpt.optimizer import get_lr_multiplier
from tinygpt.train import TinyGPTTrainer, causal_lm_loss


def _fsdp_optimizer_worker(rank, rendezvous):
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from tinygpt.optimizer import make_optimizer

    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        reference = TinyGPTForCausalLM(make_config(3, aspect_ratio=16, head_dim=16, vocab_size=64, sequence_len=8))
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.03)
        model = copy.deepcopy(reference)
        mesh = init_device_mesh("cpu", (2,))
        for layer in model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        optimizer = make_optimizer(model, weight_decay=0.0)
        expected_optimizer = make_optimizer(reference, weight_decay=0.0)
        inputs = torch.randint(0, 64, (2, 8))
        for iteration in range(2):
            model(inputs[rank : rank + 1]).logits.square().mean().backward()
            reference(inputs).logits.square().mean().backward()
            optimizer.step()
            expected_optimizer.step()
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(
                    parameter.full_tensor(),
                    dict(reference.named_parameters())[name],
                    rtol=2e-4,
                    atol=2e-5,
                    msg=f"{iteration}: {name}",
                )
            optimizer.zero_grad()
            expected_optimizer.zero_grad()
            if iteration == 0:
                state = get_optimizer_state_dict(
                    model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
                )
                optimizer = make_optimizer(model, weight_decay=0.0)
                set_optimizer_state_dict(
                    model, optimizer, state, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
                )
    finally:
        torch.distributed.destroy_process_group()


def test_fsdp2_muon_matches_unsharded_training_and_resumes(tmp_path):
    torch.multiprocessing.spawn(
        _fsdp_optimizer_worker,
        args=((tmp_path / "fsdp-rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


def _distributed_loss_worker(rank, rendezvous):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[2.0, 0.0], [0.0, 1.0]]))

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.weight[input_ids])

    torch.distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        inputs = torch.tensor([[rank, rank]])
        labels = torch.tensor([[1, -1]]) if rank == 0 else torch.tensor([[0, 0]])
        actual = evaluate_loss(Model(), iter([(inputs, labels)]), 1)
        expected = (math.log(math.exp(2) + 1) + 2 * math.log(1 + math.e)) / 3
        assert actual == pytest.approx(expected)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(sys.platform != "linux", reason="Distributed CPU integration runs on Linux CI")
def test_distributed_validation_loss(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_loss_worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


def test_validation_loss_weights_supervised_tokens():
    model = TinyGPTForCausalLM(make_config(1, aspect_ratio=8, head_dim=8, vocab_size=32, sequence_len=8))
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

    model = TinyGPTForCausalLM(make_config(1, aspect_ratio=8, head_dim=8, vocab_size=32, sequence_len=8))
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
    model = TinyGPTForCausalLM(config)
    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.full((2, 4), -1, dtype=torch.long)

    loss = causal_lm_loss(model, idx, targets)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_custom_distillation_uses_local_conversations(tmp_path):
    identity_path = tmp_path / "identity.jsonl"
    identity_path.write_text(
        '[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello."}]\n',
        encoding="utf-8",
    )

    train_tasks, val_tasks = build_distillation_tasks(
        {"custom"},
        conversations=str(identity_path),
        mmlu_epochs=3,
        gsm8k_epochs=4,
    )

    assert len(train_tasks) == 1
    assert len(val_tasks) == 1
    assert all(type(task).__name__ == "CustomJSON" for task in train_tasks + val_tasks)


def test_default_distillation_validation_matches_reference_core_mix():
    _, val_tasks = build_distillation_tasks(
        {"smoltalk", "mmlu", "gsm8k"},
        conversations="missing.jsonl",
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


def test_scheduler_handles_zero_warmup_and_warmdown():
    assert get_lr_multiplier(0, 10, warmup_steps=0, warmdown_ratio=0) == 1.0
    assert get_lr_multiplier(10, 10, warmup_steps=0, warmdown_ratio=0.5, final_lr_frac=0.1) == 0.1

    with pytest.raises(ValueError, match="warmdown_ratio"):
        get_lr_multiplier(0, 10, warmup_steps=0, warmdown_ratio=1.1)
