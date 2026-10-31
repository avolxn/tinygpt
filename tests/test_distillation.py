"""Tests for online distillation helpers."""

import pytest
import torch
from tests.helpers import SPECIAL_TOKENS

from tinygpt.distillation import masked_distillation_loss, validate_teacher_tokenizer_compatibility


class FakeTokenizer:
    def __init__(self, vocab_size: int = 32, offset: int = 0) -> None:
        self._vocab_size = vocab_size
        self._specials = {token: i + offset for i, token in enumerate(SPECIAL_TOKENS)}
        self.eos_token_id = offset

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def encode_special(self, text: str) -> int | None:
        return self._specials.get(text)

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % 17 for ch in text]


def test_masked_distillation_loss_zero_for_matching_logits() -> None:
    logits = torch.randn(2, 3, 5)
    labels = torch.tensor([[1, 2, -1], [3, 4, 0]])
    loss = masked_distillation_loss(logits, logits.clone(), labels, temperature=1.0)
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_masked_distillation_loss_ignores_masked_positions() -> None:
    student = torch.tensor([[[3.0, 1.0], [0.0, 0.0]]])
    teacher = torch.tensor([[[1.0, 3.0], [9.0, -9.0]]])
    labels = torch.tensor([[1, -1]])

    masked = masked_distillation_loss(student, teacher, labels, temperature=1.0)
    expected = masked_distillation_loss(student[:, :1], teacher[:, :1], torch.tensor([[1]]), temperature=1.0)
    torch.testing.assert_close(masked, expected)


def test_chunked_distillation_matches_full_loss_and_gradients() -> None:
    student = torch.randn(2, 1050, 7, requires_grad=True)
    teacher = torch.randn_like(student)
    labels = torch.randint(-1, 7, (2, 1050))
    temperature = 1.5
    actual = masked_distillation_loss(student, teacher, labels, temperature)
    per_token = torch.nn.functional.kl_div(
        torch.log_softmax(student / temperature, -1), torch.softmax(teacher / temperature, -1), reduction="none"
    ).sum(-1)
    expected = per_token[labels != -1].mean() * temperature**2
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, student, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, student)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_validate_teacher_tokenizer_compatibility_accepts_matching_tokenizers() -> None:
    student = FakeTokenizer(vocab_size=64)
    teacher = FakeTokenizer(vocab_size=64)
    validate_teacher_tokenizer_compatibility(student, teacher)


def test_validate_teacher_tokenizer_compatibility_rejects_vocab_mismatch() -> None:
    student = FakeTokenizer(vocab_size=64)
    teacher = FakeTokenizer(vocab_size=65)

    with pytest.raises(ValueError, match="vocab sizes differ"):
        validate_teacher_tokenizer_compatibility(student, teacher)


def test_validate_teacher_tokenizer_compatibility_rejects_special_token_mismatch() -> None:
    student = FakeTokenizer(vocab_size=64)
    teacher = FakeTokenizer(vocab_size=64, offset=1)

    with pytest.raises(ValueError, match="tokenizers differ in eos_token_id"):
        validate_teacher_tokenizer_compatibility(student, teacher)


@pytest.mark.parametrize("architecture", ["mistral", "qwen3"])
def test_native_teacher_distillation_with_padded_vocabulary(tmp_path, architecture):
    from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments

    from tinygpt.config import make_config
    from tinygpt.model import TinyGPTForCausalLM
    from tinygpt.train import TinyGPTTrainer

    config = AutoConfig.for_model(
        architecture,
        vocab_size=80,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
    )
    teacher = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
    student = TinyGPTForCausalLM(make_config(1, aspect_ratio=32, head_dim=8, vocab_size=64, sequence_len=16))
    trainer = TinyGPTTrainer(
        model=student,
        teacher_model=teacher,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
        matrix_lr=0.02,
        embedding_lr=0.3,
        scalar_lr=0.5,
        train_loader=iter([]),
        distill_alpha=0.75,
        distill_temperature=1.5,
    )
    inputs = torch.randint(0, 64, (1, 8))
    labels = inputs.roll(-1, dims=1)
    labels[:, :3] = -1
    loss = trainer.compute_loss(student, {"input_ids": inputs, "labels": labels})
    assert torch.isfinite(loss)
    loss.backward()
    assert student.lm_head.weight.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_teacher_snapshot_keeps_native_weights_and_tokenizer(tmp_path, monkeypatch):
    import json
    import runpy
    import shutil
    import sys
    from types import SimpleNamespace

    from tests.helpers import make_test_tokenizer
    from transformers import AutoModelForCausalLM, MistralConfig

    from tinygpt.tokenizer import HuggingFaceTokenizer

    source = tmp_path / "hub"
    target = tmp_path / "teacher"
    tokenizer = make_test_tokenizer(["Hello world"])
    teacher = AutoModelForCausalLM.from_config(
        MistralConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    teacher.save_pretrained(source)
    tokenizer.save(str(source))

    def download(model, *, revision, local_dir, allow_patterns):
        assert model == "example/teacher" and revision == "pinned-revision"
        shutil.copytree(source, local_dir, dirs_exist_ok=True)

    monkeypatch.setattr("huggingface_hub.model_info", lambda *a, **kw: SimpleNamespace(sha="pinned-revision"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr(sys, "argv", ["prepare_teacher", "--model", "example/teacher", "--out-dir", str(target)])
    runpy.run_module("scripts.prepare_teacher", run_name="__main__")
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        assert (source / name).read_bytes() == (target / name).read_bytes()
    validate_teacher_tokenizer_compatibility(tokenizer, HuggingFaceTokenizer.from_directory(str(target)))
    assert json.loads((target / "teacher_source.json").read_text())["revision"] == "pinned-revision"


def test_changed_chat_template_requires_new_student():
    from tests.helpers import make_test_tokenizer

    student = make_test_tokenizer(["Hello world"])
    teacher = make_test_tokenizer(["Hello world"])
    teacher.tokenizer.chat_template += "modified"
    with pytest.raises(ValueError, match="chat_template"):
        validate_teacher_tokenizer_compatibility(student, teacher)
