"""
Tests for Hugging Face style model serialization helpers.
"""

import argparse
import os
import tempfile

import torch
from transformers import LlamaForCausalLM

from tinygpt.checkpoint import (
    CONFIG_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    build_checkpoint_metadata,
    build_model_from_checkpoint,
    get_checkpoint_dir,
    resolve_model_directory,
    resolve_trainer_checkpoint,
    save_model_checkpoint,
)
from tinygpt.config import RuntimeConfig, make_config


def make_test_model() -> LlamaForCausalLM:
    config = make_config(
        depth=2,
        aspect_ratio=8,
        head_dim=8,
        vocab_size=32,
        sequence_len=16,
    )
    return LlamaForCausalLM(config)


def test_save_and_load_model_checkpoint_roundtrip() -> None:
    model = make_test_model()

    with tempfile.TemporaryDirectory() as tmp:
        save_model_checkpoint(tmp, model)

        assert os.path.exists(os.path.join(tmp, CONFIG_NAME))
        assert os.path.exists(os.path.join(tmp, SAFE_WEIGHTS_NAME))
        assert not os.path.exists(os.path.join(tmp, TRAINER_STATE_NAME))

        loaded_model, metadata = build_model_from_checkpoint(tmp, torch.device("cpu"), phase="eval")
        assert "step" not in metadata
        assert loaded_model.training is False

        for (expected_name, expected_tensor), (actual_name, actual_tensor) in zip(
            model.state_dict().items(),
            loaded_model.state_dict().items(),
            strict=True,
        ):
            assert expected_name == actual_name
            torch.testing.assert_close(expected_tensor, actual_tensor)


def test_save_model_checkpoint_copies_tokenizer_files() -> None:
    model = make_test_model()

    with tempfile.TemporaryDirectory() as tmp:
        tokenizer_dir = os.path.join(tmp, "tokenizer")
        out_dir = os.path.join(tmp, "checkpoint")
        os.makedirs(tokenizer_dir)
        with open(os.path.join(tokenizer_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        torch.save(torch.tensor([0, 1], dtype=torch.int32), os.path.join(tokenizer_dir, "token_bytes.pt"))

        save_model_checkpoint(out_dir, model, tokenizer_dir=tokenizer_dir)

        assert os.path.exists(os.path.join(out_dir, "tokenizer.json"))
        assert os.path.exists(os.path.join(out_dir, "token_bytes.pt"))


def test_resolve_model_directory_prefers_latest_trainer_checkpoint() -> None:
    model = make_test_model()

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = get_checkpoint_dir(tmp, "demo")
        checkpoint_5 = os.path.join(run_dir, "checkpoint-5")
        checkpoint_10 = os.path.join(run_dir, "checkpoint-10")
        save_model_checkpoint(checkpoint_5, model)
        save_model_checkpoint(checkpoint_10, model)
        with open(os.path.join(checkpoint_5, TRAINER_STATE_NAME), "w", encoding="utf-8") as f:
            f.write('{"global_step": 5}')
        with open(os.path.join(checkpoint_10, TRAINER_STATE_NAME), "w", encoding="utf-8") as f:
            f.write('{"global_step": 10}')

        resolved = resolve_model_directory(run_dir)
        assert resolved.endswith("checkpoint-10")

        _, metadata = build_model_from_checkpoint(run_dir, torch.device("cpu"), phase="train")
        assert metadata["step"] == 10


def test_build_checkpoint_metadata_captures_runtime_context() -> None:
    args = argparse.Namespace(seed=7, value="demo")
    runtime_config = RuntimeConfig(seed=7, run_name="demo")

    metadata = build_checkpoint_metadata(
        phase="pretrain",
        args=args,
        runtime_config=runtime_config,
        total_batch_size=128,
    )

    assert metadata["phase"] == "pretrain"
    assert metadata["user_config"] == {"seed": 7, "value": "demo"}
    assert metadata["runtime_config"]["seed"] == 7
    assert metadata["derived_values"] == {"total_batch_size": 128}
    assert metadata["command"]
    assert metadata["torch_version"] == torch.__version__


def test_resolve_trainer_checkpoint_requires_training_state() -> None:
    model = make_test_model()

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = os.path.join(tmp, "checkpoint-10")
        save_model_checkpoint(checkpoint, model)
        assert resolve_trainer_checkpoint(checkpoint) is None

        for filename in (TRAINER_STATE_NAME, "optimizer.pt", "scheduler.pt"):
            open(os.path.join(checkpoint, filename), "w", encoding="utf-8").close()
        assert resolve_trainer_checkpoint(checkpoint) == checkpoint
