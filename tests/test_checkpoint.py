"""
Tests for Hugging Face style model serialization helpers.
"""

import os
import tempfile

import torch

from tinygpt.checkpoint import (
    CONFIG_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    build_model_from_checkpoint,
    get_checkpoint_dir,
    resolve_model_directory,
    save_model_checkpoint,
)
from tinygpt.config import make_config
from tinygpt.model import GPT


def make_test_model() -> GPT:
    config = make_config(
        depth=2,
        aspect_ratio=8,
        head_dim=8,
        vocab_size=32,
        sequence_len=16,
        window_pattern="SL",
    )
    model = GPT(config)
    model.init_weights()
    return model


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
