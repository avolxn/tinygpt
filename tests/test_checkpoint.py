"""
Tests for Hugging Face style model serialization helpers.
"""

import os
import tempfile

import torch

from tinygpt.checkpoint import (
    CONFIG_NAME,
    METADATA_NAME,
    SAFE_WEIGHTS_NAME,
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
        save_model_checkpoint(tmp, model, extra_meta={"step": 123, "tag": "roundtrip"})

        assert os.path.exists(os.path.join(tmp, CONFIG_NAME))
        assert os.path.exists(os.path.join(tmp, SAFE_WEIGHTS_NAME))
        assert os.path.exists(os.path.join(tmp, METADATA_NAME))

        loaded_model, metadata = build_model_from_checkpoint(tmp, torch.device("cpu"), phase="eval")
        assert metadata["step"] == 123
        assert metadata["tag"] == "roundtrip"
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
        save_model_checkpoint(os.path.join(run_dir, "checkpoint-5"), model, extra_meta={"step": 5})
        save_model_checkpoint(os.path.join(run_dir, "checkpoint-10"), model, extra_meta={"step": 10})

        resolved = resolve_model_directory(run_dir)
        assert resolved.endswith("checkpoint-10")

        _, metadata = build_model_from_checkpoint(run_dir, torch.device("cpu"), phase="train")
        assert metadata["step"] == 10
