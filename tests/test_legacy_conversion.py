"""
Tests for legacy checkpoint conversion into Hugging Face model files.
"""

import json
import os
import tempfile
from dataclasses import asdict

import torch
from scripts.convert import convert_legacy_model_to_hf

from tinygpt.checkpoint import build_model_from_checkpoint
from tinygpt.config import make_config
from tinygpt.model import GPT


def test_convert_legacy_model_to_hf_patches_missing_keys() -> None:
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

    legacy_state = {
        key: value for key, value in model.state_dict().items() if key not in {"resid_lambdas", "x0_lambdas"}
    }
    legacy_meta = {
        "step": 123,
        "model_config": {key: value for key, value in asdict(config).items() if key != "window_pattern"},
        "tag": "legacy",
    }

    with tempfile.TemporaryDirectory() as tmp:
        torch.save(legacy_state, os.path.join(tmp, "model_123.pt"))
        with open(os.path.join(tmp, "meta_123.json"), "w", encoding="utf-8") as f:
            json.dump(legacy_meta, f, indent=2)

        convert_legacy_model_to_hf(tmp, tmp, step=123)
        converted_model, metadata = build_model_from_checkpoint(tmp, torch.device("cpu"), phase="eval")

        assert metadata["step"] == 123
        assert converted_model.config.window_pattern == "L"
        torch.testing.assert_close(converted_model.resid_lambdas, torch.ones_like(converted_model.resid_lambdas))
        torch.testing.assert_close(converted_model.x0_lambdas, torch.zeros_like(converted_model.x0_lambdas))
