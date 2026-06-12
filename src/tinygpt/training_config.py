"""Training defaults shared by scripts.

The formulas intentionally mirror nanochat's depth-driven pretraining setup.
"""

from __future__ import annotations

import math

NANOCHAT_REFERENCE_BATCH_SIZE = 2**19


def compute_nanochat_total_batch_size(
    *,
    scaling_params: int,
    d12_scaling_params: int,
    target_param_data_ratio: float,
    requested_total_batch_size: int,
) -> int:
    """Return the total token batch size using nanochat's scaling law."""
    if requested_total_batch_size > 0:
        return requested_total_batch_size
    target_tokens = target_param_data_ratio * scaling_params
    d12_target_tokens = target_param_data_ratio * d12_scaling_params
    batch_size_ratio = target_tokens / d12_target_tokens
    predicted_batch_size = NANOCHAT_REFERENCE_BATCH_SIZE * batch_size_ratio**0.383
    return 2 ** round(math.log2(predicted_batch_size))


def compute_nanochat_weight_decay(
    *,
    base_weight_decay: float,
    total_batch_size: int,
    target_tokens: int,
    d12_target_tokens: float,
) -> float:
    """Return nanochat's scaled pretraining weight decay."""
    return base_weight_decay * math.sqrt(total_batch_size / NANOCHAT_REFERENCE_BATCH_SIZE) * (
        d12_target_tokens / target_tokens
    )
