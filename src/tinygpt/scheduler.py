"""
Learning rate scheduler: linear warmup + linear decay.

Applied to `initial_lr` stored in each optimizer param group.
"""


def get_lr_multiplier(
    step: int,
    num_steps: int,
    warmup_steps: int,
    warmdown_ratio: float = 0.65,
    final_lr_frac: float = 0.05,
) -> float:
    """
    Compute the LR multiplier for a given step.

    Schedule:
      [0, warmup_steps)           : linear ramp 0 → 1
      [warmup_steps, decay_start] : constant 1
      [decay_start, num_steps]    : linear decay from 1 → final_lr_frac

    Args:
        step: current optimization step (0-indexed)
        num_steps: total number of optimization steps
        warmup_steps: number of warmup steps
        warmdown_ratio: fraction of num_steps used for linear decay
        final_lr_frac: LR at the end as a fraction of peak LR

    Returns:
        Multiplier in (0, 1] to be applied to each group's initial_lr.
    """
    warmdown_steps = round(warmdown_ratio * num_steps)
    decay_start = num_steps - warmdown_steps

    if step < warmup_steps:
        return (step + 1) / warmup_steps
    elif step <= decay_start:
        return 1.0
    else:
        progress = (num_steps - step) / warmdown_steps  # 1.0 at decay_start → 0.0 at num_steps
        return final_lr_frac + (1.0 - final_lr_frac) * progress
