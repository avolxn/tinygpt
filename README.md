# tinygpt

`tinygpt` is a compact training and evaluation stack for small GPT-style models.
The repository is organized around four opinionated workflows:

1. Full training from scratch: tokenizer training, pretraining, SFT, and evaluation.
2. Pretraining with the `karpathy/nanochat-d32` tokenizer.
3. Distillation from `karpathy/nanochat-d32` into a student trained with the same tokenizer.
4. A smoke test for CPU or a small GPU.

## Repository Workflows

| Workflow | Script | What it does |
| --- | --- | --- |
| From scratch | `runs/from_scratch.sh` | Trains a tokenizer, pretrains a base model, runs SFT, then evaluates the result. |
| Nanochat tokenizer pretrain | `runs/pretrain_with_nanochat_d32.sh` | Reuses the `karpathy/nanochat-d32` tokenizer and runs pretraining only. |
| Distillation | `runs/distill_from_nanochat_d32.sh` | Distills from `karpathy/nanochat-d32` into a student checkpoint produced by `pretrain_with_nanochat_d32.sh`. |
| Smoke test | `runs/smoke.sh` | Runs a minimal end-to-end validation path on CPU or a small GPU. |

## Recommended Usage

Run commands from the `tinygpt` root:

```bash
bash runs/from_scratch.sh
bash runs/pretrain_with_nanochat_d32.sh
bash runs/distill_from_nanochat_d32.sh
bash runs/smoke.sh
```

## Storage Layout

All generated artifacts and support files are stored under `data/`.

Typical outputs:

- `data/tokenizer_from_scratch`
- `data/tokenizer_nanochat_d32`
- `data/teacher_nanochat_d32`
- `data/tokenizer_smoke`
- `data/pretrain_checkpoints/from_scratch`
- `data/pretrain_checkpoints/pretrain_with_nanochat_d32`
- `data/distill_checkpoints/distill_from_nanochat_d32`
- `data/sft_checkpoints/from_scratch`
- `data/sft_checkpoints/smoke`
- `data/identity_conversations.jsonl`

## Runtime Overrides

The run scripts are intentionally simple. Only a small number of environment overrides are supported:

- `WANDB_RUN`: Weights & Biases run name. If unset, scripts default to `dummy`.
- `NPROC_PER_NODE`: Number of `torchrun` processes per node for GPU workflows.
- `DEVICE_TYPE`: Runtime override for `runs/smoke.sh`, typically `cpu`, `cuda`, or `mps`.
- `TEACHER_DEVICE`: Teacher placement override for `runs/distill_from_nanochat_d32.sh`.

Examples:

```bash
WANDB_RUN=from_scratch_exp bash runs/from_scratch.sh
WANDB_RUN=student_d32 bash runs/pretrain_with_nanochat_d32.sh
WANDB_RUN=distill_d32 TEACHER_DEVICE=cpu bash runs/distill_from_nanochat_d32.sh
DEVICE_TYPE=cpu bash runs/smoke.sh
```

## Important Constraint

Online KL distillation in this codebase requires tokenizer compatibility between teacher and student.
In practice, the distillation workflow assumes:

- the teacher is `karpathy/nanochat-d32`
- the student was pretrained with `runs/pretrain_with_nanochat_d32.sh`

If the student uses a different tokenizer or token ID mapping, distillation will fail by design.

## Python Entry Points

Primary modules:

- `python -m scripts.train_tokenizer`
- `python -m scripts.pretrain`
- `python -m scripts.finetune`
- `python -m scripts.distill`
- `python -m scripts.evaluate_tokenizer`
- `python -m scripts.evaluate_model`
- `python -m scripts.chat`

Defaults are aligned with the `data/` directory layout used by the run scripts.

## Environment

Expected baseline:

- Python 3.12+
- `uv` for environment setup
- PyTorch-compatible CPU, CUDA, or MPS runtime

The run scripts create or reuse `.venv` and install dependencies via `uv sync`.
