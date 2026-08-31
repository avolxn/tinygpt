#!/usr/bin/env bash
set -euo pipefail

# Distillation run:
# 1. convert the reference teacher locally
# 2. load the student trained with the same tokenizer
# 3. run online KL + CE distillation on chat tasks
# 4. run a chat eval pass
#
# Important: the student checkpoint should come from
# runs/pretrain_reference_d32.sh.
#
# From repo root:
#   bash runs/distill_reference_d32.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1
mkdir -p data

command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
# shellcheck source=/dev/null
source .venv/bin/activate

MLFLOW_RUN="${MLFLOW_RUN:-distill_reference_d32}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TEACHER_DEVICE="${TEACHER_DEVICE:-same}"
REFERENCE_MODEL="${REFERENCE_MODEL:-karpathy/nanochat-d32}"

python -m scripts.check_mlflow

if [ ! -d "data/pretrain_checkpoints/pretrain_reference_d32" ]; then
  echo "Student checkpoint not found: data/pretrain_checkpoints/pretrain_reference_d32"
  echo "Run bash runs/pretrain_reference_d32.sh first."
  exit 1
fi

if [ ! -f "data/teacher_reference_d32/config.json" ] || [ ! -f "data/teacher_reference_d32/model.safetensors" ]; then
  echo "==> Converting reference teacher"
  python -m scripts.convert \
    --input "$REFERENCE_MODEL" \
    --out-dir data/teacher_reference_d32
else
  echo "==> Reusing converted teacher at data/teacher_reference_d32"
fi

if [ ! -f "data/identity_conversations.jsonl" ]; then
  curl -fsSL -o "data/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

echo "==> Distilling data/pretrain_checkpoints/pretrain_reference_d32 from data/teacher_reference_d32"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.distill \
  --checkpoint data/pretrain_checkpoints/pretrain_reference_d32 \
  --tokenizer-dir data/tokenizer_reference_d32 \
  --teacher-model data/teacher_reference_d32 \
  --teacher-tokenizer data/teacher_reference_d32 \
  --teacher-device "$TEACHER_DEVICE" \
  --eval-every 500 \
  --distill-alpha 0.75 \
  --distill-temperature 1.5 \
  --tasks smoltalk,mmlu,gsm8k,identity,spelling \
  --identity-conversations data/identity_conversations.jsonl \
  --run "$MLFLOW_RUN" \
  --run-name distill_reference_d32 \
  --out-dir data

echo "==> Evaluating distilled checkpoint data/distill_checkpoints/distill_reference_d32"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.evaluate_model \
  --checkpoint data/distill_checkpoints/distill_reference_d32 \
  --tokenizer-dir data/tokenizer_reference_d32 \
  --eval chat \
  --device-batch-size 32 \
  --max-problems 64
