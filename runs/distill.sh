#!/usr/bin/env bash
set -euo pipefail

# Teacher-to-student pipeline:
# 1. download and prepare the Hugging Face teacher
# 2. pretrain a student with the teacher tokenizer
# 3. distill the teacher into the student
# 4. evaluate the distilled checkpoint on CUDA
#
# From repo root:
#   TEACHER_MODEL=<org>/<model> bash runs/distill.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1
mkdir -p data

command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv

MLFLOW_RUN="${MLFLOW_RUN:-distill}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
TEACHER_MODEL="${TEACHER_MODEL:?Set TEACHER_MODEL to a Hugging Face model ID}"
TEACHER_DIR="${TEACHER_DIR:-data/teacher}"
STUDENT_RUN="${STUDENT_RUN:-student}"
DISTILL_RUN="${DISTILL_RUN:-distill}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TEACHER_DEVICE="${TEACHER_DEVICE:-same}"

if [ "$DEVICE_TYPE" = "cuda" ]; then
  uv sync --extra gpu
else
  uv sync --extra cpu
fi
# shellcheck source=/dev/null
source .venv/bin/activate

if [ ! -f "$TEACHER_DIR/config.json" ] || [ ! -f "$TEACHER_DIR/tokenizer.json" ]; then
  python -m scripts.prepare_teacher \
    --model "$TEACHER_MODEL" \
    --out-dir "$TEACHER_DIR"
else
  echo "==> Reusing prepared teacher at $TEACHER_DIR"
fi

echo "==> Pretraining student with teacher tokenizer"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.pretrain \
  --teacher-model "$TEACHER_DIR" \
  --device-type "$DEVICE_TYPE" \
  --depth 20 \
  --target-param-data-ratio 12 \
  --device-batch-size 32 \
  --run "$MLFLOW_RUN" \
  --run-name "$STUDENT_RUN" \
  --out-dir data

if [ ! -f "data/identity_conversations.jsonl" ]; then
  curl -fsSL --retry 3 \
    -o data/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

echo "==> Distilling teacher into data/pretrain_checkpoints/$STUDENT_RUN"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.distill \
  --checkpoint "data/pretrain_checkpoints/$STUDENT_RUN" \
  --teacher-model "$TEACHER_DIR" \
  --device-type "$DEVICE_TYPE" \
  --teacher-device "$TEACHER_DEVICE" \
  --eval-every 500 \
  --distill-alpha 0.75 \
  --distill-temperature 1.5 \
  --tasks smoltalk,mmlu,gsm8k,identity,spelling \
  --identity-conversations data/identity_conversations.jsonl \
  --run "$MLFLOW_RUN" \
  --run-name "$DISTILL_RUN" \
  --out-dir data

if [ "${DEVICE_TYPE:-cuda}" = "cuda" ]; then
  echo "==> Evaluating data/distill_checkpoints/$DISTILL_RUN"
  python -m scripts.evaluate_model \
    --checkpoint "data/distill_checkpoints/$DISTILL_RUN" \
    --eval chat \
    --device-batch-size 32 \
    --vllm-tensor-parallel-size "$NPROC_PER_NODE" \
    --vllm-model "data/distill_checkpoints/$DISTILL_RUN"
fi
