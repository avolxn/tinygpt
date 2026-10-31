#!/usr/bin/env bash
set -euo pipefail


ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1
mkdir -p data

MLFLOW_RUN="${MLFLOW_RUN:-distill}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
TEACHER_MODEL="${TEACHER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
TEACHER_DIR="${TEACHER_DIR:-data/teachers/${TEACHER_MODEL//\//--}}"
STUDENT_RUN="${STUDENT_RUN:-student}"
DISTILL_RUN="${DISTILL_RUN:-distill}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TEACHER_DEVICE="${TEACHER_DEVICE:-same}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-4}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
PRETRAIN_ARGS=()
if [ "${FP8:-1}" = "1" ]; then
  PRETRAIN_ARGS+=(--fp8)
fi

if [ "$DEVICE_TYPE" = "cuda" ]; then
  if ! docker info >/dev/null 2>&1; then
    echo "Docker is required for calculator and HumanEval evaluation." >&2
    exit 1
  fi
  docker image inspect python:3.12-slim >/dev/null 2>&1 || docker pull python:3.12-slim
fi

if [ "$DEVICE_TYPE" = "cuda" ]; then
  uv sync --frozen --extra gpu
else
  uv sync --frozen --extra cpu
fi
# shellcheck source=/dev/null
source .venv/bin/activate

python -m scripts.download_teacher \
  --model "$TEACHER_MODEL" \
  --out-dir "$TEACHER_DIR"

echo "==> Pretraining student with teacher tokenizer"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.pretrain \
  --teacher-model "$TEACHER_DIR" \
  --device-type "$DEVICE_TYPE" \
  --depth 24 \
  --target-param-data-ratio 8 \
  --device-batch-size "$DEVICE_BATCH_SIZE" \
  --run "$MLFLOW_RUN" \
  --run-name "$STUDENT_RUN" \
  --out-dir data \
  "${PRETRAIN_ARGS[@]}"
echo "==> Evaluating pretrained student"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.evaluate_model \
  --checkpoint "data/pretrain_checkpoints/$STUDENT_RUN" \
  --device-type "$DEVICE_TYPE" \
  --device-batch-size "$DEVICE_BATCH_SIZE" \
  --eval core,bpb

echo "==> Distilling teacher into data/pretrain_checkpoints/$STUDENT_RUN"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.distill \
  --checkpoint "data/pretrain_checkpoints/$STUDENT_RUN" \
  --teacher-model "$TEACHER_DIR" \
  --device-type "$DEVICE_TYPE" \
  --teacher-device "$TEACHER_DEVICE" \
  --device-batch-size "$DISTILL_BATCH_SIZE" \
  --tasks smoltalk,mmlu,gsm8k \
  --run "$MLFLOW_RUN" \
  --run-name "$DISTILL_RUN" \
  --out-dir data

if [ "${DEVICE_TYPE:-cuda}" = "cuda" ]; then
  echo "==> Evaluating data/distill_checkpoints/$DISTILL_RUN"
  python -m scripts.evaluate_model \
    --checkpoint "data/distill_checkpoints/$DISTILL_RUN" \
    --eval chat \
    --device-batch-size 32 \
    --vllm-tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --vllm-model "data/distill_checkpoints/$DISTILL_RUN"
fi
