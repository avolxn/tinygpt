#!/usr/bin/env bash
set -euo pipefail

# Approximate "$1000" distillation run:
# 1. convert the nanochat-d32 teacher locally
# 2. load the $100 student trained with the same tokenizer
# 3. run online KL + CE distillation on chat tasks
# 4. run a chat eval pass
#
# Important: the student checkpoint should come from
# runs/train_100usd_nanochat_tokenizer.sh.
#
# From repo root:
#   bash runs/distill_1000usd.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1
export TINYGPT_BASE_DIR="${TINYGPT_BASE_DIR:-$HOME/.cache/tinygpt}"
mkdir -p "$TINYGPT_BASE_DIR"

command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
# shellcheck source=/dev/null
source .venv/bin/activate

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TEACHER_DEVICE="${TEACHER_DEVICE:-same}"

if [ ! -d "out/pretrain_checkpoints/100usd_nanochat_tokenizer" ]; then
  echo "Student checkpoint not found: out/pretrain_checkpoints/100usd_nanochat_tokenizer"
  echo "Run bash runs/train_100usd_nanochat_tokenizer.sh first."
  exit 1
fi

if [ ! -f "out/nanochat_d32/config.json" ] || [ ! -f "out/nanochat_d32/model.safetensors" ]; then
  echo "==> Converting nanochat teacher"
  python -m scripts.convert \
    --input karpathy/nanochat-d32 \
    --out-dir out/nanochat_d32
else
  echo "==> Reusing converted teacher at out/nanochat_d32"
fi

if [ ! -f "$TINYGPT_BASE_DIR/identity_conversations.jsonl" ]; then
  curl -fsSL -o "$TINYGPT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

echo "==> Distilling out/pretrain_checkpoints/100usd_nanochat_tokenizer from out/nanochat_d32"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.distill \
  --checkpoint out/pretrain_checkpoints/100usd_nanochat_tokenizer \
  --tokenizer-dir out/nanochat_d32 \
  --teacher-model out/nanochat_d32 \
  --teacher-tokenizer out/nanochat_d32 \
  --teacher-device "$TEACHER_DEVICE" \
  --device-batch-size 4 \
  --num-iterations 12000 \
  --eval-every 500 \
  --distill-alpha 0.75 \
  --distill-temperature 1.5 \
  --tasks smoltalk,mmlu,gsm8k,identity \
  --identity-conversations "$TINYGPT_BASE_DIR/identity_conversations.jsonl" \
  --run "$WANDB_RUN" \
  --run-name 1000usd_distill

echo "==> Evaluating distilled checkpoint out/distill_checkpoints/1000usd_distill"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.evaluate \
  --checkpoint out/distill_checkpoints/1000usd_distill \
  --tokenizer-dir out/nanochat_d32 \
  --eval chat \
  --device-batch-size 4 \
  --max-problems 64
