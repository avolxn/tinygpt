#!/usr/bin/env bash
set -euo pipefail

# Approximate "$100" pretraining run using the tokenizer from:
#   https://huggingface.co/karpathy/nanochat-d32
#
# This is the student you should use later with runs/distill_from_nanochat_d32.sh,
# because tinygpt's online KL distillation requires an identical tokenizer.
#
# From repo root:
#   bash runs/pretrain_with_nanochat_d32.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1
mkdir -p data

command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
# shellcheck source=/dev/null
source .venv/bin/activate

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

if [ ! -f "data/tokenizer_nanochat_d32/tokenizer.json" ]; then
  echo "==> Converting nanochat tokenizer"
  python -m scripts.convert \
    --input karpathy/nanochat-d32 \
    --out-dir data/tokenizer_nanochat_d32 \
    --skip-model
else
  echo "==> Reusing tokenizer at data/tokenizer_nanochat_d32"
fi

python -m scripts.eval_tokenizer --tokenizer-dir data/tokenizer_nanochat_d32

echo "==> Pretraining student with nanochat tokenizer"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.pretrain \
  --depth 24 \
  --tokenizer-dir data/tokenizer_nanochat_d32 \
  --target-param-data-ratio 8 \
  --device-batch-size 16 \
  --run "$WANDB_RUN" \
  --run-name pretrain_with_nanochat_d32 \
  --out-dir data

echo "==> Evaluating base checkpoint data/pretrain_checkpoints/pretrain_with_nanochat_d32"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.evaluate \
  --checkpoint data/pretrain_checkpoints/pretrain_with_nanochat_d32 \
  --tokenizer-dir data/tokenizer_nanochat_d32 \
  --device-batch-size 16
