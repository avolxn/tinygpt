#!/usr/bin/env bash
set -euo pipefail

# Pretraining run using a tokenizer converted from the reference teacher.
#
# From repo root:
#   bash runs/pretrain_reference_d32.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1
mkdir -p data

command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
# shellcheck source=/dev/null
source .venv/bin/activate

WANDB_RUN="${WANDB_RUN:-pretrain_reference_d32}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
REFERENCE_MODEL="${REFERENCE_MODEL:-karpathy/nanochat-d32}"

python -m scripts.check_wandb

if [ ! -f "data/tokenizer_reference_d32/tokenizer.json" ]; then
  echo "==> Converting reference tokenizer"
  python -m scripts.convert \
    --input "$REFERENCE_MODEL" \
    --out-dir data/tokenizer_reference_d32 \
    --skip-model
else
  echo "==> Reusing tokenizer at data/tokenizer_reference_d32"
fi

python -m scripts.evaluate_tokenizer --tokenizer-dir data/tokenizer_reference_d32

echo "==> Pretraining student with reference tokenizer"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.pretrain \
  --depth 32 \
  --tokenizer-dir data/tokenizer_reference_d32 \
  --target-param-data-ratio 12 \
  --device-batch-size 32 \
  --run "$WANDB_RUN" \
  --run-name pretrain_reference_d32 \
  --out-dir data

echo "==> Evaluating base checkpoint data/pretrain_checkpoints/pretrain_reference_d32"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.evaluate_model \
  --checkpoint data/pretrain_checkpoints/pretrain_reference_d32 \
  --tokenizer-dir data/tokenizer_reference_d32 \
  --device-batch-size 32
