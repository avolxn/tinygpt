#!/usr/bin/env bash
set -euo pipefail

# Full training pipeline:
# 1. train a tokenizer from scratch
# 2. pretrain a d24 model
# 3. run SFT
# 4. run base/chat eval
#
# From repo root:
#   bash runs/from_scratch.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1
mkdir -p data

command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

echo "Training tokenizer"
python -m scripts.train_tokenizer \
  --out-dir data/tokenizer_from_scratch

echo "Evaluating tokenizer data/tokenizer_from_scratch"
python -m scripts.evaluate_tokenizer --tokenizer-dir data/tokenizer_from_scratch

echo "Pretraining student"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.pretrain \
  --depth 24 \
  --tokenizer-dir data/tokenizer_from_scratch \
  --target-param-data-ratio 8 \
  --device-batch-size 16 \
  --run "$WANDB_RUN" \
  --run-name from_scratch \
  --out-dir data

echo "Evaluating base checkpoint data/pretrain_checkpoints/from_scratch"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.evaluate_model \
  --checkpoint data/pretrain_checkpoints/from_scratch \
  --tokenizer-dir data/tokenizer_from_scratch \
  --device-batch-size 16

if [ ! -f "data/identity_conversations.jsonl" ]; then
  curl -fsSL -o "data/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

echo "Running SFT"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.finetune \
  --checkpoint data/pretrain_checkpoints/from_scratch \
  --tokenizer-dir data/tokenizer_from_scratch \
  --device-batch-size 16 \
  --tasks smoltalk,mmlu,gsm8k,identity \
  --identity-conversations data/identity_conversations.jsonl \
  --run "$WANDB_RUN" \
  --run-name from_scratch \
  --out-dir data

echo "Evaluating chat checkpoint data/sft_checkpoints/from_scratch"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.evaluate_model \
  --checkpoint data/sft_checkpoints/from_scratch \
  --tokenizer-dir data/tokenizer_from_scratch \
  --eval chat \
  --device-batch-size 16
