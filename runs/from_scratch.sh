#!/usr/bin/env bash
set -euo pipefail

# Full training pipeline:
# 1. train a tokenizer from scratch
# 2. pretrain a d24 model
# 3. run SFT
# 4. run base/chat eval
#
# From repo root:
#   bash runs/speedrun.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1
export TINYGPT_BASE_DIR="${TINYGPT_BASE_DIR:-$HOME/.cache/tinygpt}"
mkdir -p "$TINYGPT_BASE_DIR"

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
  --out-dir out/tokenizer_100usd

echo "Evaluating tokenizer out/tokenizer_100usd"
python -m scripts.eval_tokenizer --tokenizer-dir out/tokenizer_100usd

echo "Pretraining student"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.pretrain \
  --depth 24 \
  --tokenizer-dir out/tokenizer_100usd \
  --target-param-data-ratio 8 \
  --device-batch-size 16 \
  --run "$WANDB_RUN" \
  --run-name 100usd_from_scratch

echo "Evaluating base checkpoint out/pretrain_checkpoints/100usd_from_scratch"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.evaluate \
  --checkpoint out/pretrain_checkpoints/100usd_from_scratch \
  --tokenizer-dir out/tokenizer_100usd \
  --device-batch-size 16

if [ ! -f "$TINYGPT_BASE_DIR/identity_conversations.jsonl" ]; then
  curl -fsSL -o "$TINYGPT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

echo "Running SFT"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.finetune \
  --checkpoint out/pretrain_checkpoints/100usd_from_scratch \
  --tokenizer-dir out/tokenizer_100usd \
  --device-batch-size 16 \
  --tasks smoltalk,mmlu,gsm8k,identity \
  --identity-conversations "$TINYGPT_BASE_DIR/identity_conversations.jsonl" \
  --run "$WANDB_RUN" \
  --run-name 100usd_from_scratch

echo "Evaluating chat checkpoint out/sft_checkpoints/100usd_from_scratch"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.evaluate \
  --checkpoint out/sft_checkpoints/100usd_from_scratch \
  --tokenizer-dir out/tokenizer_100usd \
  --eval chat \
  --device-batch-size 16
