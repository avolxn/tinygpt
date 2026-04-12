#!/usr/bin/env bash
set -euo pipefail

# Small CPU/MPS smoke run: tokenizer → tiny pretrain → short SFT → optional eval.
# Modeled after nanochat/runs/runcpu.sh. Not for quality — only to exercise tinygpt paths.
#
# From repo root:
#   bash runs/runcpu.sh
#
# You can paste commands one-by-one if anything fails (HF download, etc.).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TINYGPT_BASE_DIR="${TINYGPT_BASE_DIR:-$HOME/.cache/tinygpt}"
mkdir -p "$TINYGPT_BASE_DIR"

command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
# shellcheck source=/dev/null
source .venv/bin/activate

RUN_NAME="${WANDB_RUN:-dummy}"

# Tiny local corpus so pretrain does not depend on repo shipping data/shakespeare.txt
CORPUS="$TINYGPT_BASE_DIR/runcpu_corpus.txt"
if [ ! -s "$CORPUS" ]; then
  for _ in $(seq 1 400); do
    echo "The quick brown fox jumps over the lazy dog. TinyGPT CPU smoke run."
  done >"$CORPUS"
fi

# Tokenizer (smaller char budget than full speedrun for laptops)
python -m scripts.train_tokenizer --txt "$CORPUS" --max-chars 5000000 --vocab-size 8192
python -m scripts.eval_tokenizer

# Small model, local text
python -m scripts.pretrain \
  --depth 6 \
  --head-dim 64 \
  --window-pattern L \
  --max-seq-len 512 \
  --device-batch-size 4 \
  --total-batch-size 4096 \
  --eval-every 50 \
  --eval-tokens 524288 \
  --sample-every 50 \
  --num-iterations 200 \
  --dataset "" \
  --txt "$CORPUS" \
  --run "$RUN_NAME"

python -m scripts.evaluate \
  --checkpoint out/checkpoints/d6 \
  --device-batch-size 1 \
  --split-tokens 16384 \
  --max-problems 8

curl -fsSL -o "$TINYGPT_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.finetune \
  --checkpoint out/checkpoints/d6 \
  --max-seq-len 512 \
  --device-batch-size 2 \
  --num-iterations 100 \
  --eval-every 50 \
  --tasks identity \
  --identity-conversations "$TINYGPT_BASE_DIR/identity_conversations.jsonl" \
  --run "$RUN_NAME"

# Optional: Paris / sky prompts — interactive:
#   python -m scripts.chat --checkpoint out/checkpoints/sft
