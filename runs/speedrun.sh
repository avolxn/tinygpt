#!/usr/bin/env bash
set -euo pipefail

# End-to-end tinygpt pipeline: tokenizer → pretrain (multi-GPU) → SFT → eval.
# Modeled after nanochat/runs/speedrun.sh; adapted to tinygpt scripts and HF streaming data.
#
# Intended: a clean 8×GPU CUDA node (e.g. 8×H100). Duration depends on dataset/model; plan hours.
#
# From repo root:
#   bash runs/speedrun.sh
# With wandb (login first: `wandb login`):
#   WANDB_RUN=speedrun bash runs/speedrun.sh
# In screen:
#   screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1
export TINYGPT_BASE_DIR="${TINYGPT_BASE_DIR:-$HOME/.cache/tinygpt}"
mkdir -p "$TINYGPT_BASE_DIR"

# -----------------------------------------------------------------------------
# uv + venv

command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
# shellcheck source=/dev/null
source .venv/bin/activate

RUN_NAME="${WANDB_RUN:-dummy}"

# -----------------------------------------------------------------------------
# Tokenizer (streams HuggingFaceFW/fineweb by default; ~2B chars)

python -m scripts.train_tokenizer --max-chars 2000000000
python -m scripts.eval_tokenizer

# -----------------------------------------------------------------------------
# Base model (pretrain): depth 24, tighter data:params ratio like nanochat speedrun

NPROC="${NPROC_PER_NODE:-8}"
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.pretrain \
  --depth 24 \
  --target-param-data-ratio 8 \
  --device-batch-size 16 \
  --run "$RUN_NAME"

torchrun --standalone --nproc_per_node="$NPROC" -m scripts.evaluate \
  --checkpoint out/checkpoints/d24 \
  --device-batch-size 16

# -----------------------------------------------------------------------------
# SFT (conversation tokens + tasks; identity JSONL optional)

curl -fsSL -o "$TINYGPT_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node="$NPROC" -m scripts.finetune \
  --checkpoint out/checkpoints/d24 \
  --device-batch-size 16 \
  --tasks smoltalk,mmlu,gsm8k,identity \
  --identity-conversations "$TINYGPT_BASE_DIR/identity_conversations.jsonl" \
  --run "$RUN_NAME"

torchrun --standalone --nproc_per_node="$NPROC" -m scripts.evaluate \
  --checkpoint out/checkpoints/sft \
  --eval chat \
  --device-batch-size 16

# Interactive chat (optional):
#   python -m scripts.chat --checkpoint out/checkpoints/sft
