#!/usr/bin/env bash
set -euo pipefail

# Cheap smoke test for CPU or a low-end GPU:
# 1. train a tiny tokenizer on a local corpus
# 2. run a very small pretrain job
# 3. run a tiny local SFT pass
# 4. issue one chat prompt
#
# From repo root:
#   bash runs/runcpu.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
DEVICE_TYPE="${DEVICE_TYPE:-cpu}"

CORPUS="$TINYGPT_BASE_DIR/smoke_corpus.txt"
cat >"$CORPUS" <<'EOF'
TinyGPT smoke test.
The sky is blue because shorter wavelengths scatter more strongly.
Paris is the capital of France.
Two plus two equals four.
Writing small tests catches obvious integration issues early.
EOF

IDENTITY_JSONL="$TINYGPT_BASE_DIR/smoke_identity.jsonl"
cat >"$IDENTITY_JSONL" <<'EOF'
[{"role":"user","content":"Say hello in one sentence."},{"role":"assistant","content":"Hello from TinyGPT."}]
[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"2 + 2 = 4."}]
[{"role":"user","content":"What color is the daytime sky?"},{"role":"assistant","content":"The daytime sky usually looks blue."}]
EOF

echo "==> Training smoke tokenizer"
python -m scripts.train_tokenizer \
  --txt "$CORPUS" \
  --max-chars 200000 \
  --vocab-size 2048 \
  --out-dir out/tokenizer_smoke

echo "==> Tiny local pretrain"
python -m scripts.pretrain \
  --device-type "$DEVICE_TYPE" \
  --depth 4 \
  --aspect-ratio 32 \
  --head-dim 32 \
  --window-pattern L \
  --max-seq-len 128 \
  --device-batch-size 2 \
  --total-batch-size 512 \
  --num-iterations 20 \
  --eval-every 10 \
  --eval-tokens 2048 \
  --sample-every -1 \
  --dataset "" \
  --txt "$CORPUS" \
  --tokenizer-dir out/tokenizer_smoke \
  --run "$WANDB_RUN" \
  --run-name smoke_test

echo "==> Tiny local SFT"
python -m scripts.finetune \
  --device-type "$DEVICE_TYPE" \
  --checkpoint out/pretrain_checkpoints/smoke_test \
  --tokenizer-dir out/tokenizer_smoke \
  --device-batch-size 1 \
  --num-iterations 20 \
  --eval-every 10 \
  --tasks identity \
  --identity-conversations "$IDENTITY_JSONL" \
  --run "$WANDB_RUN" \
  --run-name smoke_test

echo "==> One chat prompt"
python -m scripts.chat \
  --device-type "$DEVICE_TYPE" \
  --checkpoint out/sft_checkpoints/smoke_test \
  --tokenizer-dir out/tokenizer_smoke \
  --prompt "Say hello in one short sentence." \
  --temperature 0.0 \
  --max-tokens 24
