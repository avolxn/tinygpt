#!/usr/bin/env bash
set -euo pipefail

# Cheap teacher-to-student smoke test for CPU or a low-end GPU.
#
# From repo root:
#   bash runs/smoke.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p data

command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv

MLFLOW_RUN="${MLFLOW_RUN:-smoke}"
DEVICE_TYPE="${DEVICE_TYPE:-cpu}"
TEACHER_MODEL="${TEACHER_MODEL:-hf-internal-testing/tiny-random-LlamaForCausalLM}"
TEACHER_DIR="${TEACHER_DIR:-data/teacher_smoke}"

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
fi

CORPUS="data/smoke_corpus.txt"
cat >"$CORPUS" <<'EOF'
TinyGPT smoke test.
The sky is blue because shorter wavelengths scatter more strongly.
Paris is the capital of France.
Two plus two equals four.
EOF

IDENTITY_JSONL="data/smoke_identity.jsonl"
cat >"$IDENTITY_JSONL" <<'EOF'
[{"role":"user","content":"Say hello in one sentence."},{"role":"assistant","content":"Hello from TinyGPT."}]
[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"2 + 2 = 4."}]
EOF

echo "==> Tiny local pretrain"
python -m scripts.pretrain \
  --device-type "$DEVICE_TYPE" \
  --teacher-model "$TEACHER_DIR" \
  --depth 4 \
  --aspect-ratio 32 \
  --head-dim 32 \
  --max-seq-len 128 \
  --device-batch-size 2 \
  --total-batch-size 512 \
  --num-iterations 20 \
  --eval-every 10 \
  --eval-tokens 2048 \
  --dataset "" \
  --txt "$CORPUS" \
  --run "$MLFLOW_RUN" \
  --run-name smoke \
  --out-dir data

echo "==> Tiny local distillation"
python -m scripts.distill \
  --device-type "$DEVICE_TYPE" \
  --checkpoint data/pretrain_checkpoints/smoke \
  --teacher-model "$TEACHER_DIR" \
  --teacher-device cpu \
  --device-batch-size 1 \
  --total-batch-size 512 \
  --num-iterations 20 \
  --eval-every 10 \
  --eval-tokens 2048 \
  --tasks identity \
  --identity-conversations "$IDENTITY_JSONL" \
  --run "$MLFLOW_RUN" \
  --run-name smoke \
  --out-dir data

if [ "$DEVICE_TYPE" != "cuda" ]; then
  echo "Skipping vLLM chat smoke test on non-CUDA workers"
  exit 0
fi

python -c "import vllm"
python -m scripts.evaluate_model \
  --checkpoint data/distill_checkpoints/smoke \
  --eval chat \
  --vllm-model data/distill_checkpoints/smoke \
  --max-problems 2
