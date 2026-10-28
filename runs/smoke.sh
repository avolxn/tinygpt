#!/usr/bin/env bash
set -euo pipefail

# Short teacher-to-student integration check with a reduced student.
#
# From repo root:
#   bash runs/smoke.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p data

MLFLOW_RUN="${MLFLOW_RUN:-smoke}"
DEVICE_TYPE="${DEVICE_TYPE:-cpu}"
TEACHER_MODEL="${TEACHER_MODEL:-karpathy/nanochat-d32}"
TEACHER_DIR="${TEACHER_DIR:-data/teacher_smoke}"
TEACHER_DEVICE="${TEACHER_DEVICE:-same}"

if [ "$DEVICE_TYPE" = "cuda" ]; then
  uv sync --frozen --extra gpu
else
  uv sync --frozen --extra cpu
fi
# shellcheck source=/dev/null
source .venv/bin/activate

python -m scripts.prepare_teacher \
  --model "$TEACHER_MODEL" \
  --out-dir "$TEACHER_DIR"

CORPUS="data/smoke_corpus.txt"
cat >"$CORPUS" <<'EOF'
TinyGPT smoke test.
The sky is blue because shorter wavelengths scatter more strongly.
Paris is the capital of France.
Two plus two equals four.
EOF

VAL_CORPUS="data/smoke_validation.txt"
cat >"$VAL_CORPUS" <<'EOF'
Light scatters as it travels through the atmosphere.
France has its capital in Paris.
EOF

CONVERSATIONS_JSONL="data/smoke_conversations.jsonl"
cat >"$CONVERSATIONS_JSONL" <<'EOF'
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
  --val-txt "$VAL_CORPUS" \
  --run "$MLFLOW_RUN" \
  --run-name smoke \
  --out-dir data

echo "==> Tiny local distillation"
python -m scripts.distill \
  --device-type "$DEVICE_TYPE" \
  --checkpoint data/pretrain_checkpoints/smoke \
  --teacher-model "$TEACHER_DIR" \
  --teacher-device "$TEACHER_DEVICE" \
  --device-batch-size 1 \
  --total-batch-size 512 \
  --num-iterations 20 \
  --eval-every 10 \
  --eval-tokens 2048 \
  --tasks custom \
  --conversations "$CONVERSATIONS_JSONL" \
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
  --eval sample \
  --vllm-model data/distill_checkpoints/smoke \
  --max-new-tokens 8
