#!/usr/bin/env bash
# Run Airflow task scripts on the GPU worker over OpenSSH. Reverse forwards keep
# MLflow and the evaluation sandbox on the local machine without public ports.
set -euo pipefail

REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_WORKSPACE="${REMOTE_WORKSPACE:-/workspace/tinygpt}"
SSH=(ssh -p "$REMOTE_PORT" -o BatchMode=yes -o ConnectTimeout=20
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile=/opt/airflow/remote_known_hosts)
printf -v REMOTE_DIR '%q' "$REMOTE_WORKSPACE"

if [[ "${1:-}" == "setup" ]]; then
  "${SSH[@]}" "$REMOTE_HOST" "mkdir -p $REMOTE_DIR && chmod 700 $REMOTE_DIR"
  tar --exclude=__pycache__ -C /workspace -cf - pyproject.toml uv.lock README.md src scripts tasks runs |
    "${SSH[@]}" "$REMOTE_HOST" "tar --no-same-owner -xf - -C $REMOTE_DIR"
  "${SSH[@]}" "$REMOTE_HOST" "cd $REMOTE_DIR && bash -se" <<'SETUP'
set -euo pipefail
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/0.10.9/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PWD/bin:$PATH"
export UV_CACHE_DIR="$PWD/.cache/uv"
if ! command -v docker >/dev/null; then
  mkdir -p bin
  curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-28.0.1.tgz |
    tar --no-same-owner -xz --strip-components=1 -C bin docker/docker
fi
uv sync --frozen --extra gpu --python 3.12
docker --version
nvidia-smi
SETUP
  exit
fi

# sshd leaves a Unix forwarding socket behind when a session closes. Tasks in
# this DAG are sequential, so remove that stale listener before the next one.
"${SSH[@]}" -n "$REMOTE_HOST" "rm -f $REMOTE_DIR/docker.sock"

{
  echo 'set -euo pipefail'
  printf 'cd %s\n' "$REMOTE_DIR"
  # Send only the task's explicit inputs; local service credentials stay local.
  for name in RUN_NAME TEACHER_MODEL DATASET TEXT_FIELD TXT_INPUT DEVICE_TYPE \
    NPROC_PER_NODE TENSOR_PARALLEL_SIZE DEPTH FP8 MAX_SEQ_LEN DEVICE_BATCH_SIZE \
    DISTILL_BATCH_SIZE TOTAL_BATCH_SIZE PRETRAIN_ITERATIONS DISTILL_ITERATIONS \
    DISTILL_TASKS CHAT_EVAL_TASKS MAX_EVAL_PROBLEMS EVAL_TOKENS EVAL_EVERY \
    CORE_MAX_PER_TASK RESUME_STEPS MLFLOW_EXPERIMENT_NAME; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done
  printf 'export PYTHON_BIN=%q TORCHRUN_BIN=%q\n' "$REMOTE_WORKSPACE/.venv/bin/python" "$REMOTE_WORKSPACE/.venv/bin/torchrun"
  printf 'export PYTHONPATH=%q\n' "$REMOTE_WORKSPACE/src:$REMOTE_WORKSPACE"
  printf 'export DOCKER_HOST=%q\n' "unix://$REMOTE_WORKSPACE/docker.sock"
  echo 'export MLFLOW_TRACKING_URI=http://127.0.0.1:15000 OMP_NUM_THREADS=1'
  echo 'export PATH="$PWD/bin:$PATH"'
  echo 'export HF_HOME="$PWD/.cache/huggingface" TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor"'
  cat
} | "${SSH[@]}" -o ExitOnForwardFailure=yes \
  -R 127.0.0.1:15000:mlflow:5000 \
  -R "$REMOTE_WORKSPACE/docker.sock:/var/run/docker.sock" \
  "$REMOTE_HOST" bash -se
