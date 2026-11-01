#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACTION="${1:-up}"
if [[ "$ACTION" != "up" && "$ACTION" != "run" ]]; then
  echo "Usage: bash runs/airflow.sh [up|run [config.json]]" >&2
  exit 2
fi
CONF='{}'
if [[ "$ACTION" == "run" && -n "${2:-}" ]]; then
  CONF="$(cat "$2")"
fi

if [[ "$(uname)" == "Darwin" ]]; then
  export MLFLOW_PORT="${MLFLOW_PORT:-5001}"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to run the local Airflow stack." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running. Start Docker Desktop and try again." >&2
  exit 1
fi

COMPOSE_FILES=()
if [[ -f "infra/airflow/.env" ]]; then
  COMPOSE_FILES+=(--env-file infra/airflow/.env)
fi
COMPOSE_FILES+=(-f infra/airflow/compose.yaml)
if [[ -n "${REMOTE_HOST:-}" ]]; then
  if [[ "${AIRFLOW_GPU:-0}" == "1" ]]; then
    echo "Choose either a remote worker or local GPU execution." >&2
    exit 1
  fi
  if ! ssh-add -l >/dev/null 2>&1; then
    ssh-add "${REMOTE_SSH_KEY:-$HOME/.ssh/id_ed25519}"
  fi
  if [[ "$(uname)" == "Darwin" ]]; then
    export SSH_AGENT_SOCKET=/run/host-services/ssh-auth.sock
  else
    export SSH_AGENT_SOCKET="${SSH_AUTH_SOCK:?Start ssh-agent before connecting a remote worker}"
  fi
  export DOCKER_GID="$(stat -c '%g' /var/run/docker.sock 2>/dev/null || echo 0)"
  docker image inspect python:3.12-slim >/dev/null 2>&1 || docker pull python:3.12-slim
  COMPOSE_FILES+=(-f infra/airflow/compose.remote.yaml)
fi
if [[ "${AIRFLOW_GPU:-0}" == "1" ]]; then
  DOCKER_GID="$(stat -c '%g' /var/run/docker.sock)"
  export DOCKER_GID
  docker image inspect python:3.12-slim >/dev/null 2>&1 || docker pull python:3.12-slim
  COMPOSE_FILES+=(-f infra/airflow/compose.gpu.yaml)
fi

if [[ "$ACTION" == "up" ]]; then
  exec docker compose "${COMPOSE_FILES[@]}" up --build
fi

docker compose "${COMPOSE_FILES[@]}" up --build -d
docker compose "${COMPOSE_FILES[@]}" exec -T airflow airflow dags reserialize
docker compose "${COMPOSE_FILES[@]}" exec -T airflow airflow dags unpause tinygpt_training
LOGICAL_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_ID="manual__$LOGICAL_DATE"
docker compose "${COMPOSE_FILES[@]}" exec -T airflow airflow dags trigger tinygpt_training \
  --logical-date "$LOGICAL_DATE" --run-id "$RUN_ID" --conf "$CONF"

LAST_STATE=''
while true; do
  STATE="$(docker compose "${COMPOSE_FILES[@]}" exec -T airflow airflow dags state tinygpt_training "$RUN_ID")"
  STATE="${STATE%%,*}"
  if [[ "$STATE" != "$LAST_STATE" ]]; then
    echo "Pipeline: $STATE"
    LAST_STATE="$STATE"
  fi
  case "$STATE" in
    success) exit 0 ;;
    failed) exit 1 ;;
  esac
  sleep 15
done
