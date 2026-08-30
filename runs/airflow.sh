#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
if [[ "${AIRFLOW_GPU:-0}" == "1" ]]; then
  COMPOSE_FILES+=(-f infra/airflow/compose.gpu.yaml)
fi

docker compose "${COMPOSE_FILES[@]}" up --build
