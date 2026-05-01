#!/usr/bin/env bash
# Stop the MLflow tracking server. Pass --wipe to also delete the
# host-side mlflow-data/ directory (sqlite db + artifacts).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$HERE/docker-compose.yml"

if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose -f "$COMPOSE_FILE")
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose -f "$COMPOSE_FILE")
else
    echo "ERROR: neither 'docker compose' nor 'docker-compose' found" >&2
    exit 1
fi

"${COMPOSE[@]}" down

if [[ "${1:-}" == "--wipe" ]]; then
    echo "wiping mlflow-data/"
    rm -rf "$HERE/mlflow-data"
fi
