#!/usr/bin/env bash
# Bring up the local MLflow tracking server and wait until /health
# answers 200. Idempotent: if the container is already running,
# only the wait loop runs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$HERE/docker-compose.yml"
URL="${MLFLOW_URI:-http://localhost:5500}/health"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found on PATH" >&2
    exit 1
fi

# Pick the available compose CLI (v2 plugin or legacy v1).
if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose -f "$COMPOSE_FILE")
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose -f "$COMPOSE_FILE")
else
    echo "ERROR: neither 'docker compose' nor 'docker-compose' found" >&2
    exit 1
fi

echo "starting MLflow…"
"${COMPOSE[@]}" up -d --build

echo "waiting for $URL"
for _ in $(seq 1 60); do
    if curl -fsS "$URL" >/dev/null 2>&1; then
        echo "MLflow up at $URL"
        exit 0
    fi
    sleep 1
done

echo "ERROR: MLflow did not become ready within 60s" >&2
"${COMPOSE[@]}" logs --tail=50 mlflow >&2 || true
exit 1
