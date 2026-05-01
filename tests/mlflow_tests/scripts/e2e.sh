#!/usr/bin/env bash
# Full end-to-end orchestration:
#   1. start MLflow (idempotent)
#   2. run managed_basis single + grid pipelines
#   3. run tau_reset single + grid pipelines
#   4. verify artifacts (local CSVs + MLflow API)
#
# Usage:
#     bash tests/mlflow_tests/scripts/e2e.sh           # run everything
#     KEEP_RUNNING=1 bash …/e2e.sh                     # leave MLflow up after
#     bash tests/mlflow_tests/scripts/stop_mlflow.sh   # tear it down later
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
export MLFLOW_URI="${MLFLOW_URI:-http://localhost:5500}"
export PYTHONPATH=""  # M6: don't shove import-path into loader cache root
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/tests/mlflow_tests/output}"

PYTHON="${PYTHON:-python}"

cleanup() {
    if [[ -z "${KEEP_RUNNING:-}" ]]; then
        echo
        echo "tearing MLflow down (set KEEP_RUNNING=1 to skip)"
        "$HERE/stop_mlflow.sh" || true
    fi
}
trap cleanup EXIT

echo "=== 1. starting MLflow ==="
"$HERE/start_mlflow.sh"

echo
echo "=== 2. managed_basis single ==="
"$PYTHON" "$HERE/run_managed_basis_single.py"

echo
echo "=== 3. managed_basis pipeline (4-cell grid) ==="
"$PYTHON" "$HERE/run_managed_basis_pipeline.py"

echo
echo "=== 4. tau_reset single ==="
"$PYTHON" "$HERE/run_tau_reset_single.py"

echo
echo "=== 5. tau_reset pipeline (4-cell grid) ==="
"$PYTHON" "$HERE/run_tau_reset_pipeline.py"

echo
echo "=== 6. verifying artifacts ==="
"$PYTHON" "$HERE/verify_artifacts.py"
