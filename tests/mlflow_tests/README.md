# End-to-end MLflow tests

End-to-end harness that exercises the whole framework against a real
MLflow tracking server: `BasisTradingStrategy` (single + grid) and
`TauResetStrategy` (single + grid), then a verifier that walks both
the local artifact tree and the MLflow REST API to confirm everything
that should have been logged actually was.

## What it covers

| Layer | Check |
|---|---|
| Strategy run | offline 168-step replay over `examples/` CSV fixtures (no network) |
| `DefaultPipeline` | grid iteration → MLflow runs (params, metrics, artifacts) |
| MLflow stack | tracking URI, experiment auto-create, run lifecycle, artifact upload |
| Local artifacts | per-script CSV output (timestamps monotonic, `net_balance > 0`, finite) |
| MLflow API | experiment exists, expected run count, expected params/metrics/artifacts |

## Layout

```
tests/mlflow_tests/
├── README.md                 ← this file
├── docker-compose.yml        ← MLflow tracking server (sqlite + fs artifacts)
├── Dockerfile                ← python:3.11-slim + mlflow 2.14.1
├── scripts/
│   ├── start_mlflow.sh
│   ├── stop_mlflow.sh
│   ├── run_managed_basis_single.py     ← 1 run, MLflow + CSV
│   ├── run_managed_basis_pipeline.py   ← 4-cell grid
│   ├── run_tau_reset_single.py         ← 1 run, MLflow + CSV
│   ├── run_tau_reset_pipeline.py       ← 4-cell grid
│   ├── verify_artifacts.py             ← reports OK / FAIL with diagnostics
│   └── e2e.sh                          ← orchestrator (steps 1-6)
└── output/                              ← runtime, gitignored
```

## Running the whole thing

```bash
bash tests/mlflow_tests/scripts/e2e.sh
```

The orchestrator brings MLflow up, runs all four scripts, calls the
verifier, then tears MLflow down. To keep the server running so you can
poke at the UI afterwards:

```bash
KEEP_RUNNING=1 bash tests/mlflow_tests/scripts/e2e.sh
# inspect at http://localhost:5500
bash tests/mlflow_tests/scripts/stop_mlflow.sh           # later
bash tests/mlflow_tests/scripts/stop_mlflow.sh --wipe    # also delete sqlite + artifacts
```

## Running pieces individually

```bash
# Bring up MLflow only:
bash tests/mlflow_tests/scripts/start_mlflow.sh

# Run one stage:
python tests/mlflow_tests/scripts/run_managed_basis_single.py
python tests/mlflow_tests/scripts/run_managed_basis_pipeline.py
python tests/mlflow_tests/scripts/run_tau_reset_single.py
python tests/mlflow_tests/scripts/run_tau_reset_pipeline.py

# Just verify (after running anything above):
python tests/mlflow_tests/scripts/verify_artifacts.py
```

## Configuration

Environment knobs honoured by all scripts:

| Var | Default | Purpose |
|---|---|---|
| `MLFLOW_URI` | `http://localhost:5500` | Tracking server URL |
| `OUTPUT_DIR` | `tests/mlflow_tests/output` | Local CSV destination |
| `PYTHON`     | `python`                  | Interpreter for `e2e.sh` |
| `KEEP_RUNNING` | (unset)                 | If set, `e2e.sh` leaves MLflow up |

## Requirements

- Docker (with either `docker compose` plugin or legacy `docker-compose`)
- Python ≥3.10 with the project installed (`pip install -e ".[dev]"`)
- Roughly 200MB free disk for the MLflow image + sqlite/artifact volume

## Expected output

```
=== 1. starting MLflow ===
…
=== 6. verifying artifacts ===
[1/3] managed_basis_single.csv
  ✓ file exists: …/managed_basis_single.csv
  ✓ row count >= 100 (got 168)
  ✓ columns include ['HEDGE_balance', 'SPOT_balance', 'net_balance', 'timestamp']…
  ✓ net_balance is finite throughout
  ✓ net_balance > 0 throughout
  ✓ timestamps are monotonic increasing
…
ALL CHECKS PASSED.
```

A failure looks like `✗ <reason>` and the script exits non-zero with a
summary at the bottom — useful in CI.

## Notes

- Pipelines write artifacts to MLflow only; the `*_single.py` scripts
  also dump the strategy DataFrame locally so the verifier can compare
  shapes and content.
- Fixtures come from `examples/{basis,tau_reset}/`
  — the same CSVs the slow real-data tests use, so you don't need API
  keys or network.
- The harness is intentionally **not** wired into the default `pytest`
  run: it requires Docker and is gated behind a shell entrypoint. Run
  it manually before releases or in a dedicated CI stage.
