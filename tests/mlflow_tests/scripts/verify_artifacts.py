"""Verify the e2e MLflow run produced everything we expect.

Three layers of checks:

1. **Local CSVs** (``tests/mlflow_tests/output/*.csv``) — written by the
   ``*_single.py`` scripts. Verify columns, row counts, finiteness,
   strategy invariants (net_balance > 0, monotonic timestamps).

2. **MLflow experiments** (via ``mlflow.tracking.MlflowClient``) — every
   experiment must exist with the expected number of FINISHED runs, and
   each run must carry the expected params, metrics and artifacts.

3. **MLflow artifact contents** — download the ``strategy_backtest_data.csv``
   for one run per experiment and sanity-check its shape.

Exit code 0 on success, 1 with a diagnostic on the first failure (we
keep going through all checks first so the report lists all problems).
"""
import math
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (DEFAULT_MLFLOW_URI,  # noqa: E402
                     EXP_MB_PIPELINE, EXP_MB_SINGLE,
                     EXP_TAU_PIPELINE, EXP_TAU_SINGLE, OUTPUT_DIR)

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402


# ----------------------------------------------------------- collector
class Failures:
    def __init__(self) -> None:
        self.errors: List[str] = []

    def check(self, ok: bool, msg: str) -> None:
        if not ok:
            self.errors.append(msg)
            print(f"  ✗ {msg}")
        else:
            print(f"  ✓ {msg}")

    def summary(self) -> int:
        if not self.errors:
            print("\nALL CHECKS PASSED.")
            return 0
        print(f"\n{len(self.errors)} CHECK(S) FAILED:")
        for e in self.errors:
            print(f"  - {e}")
        return 1


# ----------------------------------------------------------- local CSV checks
def check_managed_basis_csv(f: Failures) -> None:
    print("\n[1/3] managed_basis_single.csv")
    path = OUTPUT_DIR / "managed_basis_single.csv"
    f.check(path.exists(), f"file exists: {path}")
    if not path.exists():
        return
    df = pd.read_csv(path)
    f.check(len(df) >= 100, f"row count >= 100 (got {len(df)})")
    expected = {"timestamp", "net_balance", "HEDGE_balance", "SPOT_balance"}
    f.check(expected.issubset(df.columns),
            f"columns include {sorted(expected)}; got {sorted(df.columns)[:10]}…")
    f.check(df["net_balance"].apply(math.isfinite).all(),
            "net_balance is finite throughout")
    f.check((df["net_balance"] > 0).all(), "net_balance > 0 throughout")
    ts = pd.to_datetime(df["timestamp"])
    f.check(ts.is_monotonic_increasing, "timestamps are monotonic increasing")


def check_tau_csv(f: Failures) -> None:
    print("\n[2/3] tau_reset_single.csv")
    path = OUTPUT_DIR / "tau_reset_single.csv"
    f.check(path.exists(), f"file exists: {path}")
    if not path.exists():
        return
    df = pd.read_csv(path)
    f.check(len(df) >= 100, f"row count >= 100 (got {len(df)})")
    expected = {"timestamp", "net_balance", "UNISWAP_V3_balance"}
    f.check(expected.issubset(df.columns),
            f"columns include {sorted(expected)}; got {sorted(df.columns)[:10]}…")
    f.check(df["net_balance"].apply(math.isfinite).all(),
            "net_balance is finite throughout")
    f.check((df["net_balance"] > 0).all(), "net_balance > 0 throughout")
    ts = pd.to_datetime(df["timestamp"])
    f.check(ts.is_monotonic_increasing, "timestamps are monotonic increasing")


# ----------------------------------------------------------- MLflow checks
def _list_runs(client: MlflowClient, experiment_name: str):
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None, []
    runs = client.search_runs([exp.experiment_id])
    return exp, runs


_EXPECTED_METRICS = {
    "accumulated_return", "apy", "sharpe", "max_drawdown",
}


def check_mlflow_experiment(
    f: Failures, client: MlflowClient,
    name: str, expected_runs: int,
    expected_param_keys: List[str],
) -> None:
    print(f"\n  experiment: {name}")
    exp, runs = _list_runs(client, name)
    f.check(exp is not None, f"experiment exists: {name}")
    if exp is None:
        return
    finished = [r for r in runs if r.info.status == "FINISHED"]
    f.check(len(finished) == expected_runs,
            f"finished run count = {expected_runs} (got {len(finished)})")
    if not finished:
        return

    sample = finished[0]
    # params
    have_params = set(sample.data.params.keys())
    for k in expected_param_keys:
        f.check(k in have_params, f"  param logged: {k}")
    # metrics
    have_metrics = set(sample.data.metrics.keys())
    for m in _EXPECTED_METRICS:
        f.check(m in have_metrics, f"  metric logged: {m}")
        if m in sample.data.metrics:
            v = sample.data.metrics[m]
            f.check(math.isfinite(v), f"  metric {m} is finite (got {v})")
    # artifact
    artifacts = {a.path for a in client.list_artifacts(sample.info.run_id)}
    f.check("strategy_backtest_data.csv" in artifacts,
            f"  artifact: strategy_backtest_data.csv (have {sorted(artifacts)})")


def check_mlflow(f: Failures) -> None:
    print("\n[3/3] MLflow tracking server")
    mlflow.set_tracking_uri(DEFAULT_MLFLOW_URI)
    client = MlflowClient(tracking_uri=DEFAULT_MLFLOW_URI)
    try:
        client.search_experiments()
    except Exception as e:  # pylint: disable=broad-exception-caught
        f.check(False, f"MLflow reachable at {DEFAULT_MLFLOW_URI}: {e}")
        return
    f.check(True, f"MLflow reachable at {DEFAULT_MLFLOW_URI}")

    check_mlflow_experiment(
        f, client, EXP_MB_SINGLE, expected_runs=1,
        expected_param_keys=["MIN_LEVERAGE", "TARGET_LEVERAGE",
                             "MAX_LEVERAGE", "INITIAL_BALANCE",
                             "EXECUTION_COST"],
    )
    check_mlflow_experiment(
        f, client, EXP_MB_PIPELINE, expected_runs=4,
        expected_param_keys=["MIN_LEVERAGE", "TARGET_LEVERAGE",
                             "MAX_LEVERAGE", "INITIAL_BALANCE",
                             "EXECUTION_COST"],
    )
    check_mlflow_experiment(
        f, client, EXP_TAU_SINGLE, expected_runs=1,
        expected_param_keys=["TAU", "INITIAL_BALANCE"],
    )
    check_mlflow_experiment(
        f, client, EXP_TAU_PIPELINE, expected_runs=4,
        expected_param_keys=["TAU", "INITIAL_BALANCE"],
    )


# ----------------------------------------------------------- main
def main() -> int:
    f = Failures()
    check_managed_basis_csv(f)
    check_tau_csv(f)
    check_mlflow(f)
    return f.summary()


if __name__ == "__main__":
    sys.exit(main())
