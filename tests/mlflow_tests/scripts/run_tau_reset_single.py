"""End-to-end single-pipeline run for TauResetStrategy.

Mirrors ``examples/tau_reset/backtest.py`` but reads observations
from the cached ``tau_strategy_result.csv`` fixture (~745 hourly rows,
ETH/USDC pool).

Logs to MLflow under ``e2e_tau_reset_single`` and writes the local
strategy DataFrame to ``output/tau_reset_single.csv``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EXP_TAU_SINGLE, OUTPUT_DIR, load_tau_observations, make_mlflow_config  # noqa: E402

from fractal.core.pipeline import DefaultPipeline, ExperimentConfig  # noqa: E402
from fractal.strategies.tau_reset_strategy import TauResetParams, TauResetStrategy  # noqa: E402

# Pool-config attributes flow through class-level state because the
# ``Launcher`` inside the pipeline still uses the legacy positional
# constructor (see comment in ``examples/tau_reset/grid.py``).
TauResetStrategy.token0_decimals = 6
TauResetStrategy.token1_decimals = 18
TauResetStrategy.tick_spacing = 60


def main() -> int:
    print("[tau_reset_single] loading observations…")
    obs = load_tau_observations(n=168)
    print(f"[tau_reset_single] {len(obs)} observations from "
          f"{obs[0].timestamp} to {obs[-1].timestamp}")

    grid = [TauResetParams(TAU=15, INITIAL_BALANCE=1_000_000.0)]

    pipeline = DefaultPipeline(
        mlflow_config=make_mlflow_config(EXP_TAU_SINGLE),
        experiment_config=ExperimentConfig(
            strategy_type=TauResetStrategy,
            backtest_observations=obs,
            params_grid=grid,
        ),
    )
    pipeline.run()

    # Local CSV dump for the verifier.
    s = TauResetStrategy(
        params=grid[0],
        token0_decimals=6, token1_decimals=18, tick_spacing=60,
    )
    result = s.run(obs)
    out_path = OUTPUT_DIR / "tau_reset_single.csv"
    result.to_dataframe().to_csv(out_path, index=False)
    metrics = result.get_default_metrics()
    print(f"[tau_reset_single] metrics: {metrics}")
    print(f"[tau_reset_single] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
