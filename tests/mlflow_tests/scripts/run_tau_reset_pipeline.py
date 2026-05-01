"""End-to-end MLflow grid pipeline for TauResetStrategy.

Mirrors ``examples/tau_reset/grid.py`` but with a 4-cell
grid + offline CSV fixture so it finishes in seconds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXP_TAU_PIPELINE,
    load_tau_observations,
    make_mlflow_config,
)

from fractal.core.pipeline import DefaultPipeline, ExperimentConfig  # noqa: E402
from fractal.strategies.tau_reset_strategy import TauResetParams, TauResetStrategy  # noqa: E402

# Same class-level pool config as the example pipeline.
TauResetStrategy.token0_decimals = 6
TauResetStrategy.token1_decimals = 18
TauResetStrategy.tick_spacing = 60


GRID = [
    TauResetParams(TAU=tau, INITIAL_BALANCE=1_000_000.0)
    for tau in (5, 10, 20, 30)
]


def main() -> int:
    print("[tau_reset_pipeline] loading observations…")
    obs = load_tau_observations(n=168)
    print(f"[tau_reset_pipeline] {len(obs)} observations, grid size {len(GRID)}")

    pipeline = DefaultPipeline(
        mlflow_config=make_mlflow_config(EXP_TAU_PIPELINE),
        experiment_config=ExperimentConfig(
            strategy_type=TauResetStrategy,
            backtest_observations=obs,
            params_grid=GRID,
        ),
    )
    pipeline.run()
    print(f"[tau_reset_pipeline] {len(GRID)} runs logged to MLflow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
