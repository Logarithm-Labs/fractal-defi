"""End-to-end MLflow grid pipeline for HyperliquidBasis.

Mirrors ``examples/basis/grid.py`` but with a
tiny 4-cell grid + offline CSV fixtures + 168-step horizon so the whole
pipeline finishes in seconds.

Each grid cell becomes one MLflow run under experiment
``e2e_managed_basis_pipeline`` with logged params, metrics and the
``strategy_backtest_data.csv`` artifact.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXP_MB_PIPELINE,
    HyperliquidBasisParams,
    load_managed_basis_observations,
    make_mlflow_config,
)

from fractal.core.pipeline import DefaultPipeline, ExperimentConfig  # noqa: E402
from fractal.strategies.hyperliquid_basis import HyperliquidBasis  # noqa: E402

GRID = [
    HyperliquidBasisParams(
        MIN_LEVERAGE=min_lev, TARGET_LEVERAGE=target, MAX_LEVERAGE=max_lev,
        INITIAL_BALANCE=1_000_000.0, EXECUTION_COST=0.0005,
    )
    for (min_lev, target, max_lev) in [
        (1.0, 2.0, 4.0),
        (1.0, 3.0, 5.0),
        (2.0, 3.0, 5.0),
        (1.5, 2.5, 4.5),
    ]
]


def main() -> int:
    print("[managed_basis_pipeline] loading observations…")
    obs = load_managed_basis_observations(ticker="BTC", n=168)
    print(f"[managed_basis_pipeline] {len(obs)} observations, grid size {len(GRID)}")

    pipeline = DefaultPipeline(
        mlflow_config=make_mlflow_config(EXP_MB_PIPELINE),
        experiment_config=ExperimentConfig(
            strategy_type=HyperliquidBasis,
            backtest_observations=obs,
            params_grid=GRID,
        ),
    )
    pipeline.run()
    print(f"[managed_basis_pipeline] {len(GRID)} runs logged to MLflow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
