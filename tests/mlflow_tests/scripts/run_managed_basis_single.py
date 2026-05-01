"""End-to-end single-pipeline run for HyperliquidBasis.

Mirrors ``examples/basis/backtest.py`` but reads
its observations from the cached CSV fixtures so the run is offline,
deterministic and small (1 week × hourly = 168 obs).

Logs the run to MLflow under experiment ``e2e_managed_basis_single``
(via :class:`DefaultPipeline` with a single-cell grid) and writes the
resulting strategy DataFrame to ``output/managed_basis_single.csv``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXP_MB_SINGLE,
    OUTPUT_DIR,
    HyperliquidBasisParams,
    load_managed_basis_observations,
    make_mlflow_config,
)

from fractal.core.pipeline import DefaultPipeline, ExperimentConfig  # noqa: E402
from fractal.strategies.hyperliquid_basis import HyperliquidBasis  # noqa: E402


def main() -> int:
    print("[managed_basis_single] loading observations…")
    obs = load_managed_basis_observations(ticker="BTC", n=168)
    print(f"[managed_basis_single] {len(obs)} observations from "
          f"{obs[0].timestamp} to {obs[-1].timestamp}")

    grid = [HyperliquidBasisParams(
        MIN_LEVERAGE=1.0,
        TARGET_LEVERAGE=3.0,
        MAX_LEVERAGE=5.0,
        INITIAL_BALANCE=1_000_000.0,
        EXECUTION_COST=0.0005,
    )]

    pipeline = DefaultPipeline(
        mlflow_config=make_mlflow_config(EXP_MB_SINGLE),
        experiment_config=ExperimentConfig(
            strategy_type=HyperliquidBasis,
            backtest_observations=obs,
            params_grid=grid,
        ),
    )
    pipeline.run()

    # Re-run the strategy locally just to dump a CSV the verifier can
    # cross-check against the artifact MLflow received.
    s = HyperliquidBasis(params=grid[0])
    result = s.run(obs)
    out_path = OUTPUT_DIR / "managed_basis_single.csv"
    result.to_dataframe().to_csv(out_path, index=False)
    metrics = result.get_default_metrics()
    print(f"[managed_basis_single] metrics: {metrics}")
    print(f"[managed_basis_single] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
