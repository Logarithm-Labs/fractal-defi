"""Parameter grid for ``MorphoLeveragedPT`` on one leveraged market through
the standard :class:`DefaultPipeline` (one MLflow run per cell): target LTV,
loop count, multiply mode and the carry-gate smoothing.

Usage: ``MLFLOW_URI=... python grid.py [market_key]`` (default ``usde_25sep2025``)
"""
import os
import sys
import warnings

from helpers import build_leveraged_frame, leveraged_observations, load_registry, window
from sklearn.model_selection import ParameterGrid

from fractal.core.pipeline import DefaultPipeline, ExperimentConfig, MLflowConfig
from fractal.strategies import MorphoLeveragedPT

warnings.filterwarnings("ignore")


def build_grid(cfg: dict) -> list:
    bar_hours = 24 if cfg["bar"] == "day" else 1
    raw_grid = ParameterGrid({
        "TARGET_LTV": [0.50, 0.60, 0.70, 0.80, 0.86],
        "MAX_LOOPS": [0, 2, 4, 8],
        "MULTIPLY_MODE": ["loop", "flash"],
        "CARRY_GATE_LOOKBACK_BARS": [1, 7 * 24 // bar_hours],
        "INITIAL_BALANCE": [100_000.0],
        "MIN_HEALTH_FACTOR": [1.03],
        "MIN_CARRY_SPREAD": [-0.05],
        "MAX_BORROW_APY": [0.40],
        "MIN_DAYS_TO_MATURITY_AT_ENTRY": [1],
        "BAR_HOURS": [bar_hours],
        "LLTV": [float(cfg["lltv"])],
        "PT_IMPACT_MODEL": [cfg.get("pt_impact_model", "rate_spread")],
        "PT_FEE_LN_RATE": [0.001],
        "PT_IMPACT_LN_RATE_PER_SHARE": [0.075],
    })
    valid_grid = []
    for params in raw_grid:
        if params["MULTIPLY_MODE"] == "flash" and params["MAX_LOOPS"] != 8:
            continue  # flash mode ignores the loop count
        ltv = params["TARGET_LTV"]
        valid_grid.append({**params, "REBALANCE_LTV_BAND": (round(ltv - 0.10, 2), round(min(ltv + 0.08, 0.90), 2))})
    print(f"Length of valid grid: {len(valid_grid)}")
    return valid_grid


if __name__ == "__main__":
    registry = load_registry()["leveraged"]
    key = sys.argv[1] if len(sys.argv) > 1 else "usde_25sep2025"
    cfg = registry[key]
    mlflow_uri = os.getenv("MLFLOW_URI")
    if not mlflow_uri:
        raise ValueError("MLFLOW_URI isn't set.")
    start, end = window(cfg)
    frame, _ = build_leveraged_frame(cfg, start, end)
    observations = leveraged_observations(frame, cfg)
    assert len(observations) > 0
    experiment_config = ExperimentConfig(
        strategy_type=MorphoLeveragedPT,
        backtest_observations=observations,
        params_grid=build_grid(cfg),
        debug=False,
    )
    mlflow_config = MLflowConfig(
        mlflow_uri=mlflow_uri,
        experiment_name=f"leveraged_pt_{key}_{start:%Y-%m-%d}_{end:%Y-%m-%d}",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    DefaultPipeline(experiment_config=experiment_config, mlflow_config=mlflow_config).run()
