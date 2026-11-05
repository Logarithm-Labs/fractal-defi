"""Parameter grid for the sUSDe PT carry through the standard
:class:`DefaultPipeline` (one MLflow run per cell): target LTV, hedge kind,
hedge ratio, margin share and exit policy.

Usage: ``MLFLOW_URI=... python grid.py [market_key]`` (default ``susde_27nov2025_usds``)
"""
import os
import sys
import warnings

from helpers import build_frame, load_registry, observations, window
from run import make_params
from sklearn.model_selection import ParameterGrid

from fractal.core.pipeline import DefaultPipeline, ExperimentConfig, MLflowConfig
from fractal.strategies import MorphoRateHedgedLeveragedPT

warnings.filterwarnings("ignore")


def build_grid(cfg: dict) -> list:
    raw_grid = ParameterGrid({
        "TARGET_LTV": [0.60, 0.70, 0.80, 0.86],
        "RATE_HEDGE": ["none", "boros"],
        "HEDGE_BETA": [0.05, 0.10, 0.20],
        "HEDGE_MARGIN_SHARE": [0.05, 0.10],
        "BOROS_EXIT_POLICY": ["settle", "close"],
    })
    valid_grid = []
    for cell in raw_grid:
        if cell["RATE_HEDGE"] == "none" and (cell["HEDGE_BETA"] != 0.10 or cell["HEDGE_MARGIN_SHARE"] != 0.05
                                             or cell["BOROS_EXIT_POLICY"] != "settle"):
            continue  # the plain loop has no hedge parameters
        ltv = cell["TARGET_LTV"]
        valid_grid.append(make_params(cfg, cell["RATE_HEDGE"], **cell,
                                      REBALANCE_LTV_BAND=(round(ltv - 0.10, 2), round(min(ltv + 0.08, 0.90), 2))))
    print(f"Length of valid grid: {len(valid_grid)}")
    return valid_grid


if __name__ == "__main__":
    registry = load_registry()
    key = sys.argv[1] if len(sys.argv) > 1 else "susde_27nov2025_usds"
    cfg = registry[key]
    mlflow_uri = os.getenv("MLFLOW_URI")
    if not mlflow_uri:
        raise ValueError("MLFLOW_URI isn't set.")
    start, end = window(cfg)
    frame, _, _ = build_frame(cfg, start, end)
    obs = observations(frame, cfg, "boros")  # a superset: the plain loop ignores the BOROS state
    assert len(obs) > 0
    experiment_config = ExperimentConfig(
        strategy_type=MorphoRateHedgedLeveragedPT,
        backtest_observations=obs,
        params_grid=build_grid(cfg),
        debug=False,
    )
    mlflow_config = MLflowConfig(
        mlflow_uri=mlflow_uri,
        experiment_name=f"susde_pt_carry_{key}_{start:%Y-%m-%d}_{end:%Y-%m-%d}",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    DefaultPipeline(experiment_config=experiment_config, mlflow_config=mlflow_config).run()
