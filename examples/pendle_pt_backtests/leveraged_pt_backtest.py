"""Run ``MorphoLeveragedPT`` on the markets in ``markets.json["leveraged"]``.

Writes ``results/leveraged_<market>.csv`` (per-bar entity states) and
appends one row per run to ``results/validation.csv`` comparing the
realised APY with the closed form ``L·y_pt − (L−1)·r_borrow``.

Usage: ``python leveraged_pt_backtest.py [market_key ...]``
"""
import os
import sys

import pandas as pd
from helpers import RESULTS_DIR, build_leveraged_frame, leveraged_observations, leveraged_summary, load_registry, window

from fractal.strategies import MorphoLeveragedPT, MorphoLeveragedPTParams

TARGET_LTV = 0.80
BAND = (0.70, 0.88)


def run_market(key: str, cfg: dict) -> dict:
    start, end = window(cfg)
    frame, expiry = build_leveraged_frame(cfg, start, end)
    bar_hours = 24 if cfg["bar"] == "day" else 1
    strategy = MorphoLeveragedPT(params=MorphoLeveragedPTParams(
        INITIAL_BALANCE=100_000.0, TARGET_LTV=TARGET_LTV, MAX_LOOPS=8, REBALANCE_LTV_BAND=BAND,
        MIN_HEALTH_FACTOR=1.03, MIN_CARRY_SPREAD=-0.05, MAX_BORROW_APY=0.40,
        CARRY_GATE_LOOKBACK_BARS=7 * 24 // bar_hours,  # a week of bars: Morpho rate spikes last hours, not weeks
        MIN_DAYS_TO_MATURITY_AT_ENTRY=1, BAR_HOURS=bar_hours,
        LLTV=float(cfg["lltv"]), PT_IMPACT_MODEL=cfg.get("pt_impact_model", "rate_spread"),
        PT_FEE_LN_RATE=0.001, PT_IMPACT_LN_RATE_PER_SHARE=0.075,
    ))
    observations = leveraged_observations(frame, cfg)
    result = strategy.run(observations)
    df = result.to_dataframe()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, f"leveraged_{key}.csv"), index=False)
    summary = leveraged_summary(key, cfg, frame, df, TARGET_LTV)
    summary["expiry"] = expiry
    return summary


if __name__ == "__main__":
    registry = load_registry()["leveraged"]
    keys = sys.argv[1:] or list(registry)
    rows = [run_market(k, registry[k]) for k in keys]
    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "validation.csv")
    if os.path.exists(path):
        previous = pd.read_csv(path)
        out = pd.concat([previous[~previous["market"].isin(out["market"])], out], ignore_index=True)
    out.to_csv(path, index=False)
    pd.set_option("display.width", 240)
    print(out.to_string(index=False))
