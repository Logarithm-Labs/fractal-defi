"""Parameter sensitivity of ``MorphoLeveragedPT`` on one leveraged market:
target LTV, loop count, carry-gate smoothing and the lender's oracle model.

Writes ``results/sensitivity_<market>.csv`` and prints the grid.

Usage: ``python sensitivity.py [market_key]`` (default ``usde_25sep2025``)
"""
import os
import sys

import pandas as pd
from helpers import (
    RESULTS_DIR,
    build_leveraged_frame,
    leveraged_observations,
    leveraged_summary,
    linear_discount_oracle_price,
    load_registry,
    window,
)

from fractal.strategies import MorphoLeveragedPT, MorphoLeveragedPTParams

GRID = [  # (label, oracle, parameter overrides)
    ("base", "market", {}),
    ("unlevered", "market", {"MAX_LOOPS": 0}),
    ("ltv_0.50", "market", {"TARGET_LTV": 0.50}),
    ("ltv_0.60", "market", {"TARGET_LTV": 0.60}),
    ("ltv_0.70", "market", {"TARGET_LTV": 0.70}),
    ("ltv_0.86", "market", {"TARGET_LTV": 0.86}),
    ("loops_2", "market", {"MAX_LOOPS": 2}),
    ("loops_4", "market", {"MAX_LOOPS": 4}),
    ("flash", "market", {"MULTIPLY_MODE": "flash"}),
    ("gate_raw", "market", {"CARRY_GATE_LOOKBACK_BARS": 1}),
    ("no_gate", "market", {"MIN_CARRY_SPREAD": -1.0, "MAX_BORROW_APY": 10.0}),
    ("oracle_linear_6pct", "linear", {}),
]


def run_grid(key: str, cfg: dict) -> pd.DataFrame:
    start, end = window(cfg)
    frame, _ = build_leveraged_frame(cfg, start, end)
    bar_hours = 24 if cfg["bar"] == "day" else 1
    rows = []
    for label, oracle, overrides in GRID:
        run_frame = frame.copy()
        if oracle == "linear":  # a PendleSparkLinearDiscountOracle at 6 %/yr
            run_frame["oracle_price"] = [linear_discount_oracle_price(0.06, s) for s in run_frame["seconds_to_expiry"]]
        target = overrides.get("TARGET_LTV", 0.80)
        params = dict(INITIAL_BALANCE=100_000.0, TARGET_LTV=target, MAX_LOOPS=8,
                      REBALANCE_LTV_BAND=(round(target - 0.10, 2), round(min(target + 0.08, 0.90), 2)),
                      MIN_HEALTH_FACTOR=1.03, MIN_CARRY_SPREAD=-0.05, MAX_BORROW_APY=0.40,
                      CARRY_GATE_LOOKBACK_BARS=7 * 24 // bar_hours, MIN_DAYS_TO_MATURITY_AT_ENTRY=1,
                      BAR_HOURS=bar_hours, LLTV=float(cfg["lltv"]),
                      PT_IMPACT_MODEL=cfg.get("pt_impact_model", "rate_spread"),
                      PT_FEE_LN_RATE=0.001, PT_IMPACT_LN_RATE_PER_SHARE=0.075)
        params.update(overrides)
        df = MorphoLeveragedPT(params=MorphoLeveragedPTParams(**params)).run(
            leveraged_observations(run_frame, cfg)).to_dataframe()
        summary = leveraged_summary(key, cfg, run_frame, df, target)
        borrowed = df["LENDING_borrowed"]
        rows.append({
            "market": key, "variant": label, "oracle": oracle, "leverage_at_entry": summary["leverage_at_entry"],
            "realised_apy": summary["realised_apy"], "closed_form_apy": summary["closed_form_apy"],
            "max_drawdown": float((df["net_balance"] / df["net_balance"].cummax() - 1).min()),
            "min_health_factor": summary["min_health_factor"], "liquidations": summary["liquidations"],
            "debt_changes": int((borrowed.pct_change().abs() > 0.01).sum()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    registry = load_registry()["leveraged"]
    key = sys.argv[1] if len(sys.argv) > 1 else "usde_25sep2025"
    out = run_grid(key, registry[key])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out.to_csv(os.path.join(RESULTS_DIR, f"sensitivity_{key}.csv"), index=False)
    pd.set_option("display.width", 240)
    print(out.round(4).to_string(index=False))
