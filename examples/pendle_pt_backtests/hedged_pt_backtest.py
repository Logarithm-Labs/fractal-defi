"""Run ``PerpHedgedPT`` on the markets in ``markets.json["hedged"]``, with and
without the Boros funding lock.

Writes ``results/hedged_<market>_{perp,boros}.csv`` and appends to
``results/validation.csv``.

Usage: ``python hedged_pt_backtest.py [market_key ...]``
"""
import os
import sys

import pandas as pd
from helpers import RESULTS_DIR, build_hedged_frame, hedged_observations, load_registry, realised_apy, window

from fractal.strategies import PerpHedgedPT, PerpHedgedPTParams


def run_market(key: str, cfg: dict) -> list:
    start, end = window(cfg)
    frame, expiry, boros_maturity = build_hedged_frame(cfg, start, end)
    rows = []
    for use_boros in (False, True):
        if use_boros and boros_maturity is None:
            continue
        strategy = PerpHedgedPT(params=PerpHedgedPTParams(
            INITIAL_BALANCE=100_000.0, TARGET_HEDGE_LEVERAGE=2.0, HEDGE_LEVERAGE_BAND=(1.2, 3.5),
            HEDGE_REBALANCE_THRESHOLD=0.02, USE_BOROS=use_boros, BOROS_MARGIN_SHARE=0.10,
            MIN_DAYS_TO_MATURITY_AT_ENTRY=1, PT_IMPACT_MODEL=cfg.get("pt_impact_model", "rate_spread"),
            PT_FEE_LN_RATE=0.001, PT_IMPACT_LN_RATE_PER_SHARE=0.075, PERP_TRADING_FEE=0.00035, PERP_MAX_LEVERAGE=10.0,
        ))
        run_frame = frame.dropna(subset=["boros_mark_apr"]) if use_boros else frame  # Boros leg needs a quote at entry
        observations = hedged_observations(run_frame, cfg, use_boros)
        df = strategy.run(observations).to_dataframe()
        tag = "boros" if use_boros else "perp"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(os.path.join(RESULTS_DIR, f"hedged_{key}_{tag}.csv"), index=False)
        funding_total = float((run_frame["funding_rate"]).sum())
        rows.append({
            "market": f"{key}_{tag}", "pendle": cfg["pendle_name"], "morpho": "", "bar": cfg["bar"],
            "bars": len(df), "start": df["timestamp"].iloc[0], "end": df["timestamp"].iloc[-1],
            "target_ltv": float("nan"), "leverage_at_entry": 2.0,
            "pt_apy_at_entry": float(run_frame["implied_apy"].iloc[0]), "borrow_apy_mean": float("nan"),
            "realised_apy": realised_apy(df), "closed_form_apy": float("nan"),
            "final_equity": float(df["net_balance"].iloc[-1]), "liquidations": 0,
            "min_health_factor": float("nan"),
            "spot_move": float(run_frame["spot"].iloc[-1] / run_frame["spot"].iloc[0] - 1),
            "funding_sum": funding_total, "boros_mark_apr_at_entry": float(run_frame["boros_mark_apr"].iloc[0])
            if use_boros else float("nan"), "expiry": expiry,
        })
    return rows


if __name__ == "__main__":
    registry = load_registry()["hedged"]
    keys = sys.argv[1:] or list(registry)
    rows = [row for k in keys for row in run_market(k, registry[k])]
    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "validation.csv")
    if os.path.exists(path):
        previous = pd.read_csv(path)
        out = pd.concat([previous[~previous["market"].isin(out["market"])], out], ignore_index=True)
    out.to_csv(path, index=False)
    pd.set_option("display.width", 240)
    print(out.to_string(index=False))
