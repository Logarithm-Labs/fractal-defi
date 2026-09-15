"""Three variants of the sUSDe PT carry on every market in ``markets.json``:
the plain Morpho loop, the loop with a Boros long-YU floating leg, and the
loop with a delta-neutral Binance basis leg.

Writes ``results/<market>_<variant>.csv`` and ``results/validation.csv``.
Usage: ``python run.py [market_key ...]``
"""
import os
import sys

import pandas as pd
from helpers import RESULTS_DIR, VARIANTS, build_frame, load_registry, observations, summarise, window

from fractal.strategies import MorphoRateHedgedLeveragedPT, MorphoRateHedgedLeveragedPTParams

BASE = dict(INITIAL_BALANCE=100_000.0, TARGET_LTV=0.80, MAX_LOOPS=8, REBALANCE_LTV_BAND=(0.70, 0.88),
            MIN_HEALTH_FACTOR=1.03, MIN_CARRY_SPREAD=-0.05, MAX_BORROW_APY=0.40, MIN_DAYS_TO_MATURITY_AT_ENTRY=1,
            PT_FEE_LN_RATE=0.001, PT_IMPACT_LN_RATE_PER_SHARE=0.075,
            HEDGE_MARGIN_SHARE=0.20, HEDGE_RATIO=1.0, HEDGE_REBALANCE_THRESHOLD=0.05, PERP_TARGET_LEVERAGE=2.0)


def make_params(cfg: dict, variant: str, **overrides) -> dict:
    bar_hours = 24 if cfg["bar"] == "day" else 1
    params = dict(BASE, RATE_HEDGE=variant, CARRY_GATE_LOOKBACK_BARS=7 * 24 // bar_hours, BAR_HOURS=bar_hours,
                  LLTV=float(cfg["lltv"]), PT_IMPACT_MODEL=cfg.get("pt_impact_model", "rate_spread"))
    params.update(overrides)
    return params


def run_variant(key: str, cfg: dict, frame, variant: str, **overrides) -> tuple:
    params = make_params(cfg, variant, **overrides)
    strategy = MorphoRateHedgedLeveragedPT(params=MorphoRateHedgedLeveragedPTParams(**params))
    df = strategy.run(observations(frame, cfg, variant)).to_dataframe()
    return df, summarise(key, variant, cfg, frame, df, params)


if __name__ == "__main__":
    registry = load_registry()
    keys = sys.argv[1:] or list(registry)
    rows = []
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for key in keys:
        cfg = registry[key]
        frame, _, _ = build_frame(cfg, *window(cfg))
        for variant in VARIANTS:
            df, summary = run_variant(key, cfg, frame, variant)
            df.to_csv(os.path.join(RESULTS_DIR, f"{key}_{variant}.csv"), index=False)
            rows.append(summary)
    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "validation.csv")
    if os.path.exists(path):
        previous = pd.read_csv(path)
        done = set(zip(out["market"], out["variant"]))
        keep = previous[[(m, v) not in done for m, v in zip(previous["market"], previous["variant"])]]
        out = pd.concat([keep, out], ignore_index=True)
    out.to_csv(path, index=False)
    pd.set_option("display.width", 260)
    print(out[["market", "variant", "bars", "leverage_at_entry", "pt_apy_at_entry", "borrow_apy_mean",
               "funding_apr_mean", "boros_mark_apr_mean", "realised_apy", "max_drawdown", "hedge_coverage_mean",
               "hedge_pnl", "liquidations"]].round(4).to_string(index=False))
