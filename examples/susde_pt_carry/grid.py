"""Parameter grid for the sUSDe PT carry: target LTV × hedge kind × hedge
margin share × hedge ratio. Writes ``results/grid_<market>.csv``.

Usage: ``python grid.py [market_key ...]``
"""
import itertools
import os
import sys

import pandas as pd
from helpers import RESULTS_DIR, build_frame, load_registry, window
from run import run_variant

LTVS = (0.60, 0.70, 0.80, 0.86)
SHARES = (0.10, 0.30)
RATIOS = (0.5, 1.0)


def grid(key: str, cfg: dict) -> pd.DataFrame:
    frame, _, _ = build_frame(cfg, *window(cfg))
    rows = []
    for ltv in LTVS:
        band = (round(ltv - 0.10, 2), round(min(ltv + 0.08, 0.90), 2))
        _, row = run_variant(key, cfg, frame, "none", TARGET_LTV=ltv, REBALANCE_LTV_BAND=band)
        rows.append(dict(row, hedge_margin_share=0.0, hedge_ratio=0.0))
        for variant, share, ratio in itertools.product(("boros", "perp"), SHARES, RATIOS):
            _, row = run_variant(key, cfg, frame, variant, TARGET_LTV=ltv, REBALANCE_LTV_BAND=band,
                                 HEDGE_MARGIN_SHARE=share, HEDGE_RATIO=ratio)
            rows.append(dict(row, hedge_margin_share=share, hedge_ratio=ratio))
            print(f"{key} ltv={ltv} {variant} share={share} ratio={ratio} apy={row['realised_apy']:+.4f} "
                  f"cov={row['hedge_coverage_mean']:.2f} mdd={row['max_drawdown']:+.4f}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    registry = load_registry()
    keys = sys.argv[1:] or list(registry)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for key in keys:
        out = grid(key, registry[key])
        out.to_csv(os.path.join(RESULTS_DIR, f"grid_{key}.csv"), index=False)
        pd.set_option("display.width", 260)
        print(out[["variant", "target_ltv", "hedge_margin_share", "hedge_ratio", "leverage_at_entry", "realised_apy",
                   "max_drawdown", "hedge_coverage_mean", "hedge_pnl", "liquidations"]].round(4).to_string(index=False))
