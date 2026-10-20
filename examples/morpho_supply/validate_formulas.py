"""Formula consistency checks against the live Morpho API.

Three independent checks, each reported in basis points:

1. supply formula — ``supplyApy == borrowApy · (1 − fee) · u``
2. IRM curve      — ``borrowApy == exp(r(u) · YEAR) − 1`` where ``r`` is
   :func:`fractal.core.entities.models.morpho_math.adaptive_curve_borrow_apr`
   fed the market's per-second ``rateAtTarget`` (it is unit-preserving; the
   API's value is a per-second WAD BigInt)
3. liquidity identity — ``liquidityAssetsUsd == (1 − u) · supplyAssetsUsd``

Runs live (no API key) against ``https://api.morpho.org/graphql`` and only
reads the current ``state`` of each market, so the list query is enough.

Usage::

    python validate_formulas.py [n_markets]      # default 100

Results are provider-backed consistency checks, not independent protocol
validation or strategy performance evidence. CSV output stays in the gitignored
``results/`` directory.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from fractal.core.entities.models.morpho_math import (
    adaptive_curve_borrow_apr,
    borrow_apy_from_per_second_rate,
    from_wad,
)
from fractal.loaders._http import HttpClient
from fractal.loaders.morpho import graphql_query

LOAN_SYMBOLS = {"USDC", "USDT"}
STUCK_UTILIZATION = 0.999

_MARKETS_QUERY = """
query Markets($first: Int!, $skip: Int!) {
  markets(first: $first, skip: $skip, orderBy: SupplyAssetsUsd, orderDirection: Desc) {
    items {
      marketId lltv loanAsset { symbol } collateralAsset { symbol }
      state { fee borrowApy supplyApy utilization rateAtTarget
              supplyAssetsUsd borrowAssetsUsd liquidityAssetsUsd }
    }
  }
}
"""


def fetch_markets(http: HttpClient, limit: int, page: int = 100) -> list[dict]:
    """Page the market list (descending supply size) until ``limit`` are read."""
    out: list[dict] = []
    skip = 0
    while len(out) < limit:
        data = graphql_query(http, _MARKETS_QUERY, {"first": page, "skip": skip})
        items = (data.get("markets") or {}).get("items") or []
        if not items:
            break
        out.extend(items)
        skip += page
    return out[:limit]


def _bp(residual: float) -> float:
    return abs(residual) * 1e4


def validate(markets: list[dict]) -> pd.DataFrame:
    """Compute the three residuals per market, in basis points."""
    rows = []
    for market in markets:
        state = market.get("state") or {}
        loan = (market.get("loanAsset") or {}).get("symbol")
        if loan not in LOAN_SYMBOLS:
            continue
        utilization = state.get("utilization")
        borrow_apy, supply_apy = state.get("borrowApy"), state.get("supplyApy")
        fee = state.get("fee")
        supply_usd = state.get("supplyAssetsUsd")
        liquidity_usd = state.get("liquidityAssetsUsd")
        rate_at_target = state.get("rateAtTarget")
        if None in (utilization, borrow_apy, supply_apy, supply_usd, liquidity_usd):
            continue
        row = {
            "market_id": market.get("marketId"),
            "loan": loan,
            "collateral": (market.get("collateralAsset") or {}).get("symbol"),
            "utilization": utilization,
            "supply_assets_usd": supply_usd,
            "liquidity_assets_usd": liquidity_usd,
            "stuck": bool(utilization >= STUCK_UTILIZATION or (liquidity_usd or 0.0) <= 0.0),
            "supply_formula_bp": np.nan,
            "irm_curve_bp": np.nan,
            "liquidity_identity_bp": np.nan,
        }
        if fee is not None:
            predicted = borrow_apy * (1.0 - fee) * utilization
            row["supply_formula_bp"] = _bp(supply_apy - predicted)
        if rate_at_target is not None:
            # ``rateAtTarget`` is a per-second WAD rate; the curve is unit-preserving.
            rate_per_second = adaptive_curve_borrow_apr(utilization, from_wad(rate_at_target))
            row["irm_curve_bp"] = _bp(borrow_apy - borrow_apy_from_per_second_rate(rate_per_second))
        row["liquidity_identity_bp"] = _bp(liquidity_usd - (1.0 - utilization) * supply_usd) / max(supply_usd, 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, results_dir: str) -> dict:
    """Print the median/p95 residuals per check and write the per-market CSV.

    Reported twice: over every market, and over the **liquid** stratum only.
    The stuck stratum (``u >= 0.999``, no withdrawable liquidity) is exactly
    where the API's lazy accrual leaves stale ``borrowApy`` values behind —
    at ``u = 1`` the IRM sits at its saturated ``rateAtTarget`` and the curve
    predicts a rate two orders of magnitude above what a dead market reports.
    That stratum is not tradable (the exit gate rejects it), so the curve
    validation is only meaningful on the liquid one.
    """
    os.makedirs(results_dir, exist_ok=True)
    frame.to_csv(os.path.join(results_dir, "formula_validation.csv"), index=False)
    summary = {
        "markets": len(frame),
        "stuck_markets": int(frame["stuck"].sum()),
        "all": _stratum(frame),
        "liquid_only": _stratum(frame[~frame["stuck"]]),
    }
    print(f"markets: {summary['markets']} (stuck at u>= {STUCK_UTILIZATION}: {summary['stuck_markets']})")
    for stratum, stats in (("all", summary["all"]), ("liquid", summary["liquid_only"])):
        print(f"-- {stratum}: n={stats['n']}")
        for name in ("supply_formula_bp", "irm_curve_bp", "liquidity_identity_bp"):
            row = stats[name]
            print(f"   {name:22s} n={row['n']:4d}  median={row['median']:12.4f}  p95={row['p95']:12.4f}  max={row['max']:14.4f}")
    print(f"csv: {os.path.join(results_dir, 'formula_validation.csv')}")
    return summary


def _stratum(frame: pd.DataFrame) -> dict:
    return {
        "n": len(frame),
        "supply_formula_bp": _stats(frame["supply_formula_bp"]),
        "irm_curve_bp": _stats(frame["irm_curve_bp"]),
        "liquidity_identity_bp": _stats(frame["liquidity_identity_bp"]),
    }


def _stats(series: pd.Series) -> dict:
    values = series.dropna().to_numpy()
    if values.size == 0:
        return {"n": 0, "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    here = os.path.dirname(os.path.abspath(__file__))
    frame = validate(fetch_markets(HttpClient(), limit))
    summary = summarize(frame, os.path.join(here, "results"))
    print(json.dumps({"summary": summary}, indent=1))


if __name__ == "__main__":
    main()
