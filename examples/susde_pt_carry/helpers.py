"""Frame and observation builders for the sUSDe PT carry example.

Reuses the Pendle + Morpho builders of ``examples/pendle_pt_backtests`` and
adds the floating-rate legs: Binance ETHUSDT price and funding (per-bar cash
flow) and the Boros mark APR of the matching yield-unit market.
"""
import importlib.util
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List

import pandas as pd

from fractal.core.base import Observation
from fractal.core.entities import (
    BorosGlobalState,
    HyperliquidGlobalState,
    MorphoGlobalState,
    PendlePTGlobalState,
    SimpleSpotExchangeGlobalState,
)
from fractal.loaders import BinanceFundingLoader, BinancePriceLoader, BorosMarketLoader
from fractal.loaders.boros import get_market_info as boros_market_info

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


def _sibling_helpers():
    """Load ``examples/pendle_pt_backtests/helpers.py`` under its own module name."""
    path = os.path.join(HERE, "..", "pendle_pt_backtests", "helpers.py")
    spec = importlib.util.spec_from_file_location("pendle_pt_backtests_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_pt = _sibling_helpers()
build_leveraged_frame, realised_apy, sy_exchange_rate, window = (
    _pt.build_leveraged_frame, _pt.realised_apy, _pt.sy_exchange_rate, _pt.window)

_BAR_HOURS = {"hour": 1, "day": 24}
VARIANTS = ("none", "boros", "perp")


def load_registry() -> Dict[str, dict]:
    with open(os.path.join(HERE, "markets.json"), encoding="utf-8") as fh:
        return json.load(fh)


def build_frame(cfg: dict, start: datetime, end: datetime) -> tuple:
    """Pendle + Morpho grid joined with the coin's price, per-bar funding and the Boros mark."""
    bar = cfg["bar"]
    frame, expiry = build_leveraged_frame(cfg, start, end)
    interval = "1d" if bar == "day" else "1h"
    price = BinancePriceLoader(cfg["binance_ticker"], interval=interval, start_time=start,
                               end_time=end).read(with_run=True)
    funding = BinanceFundingLoader(cfg["binance_ticker"], start_time=start, end_time=end).read(with_run=True)
    grid_funding = funding["rate"].resample(interval).sum()
    frame = frame.join(price.rename(columns={"price": "spot"}), how="left")
    frame["spot"] = frame["spot"].ffill()
    frame = frame.join(grid_funding.rename("funding_rate"), how="left")
    frame["funding_rate"] = frame["funding_rate"].fillna(0.0)  # no settlement in this bar → no cash flow
    binfo = boros_market_info(int(cfg["boros_market"]))
    boros = BorosMarketLoader(int(cfg["boros_market"]), start, end, time_frame=interval,
                              maturity=binfo.maturity, include_settlements=False).read(with_run=True)
    frame = frame.join(boros[["mark_apr_close"]].rename(columns={"mark_apr_close": "boros_mark_apr"}), how="left")
    frame["boros_mark_apr"] = frame["boros_mark_apr"].ffill()
    frame["boros_seconds_to_expiry"] = [max((binfo.maturity - ts).total_seconds(), 0.0) for ts in frame.index]
    frame = frame.dropna(subset=["spot"])
    return frame, expiry, binfo


def observations(frame: pd.DataFrame, cfg: dict, variant: str) -> List[Observation]:
    bar_seconds = _BAR_HOURS[cfg["bar"]] * 3600
    out = []
    for ts, row in frame.iterrows():
        states = {
            "PT": PendlePTGlobalState(
                seconds_to_expiry=float(row["seconds_to_expiry"]), implied_apy=float(row["implied_apy"]),
                asset_price=1.0, sy_exchange_rate=sy_exchange_rate(row),
                total_pt=float(row["total_pt"]), total_sy=float(row["total_sy"]),
                scalar_root=float(cfg.get("scalar_root", 0.0)), ln_fee_rate_root=float(cfg.get("ln_fee_rate_root", 0.0)),
            ),
            "LENDING": MorphoGlobalState(
                collateral_price=float(row["oracle_price"]), debt_price=1.0, lending_rate=0.0,
                borrowing_rate=float(row["borrowing_rate"]), collateral_market_price=float(row["pt_price_asset"]),
            ),
        }
        spot, funding = float(row["spot"]), float(row["funding_rate"])
        if variant == "boros" and not math.isnan(row["boros_mark_apr"]) and row["boros_seconds_to_expiry"] > 0:
            states["BOROS"] = BorosGlobalState(
                seconds_to_expiry=float(row["boros_seconds_to_expiry"]), mark_rate=float(row["boros_mark_apr"]),
                funding_rate=funding, funding_period_seconds=float(bar_seconds), underlying_price=spot,
            )
        if variant == "perp":
            states["SPOT"] = SimpleSpotExchangeGlobalState(open=spot, high=spot, low=spot, close=spot, volume=0.0)
            states["PERP"] = HyperliquidGlobalState(mark_price=spot, funding_rate=funding)
        out.append(Observation(timestamp=ts.to_pydatetime(), states=states))
    return out


def summarise(key: str, variant: str, cfg: dict, frame: pd.DataFrame, df: pd.DataFrame, params: dict) -> dict:
    first = df.iloc[0]
    pt_units = first["LENDING_collateral"] + first["PT_amount"]
    leverage = pt_units * float(frame["pt_price_asset"].iloc[0]) / first["net_balance"]
    borrowed = df["LENDING_borrowed"]
    hedge_pnl = 0.0
    coverage = float("nan")
    if variant == "boros" and "BOROS_balance" in df:
        bal = df["BOROS_balance"].ffill()
        hedge_pnl = float(bal.iloc[-1] - params["INITIAL_BALANCE"] * params["HEDGE_MARGIN_SHARE"])
        cov = df["BOROS_size"].fillna(0) * df["BOROS_underlying_price"].fillna(0) / borrowed.replace(0, float("nan"))
        coverage = float(cov[borrowed > 0].mean())
    elif variant == "perp":
        bal = df["SPOT_balance"] + df["PERP_balance"]
        hedge_pnl = float(bal.iloc[-1] - params["INITIAL_BALANCE"] * params["HEDGE_MARGIN_SHARE"])
        cov = -df["PERP_positions_0_amount"].fillna(0) * df["PERP_mark_price"] / borrowed.replace(0, float("nan"))
        coverage = float(cov[borrowed > 0].mean())
    return {
        "market": key, "variant": variant, "pendle": cfg["pendle_name"], "morpho": cfg["morpho_name"],
        "hedge": cfg["boros_name"] if variant == "boros" else (cfg["binance_ticker"] + " basis" if variant == "perp" else ""),
        "bar": cfg["bar"], "bars": len(df), "start": df["timestamp"].iloc[0], "end": df["timestamp"].iloc[-1],
        "target_ltv": params["TARGET_LTV"], "leverage_at_entry": leverage,
        "pt_apy_at_entry": float(frame["implied_apy"].iloc[0]), "borrow_apy_mean": float(frame["borrow_apy"].mean()),
        "funding_apr_mean": float(frame["funding_rate"].mean() * (365 * 24 / _BAR_HOURS[cfg["bar"]])),
        "boros_mark_apr_mean": float(frame["boros_mark_apr"].mean()),
        "realised_apy": realised_apy(df), "final_equity": float(df["net_balance"].iloc[-1]),
        "max_drawdown": float((df["net_balance"] / df["net_balance"].cummax() - 1).min()),
        "hedge_coverage_mean": coverage, "hedge_pnl": hedge_pnl,
        "liquidations": int(df["LENDING_liquidation_count"].max()),
        "min_health_factor": float((df["LENDING_collateral"] * df["LENDING_collateral_price"] * cfg["lltv"]
                                    / borrowed.replace(0, float("nan"))).min()),
        "debt_changes": int((borrowed.pct_change().abs() > 0.01).sum()),
    }
