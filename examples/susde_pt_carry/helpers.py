"""Frame and observation builders for the sUSDe PT carry example.

Reuses the Pendle + Morpho builders of ``examples/pendle_pt_backtests`` and
adds the Boros leg: the coin's price and per-bar Binance funding (the
floating settlement) and the mark APR of the matching yield-unit market.
"""
import json
import math
import os
from datetime import datetime

import pandas as pd

from examples.pendle_pt_backtests.helpers import build_leveraged_frame, sy_exchange_rate, window
from fractal.core.base import Observation
from fractal.core.base.strategy import StrategyResult
from fractal.core.entities import BorosGlobalState, MorphoGlobalState, PendlePTGlobalState
from fractal.loaders import BinanceFundingLoader, BinancePriceLoader, BorosMarketLoader
from fractal.loaders.boros import get_market_info as boros_market_info

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
_BAR_HOURS = {"hour": 1, "day": 24}
VARIANTS = ("none", "boros")

__all__ = ["RESULTS_DIR", "VARIANTS", "build_frame", "load_registry", "observations", "summarise", "window"]


def load_registry() -> dict[str, dict]:
    with open(os.path.join(HERE, "markets.json"), encoding="utf-8") as fh:
        return json.load(fh)


def build_frame(cfg: dict, start: datetime, end: datetime) -> tuple:
    """Pendle + Morpho grid joined with the coin's price, per-bar funding and the Boros mark.

    The frame runs to the later of the PT's expiry and the yield unit's
    maturity: after the PT is redeemed only the Boros columns are filled,
    so a unit held to settlement keeps settling.
    """
    bar = cfg["bar"]
    interval = "1d" if bar == "day" else "1h"
    binfo = boros_market_info(int(cfg["boros_market"]))
    leg_end = max(end, min(binfo.maturity, pd.Timestamp.now(tz="UTC").to_pydatetime()))
    frame, expiry = build_leveraged_frame(cfg, start, end)
    grid = pd.date_range(frame.index[0], leg_end, freq=interval, tz="UTC")
    frame = frame.reindex(frame.index.union(grid[grid > frame.index[-1]]))
    price = BinancePriceLoader(cfg["binance_ticker"], interval=interval, start_time=start,
                               end_time=leg_end).read(with_run=True)
    funding = BinanceFundingLoader(cfg["binance_ticker"], start_time=start, end_time=leg_end).read(with_run=True)
    frame = frame.join(price.rename(columns={"price": "spot"}), how="left")
    frame["spot"] = frame["spot"].ffill()
    frame = frame.join(funding["rate"].resample(interval).sum().rename("funding_rate"), how="left")
    frame["funding_rate"] = frame["funding_rate"].fillna(0.0)  # no settlement in this bar → no cash flow
    boros = BorosMarketLoader(int(cfg["boros_market"]), start, leg_end, time_frame=interval,
                              maturity=binfo.maturity, include_settlements=False).read(with_run=True)
    frame = frame.join(boros[["mark_apr_close"]].rename(columns={"mark_apr_close": "boros_mark_apr"}), how="left")
    frame["boros_mark_apr"] = frame["boros_mark_apr"].ffill()
    frame["boros_seconds_to_expiry"] = [max((binfo.maturity - ts).total_seconds(), 0.0) for ts in frame.index]
    return frame.dropna(subset=["spot"]), expiry, binfo


def observations(frame: pd.DataFrame, cfg: dict, variant: str) -> list[Observation]:
    bar_seconds = _BAR_HOURS[cfg["bar"]] * 3600
    out = []
    for ts, row in frame.iterrows():
        states = {}
        if not math.isnan(row["implied_apy"]):
            states["PT"] = PendlePTGlobalState(
                seconds_to_expiry=float(row["seconds_to_expiry"]), implied_apy=float(row["implied_apy"]),
                asset_price=1.0, sy_exchange_rate=sy_exchange_rate(row),
                total_pt=float(row["total_pt"]), total_sy=float(row["total_sy"]),
                scalar_root=float(cfg.get("scalar_root", 0.0)), ln_fee_rate_root=float(cfg.get("ln_fee_rate_root", 0.0)),
            )
            states["LENDING"] = MorphoGlobalState(
                collateral_price=float(row["oracle_price"]), debt_price=1.0, lending_rate=0.0,
                borrowing_rate=float(row["borrowing_rate"]), collateral_market_price=float(row["pt_price_asset"]),
            )
        if variant == "boros" and not math.isnan(row["boros_mark_apr"]):  # from listing on, through maturity
            states["BOROS"] = BorosGlobalState(
                seconds_to_expiry=float(row["boros_seconds_to_expiry"]), mark_rate=float(row["boros_mark_apr"]),
                funding_rate=float(row["funding_rate"]), funding_period_seconds=float(bar_seconds),
                underlying_price=float(row["spot"]),
            )
        if states:
            out.append(Observation(timestamp=ts.to_pydatetime(), states=states))
    return out


def summarise(key: str, variant: str, cfg: dict, frame: pd.DataFrame, result: StrategyResult, params: dict) -> dict:
    df, metrics = result.to_dataframe(), result.get_default_metrics()
    first = df.iloc[0]
    pt_units = first["LENDING_collateral"] + first["PT_amount"]
    leverage = pt_units * float(frame["pt_price_asset"].iloc[0]) / first["net_balance"]
    borrowed = df["LENDING_borrowed"]
    loop = frame.dropna(subset=["implied_apy"])
    hedge_pnl, coverage = 0.0, float("nan")
    if variant == "boros" and "BOROS_balance" in df:
        hedge_pnl = float(df["BOROS_balance"].ffill().iloc[-1] - params["INITIAL_BALANCE"] * params["HEDGE_MARGIN_SHARE"])
        pt_value = (df["PT_amount"] + df["LENDING_collateral"]) * df["LENDING_collateral_market_price"]
        cov = df["BOROS_size"].fillna(0) * df["BOROS_underlying_price"].fillna(0) / pt_value.replace(0, float("nan"))
        coverage = float(cov[pt_value > 0].mean())
    return {
        "market": key, "variant": variant, "pendle": cfg["pendle_name"], "morpho": cfg["morpho_name"],
        "boros": cfg["boros_name"] if variant == "boros" else "", "bar": cfg["bar"], "bars": len(df),
        "start": df["timestamp"].iloc[0], "end": df["timestamp"].iloc[-1],
        "target_ltv": params["TARGET_LTV"], "leverage_at_entry": leverage,
        "pt_apy_at_entry": float(loop["implied_apy"].iloc[0]), "borrow_apy_mean": float(loop["borrow_apy"].mean()),
        "funding_apr_mean": float(frame["funding_rate"].mean() * (365 * 24 / _BAR_HOURS[cfg["bar"]])),
        "boros_mark_apr_mean": float(frame["boros_mark_apr"].mean()),
        "realised_apy": metrics.cagr, "max_drawdown": metrics.max_drawdown, "sharpe": metrics.sharpe,
        "final_equity": float(df["net_balance"].iloc[-1]),
        "hedge_coverage_mean": coverage, "hedge_pnl": hedge_pnl,
        "liquidations": int(df["LENDING_liquidation_count"].max()),
        "min_health_factor": float((df["LENDING_collateral"] * df["LENDING_collateral_price"] * cfg["lltv"]
                                    / borrowed.replace(0, float("nan"))).min()),
        "debt_changes": int((borrowed.pct_change().abs() > 0.01).sum()),
    }
