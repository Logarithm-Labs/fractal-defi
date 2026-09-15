"""Shared helpers for the Pendle PT backtests: market registry, observation
builders from the live loaders, closed-form carry and result summaries."""
import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from fractal.core.base import Observation
from fractal.core.base.time import SECONDS_PER_DAY, SECONDS_PER_YEAR
from fractal.core.entities import BorosGlobalState, HyperliquidGlobalState, MorphoGlobalState, PendlePTGlobalState
from fractal.core.entities.models.pendle_math import linear_discount_oracle_price
from fractal.loaders import (
    BinanceFundingLoader,
    BinancePriceLoader,
    BorosMarketLoader,
    MorphoMarketLoader,
    PendleMarketLoader,
)
from fractal.loaders.boros import get_market_info as boros_market_info
from fractal.loaders.pendle import get_market_info as pendle_market_info

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
_BAR_HOURS = {"hour": 1, "day": 24}


def load_registry() -> Dict[str, Dict[str, dict]]:
    with open(os.path.join(HERE, "markets.json"), encoding="utf-8") as fh:
        return json.load(fh)


def window(cfg: dict) -> tuple:
    """``(start, end)`` UTC datetimes from the registry (``days_back`` when ``start`` is null)."""
    now = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = pd.Timestamp(cfg["end"], tz="UTC").to_pydatetime() if cfg.get("end") else now - timedelta(hours=2)
    if cfg.get("start"):
        start = pd.Timestamp(cfg["start"], tz="UTC").to_pydatetime()
    else:
        start = end - timedelta(days=int(cfg.get("days_back", 55)))
    return start, end


# ------------------------------------------------------------ leveraged
def build_leveraged_frame(cfg: dict, start: datetime, end: datetime) -> tuple:
    """Join Pendle market history with the Morpho market history on one grid.

    Returns ``(frame, expiry)``. The oracle price follows ``cfg["oracle_model"]``:
    ``"market"`` — the lender marks PT at the Pendle market price;
    ``"linear"`` — ``1 − base_discount · t`` (``cfg["base_discount"]``).
    """
    bar = cfg["bar"]
    info = pendle_market_info(cfg["chain_id"], cfg["pendle_market"])
    pendle = PendleMarketLoader(cfg["pendle_market"], cfg["chain_id"], start, end, time_frame=bar,
                                expiry=info.expiry).read(with_run=True)
    morpho = MorphoMarketLoader(cfg["morpho_market"], cfg["chain_id"], start, end, resolution=_BAR_HOURS[bar],
                                interval="HOUR" if bar == "hour" else "DAY").read(with_run=True)
    frame = pendle.join(morpho, how="inner").sort_index()
    if frame.empty:
        raise RuntimeError("Pendle and Morpho histories do not overlap on the requested grid")
    if cfg.get("oracle_model", "market") == "linear":
        discount = float(cfg["base_discount"])
        frame["oracle_price"] = [linear_discount_oracle_price(discount, s) for s in frame["seconds_to_expiry"]]
    else:
        frame["oracle_price"] = frame["pt_price_asset"]
    frame = append_expiry_bar(frame, info.expiry, end)
    return frame, info.expiry


def append_expiry_bar(frame: pd.DataFrame, expiry: datetime, end: datetime) -> pd.DataFrame:
    """Pendle's history stops on the expiry day; add the maturity bar so the
    strategy can redeem at par (rates carried over from the last row)."""
    if expiry > end or frame.index[-1] >= pd.Timestamp(expiry):
        return frame
    last = frame.iloc[-1].copy()
    last["seconds_to_expiry"] = 0.0
    last["pt_price_asset"] = 1.0
    last["oracle_price"] = 1.0
    last.name = pd.Timestamp(expiry)
    return pd.concat([frame, last.to_frame().T.astype(frame.dtypes.to_dict(), errors="ignore")])


def sy_exchange_rate(row) -> float:
    """Accounting asset per SY from the USD marks: ``sy_usd / (pt_usd / pt_asset)``.

    Pendle publishes no SY exchange-rate history, but ``ptPrice`` (USD) over
    ``pt_price_asset`` is the asset's USD price, so the SY rate follows from
    ``syPrice``. Falls back to ``1.0`` when a mark is missing.
    """
    pt_usd, pt_asset, sy_usd = row.get("pt_price_usd"), row.get("pt_price_asset"), row.get("sy_price_usd")
    if any(v is None or not math.isfinite(v) or v <= 0 for v in (pt_usd, pt_asset, sy_usd)):
        return 1.0
    return float(sy_usd * pt_asset / pt_usd)


def leveraged_observations(frame: pd.DataFrame, cfg: dict) -> List[Observation]:
    observations = []
    for ts, row in frame.iterrows():
        observations.append(Observation(timestamp=ts.to_pydatetime(), states={
            "PT": PendlePTGlobalState(
                seconds_to_expiry=float(row["seconds_to_expiry"]), implied_apy=float(row["implied_apy"]),
                asset_price=1.0, sy_exchange_rate=sy_exchange_rate(row),
                total_pt=float(row["total_pt"]), total_sy=float(row["total_sy"]),
                scalar_root=float(cfg.get("scalar_root", 0.0)),
                ln_fee_rate_root=float(cfg.get("ln_fee_rate_root", 0.0)),
            ),
            "LENDING": MorphoGlobalState(
                collateral_price=float(row["oracle_price"]), debt_price=1.0,
                lending_rate=0.0, borrowing_rate=float(row["borrowing_rate"]),
                collateral_market_price=float(row["pt_price_asset"]),
            ),
        }))
    return observations


# --------------------------------------------------------------- hedged
def build_hedged_frame(cfg: dict, start: datetime, end: datetime) -> tuple:
    """Pendle (volatile PT) + Binance price/funding (+ Boros mark APR) on a daily or hourly grid."""
    bar = cfg["bar"]
    info = pendle_market_info(cfg["chain_id"], cfg["pendle_market"])
    pendle = PendleMarketLoader(cfg["pendle_market"], cfg["chain_id"], start, end, time_frame=bar,
                                expiry=info.expiry).read(with_run=True)
    interval = "1d" if bar == "day" else "1h"
    price = BinancePriceLoader(cfg["binance_ticker"], interval=interval, start_time=start,
                               end_time=end).read(with_run=True)
    funding = BinanceFundingLoader(cfg["binance_ticker"], start_time=start, end_time=end).read(with_run=True)
    grid_funding = funding["rate"].resample("1d" if bar == "day" else "1h").sum()  # cash-flow semantics
    frame = pendle.join(price.rename(columns={"price": "spot"}), how="inner")
    frame = frame.join(grid_funding.rename("funding_rate"), how="left")
    frame["funding_rate"] = frame["funding_rate"].fillna(0.0)  # a bar without a settlement carries no cash flow
    boros_maturity = None
    if cfg.get("boros_market"):
        binfo = boros_market_info(int(cfg["boros_market"]))
        boros = BorosMarketLoader(int(cfg["boros_market"]), start, end, time_frame="1d" if bar == "day" else "1h",
                                  maturity=binfo.maturity, include_settlements=False).read(with_run=True)
        frame = frame.join(boros[["mark_apr_close"]].rename(columns={"mark_apr_close": "boros_mark_apr"}), how="left")
        frame["boros_mark_apr"] = frame["boros_mark_apr"].ffill()
        frame["boros_seconds_to_expiry"] = [(binfo.maturity - ts).total_seconds() for ts in frame.index]
        boros_maturity = binfo.maturity
    frame = frame.dropna(subset=["spot", "implied_apy"]).sort_index()
    return frame, info.expiry, boros_maturity


def hedged_observations(frame: pd.DataFrame, cfg: dict, use_boros: bool) -> List[Observation]:
    bar_seconds = _BAR_HOURS[cfg["bar"]] * 3600
    observations = []
    for ts, row in frame.iterrows():
        states = {
            "PT": PendlePTGlobalState(
                seconds_to_expiry=float(row["seconds_to_expiry"]), implied_apy=float(row["implied_apy"]),
                asset_price=float(row["spot"]), sy_exchange_rate=sy_exchange_rate(row),
                total_pt=float(row["total_pt"]), total_sy=float(row["total_sy"]),
                scalar_root=float(cfg.get("scalar_root", 0.0)),
                ln_fee_rate_root=float(cfg.get("ln_fee_rate_root", 0.0)),
            ),
            "HEDGE": HyperliquidGlobalState(mark_price=float(row["spot"]), funding_rate=float(row["funding_rate"])),
        }
        if use_boros and not math.isnan(row.get("boros_mark_apr", float("nan"))):
            states["BOROS"] = BorosGlobalState(
                seconds_to_expiry=float(row["boros_seconds_to_expiry"]), mark_rate=float(row["boros_mark_apr"]),
                funding_rate=float(row["funding_rate"]), funding_period_seconds=float(bar_seconds),
                underlying_price=float(row["spot"]),
            )
        observations.append(Observation(timestamp=ts.to_pydatetime(), states=states))
    return observations


# ------------------------------------------------------------ analysis
def realised_apy(df: pd.DataFrame) -> float:
    """Compounded annual return of ``net_balance`` over the run."""
    start, end = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    years = (end - start).total_seconds() / SECONDS_PER_YEAR
    if years <= 0:
        return float("nan")
    return (df["net_balance"].iloc[-1] / df["net_balance"].iloc[0]) ** (1 / years) - 1


def closed_form_leveraged_apy(leverage: float, pt_apy: float, borrow_apy: float) -> float:
    """``L · y_pt − (L − 1) · r_borrow``."""
    return leverage * pt_apy - (leverage - 1.0) * borrow_apy


def leveraged_summary(key: str, cfg: dict, frame: pd.DataFrame, df: pd.DataFrame, target_ltv: float) -> dict:
    first = df.iloc[0]
    pt_units = first["LENDING_collateral"] + first["PT_amount"]
    p0 = float(frame["pt_price_asset"].iloc[0])
    equity0 = float(first["net_balance"])
    leverage = pt_units * p0 / equity0 if equity0 else float("nan")
    borrow_apys = frame["borrow_apy"] if "borrow_apy" in frame else pd.Series(dtype=float)
    return {
        "market": key, "pendle": cfg["pendle_name"], "morpho": cfg["morpho_name"], "bar": cfg["bar"],
        "bars": len(df), "start": df["timestamp"].iloc[0], "end": df["timestamp"].iloc[-1],
        "target_ltv": target_ltv, "leverage_at_entry": leverage,
        "pt_apy_at_entry": float(frame["implied_apy"].iloc[0]),
        "borrow_apy_mean": float(borrow_apys.mean()) if len(borrow_apys) else float("nan"),
        "realised_apy": realised_apy(df),
        "closed_form_apy": closed_form_leveraged_apy(leverage, float(frame["implied_apy"].iloc[0]),
                                                     float(borrow_apys.mean()) if len(borrow_apys) else 0.0),
        "final_equity": float(df["net_balance"].iloc[-1]), "liquidations": int(df["LENDING_liquidation_count"].max()),
        "min_health_factor": float((df["LENDING_collateral"] * df["LENDING_collateral_price"] * cfg["lltv"]
                                    / df["LENDING_borrowed"].replace(0, float("nan"))).min()),
    }


def days_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / SECONDS_PER_DAY


def env_data_path() -> Optional[str]:
    return os.environ.get("DATA_PATH") or None
