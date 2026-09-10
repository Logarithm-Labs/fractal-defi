"""Pendle Boros loaders: market config and implied-APR / settlement history.

API: ``https://api-boros.pendle.finance/apis/v1`` (no key, 200 CU/min).

* ``GET /markets?isMatured=false|true`` — every market with its margin
  and fee parameters (``config.kIM/kMM/tThresh/takerFee``,
  ``extConfig.paymentPeriod/settleFeeRate``, ``imData.maturity``,
  ``metadata.maxLeverage``).
* ``GET /markets/ohlcv?marketId&timeFrame=1h&startTimestamp&endTimestamp``
  — implied-APR candles, ≤ 200 per call.
* ``POST /funding-rate/settlement-summary`` — one row per on-chain
  settlement (``settlementApr``, open interest), ≤ 5000 per call.
* ``GET /markets/historical-underlying-apr?assetSymbol&exchange&timeFrame``
  — the venue's annualised funding since before the market listed.

The floating leg a strategy settles against is the venue's plain
funding rate (``settlementApr == rate · YEAR / period``), so backtests
take it from :class:`BinanceFundingLoader` / Hyperliquid loaders; this
loader supplies the fixed side (mark APR) and the realised settlements.
"""
import time as _time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from fractal.loaders._dt import require_no_nan, to_seconds, to_utc, utcnow
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import BorosMarketHistory

BOROS_API = "https://api-boros.pendle.finance/apis/v1"
_REQUEST_SLEEP_SECONDS = 0.35
_TIME_FRAMES = {"5m": 300, "1h": 3600, "1d": 86_400, "1w": 7 * 86_400}
_OHLCV_CAP = 200
_SETTLEMENT_CAP = 5000
_TICK_BASE = 1.00005

__all__ = [
    "BOROS_API",
    "bar_seconds",
    "BorosLoaderException",
    "BorosMarketInfo",
    "get_market_info",
    "BorosMarketLoader",
]


class BorosLoaderException(RuntimeError):
    """Malformed or missing Boros API data."""


def _rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "data", "markets"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return inner
    return []


def _num(value: Any, default: float = 0.0) -> float:
    """Numeric field that may arrive as a decimal or an 18-dec WAD string."""
    if value is None:
        return default
    number = float(value)
    return number / 1e18 if abs(number) >= 1e12 else number


def _epoch(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value // 1000) if value > 10 ** 11 else int(value)
    return int(pd.Timestamp(value, tz="UTC").timestamp())


# --------------------------------------------------------- market info
@dataclass(frozen=True)
class BorosMarketInfo:
    """Margin, fee and cadence parameters of one Boros market."""
    market_id: int
    symbol: str
    maturity: datetime
    payment_period_seconds: float
    k_im: float
    k_mm: float
    time_threshold_seconds: float
    taker_fee_rate: float
    settle_fee_rate: float
    rate_floor: float
    max_leverage: float
    funding_rate_symbol: Optional[str] = None
    collateral_symbol: Optional[str] = None
    is_matured: bool = False

    @property
    def mm_to_im_ratio(self) -> float:
        return self.k_mm / self.k_im if self.k_im else 1.0


def _parse_market(raw: Dict[str, Any]) -> BorosMarketInfo:
    im_data, config = raw.get("imData") or {}, raw.get("config") or {}
    ext, meta = raw.get("extConfig") or {}, raw.get("metadata") or {}
    tick_step = int(im_data.get("tickStep") or 1)
    i_tick_thresh = im_data.get("iTickThresh")
    rate_floor = (_TICK_BASE ** (int(i_tick_thresh) * tick_step) - 1.0) if i_tick_thresh is not None else 0.0
    k_im = _num(config.get("kIM"))
    max_leverage = _num(meta.get("maxLeverage")) or (1.0 / k_im if k_im else 0.0)
    return BorosMarketInfo(
        market_id=int(raw.get("marketId", raw.get("id"))),
        symbol=str(im_data.get("symbol") or raw.get("symbol") or ""),
        maturity=datetime.fromtimestamp(_epoch(im_data.get("maturity") or raw.get("maturity")), tz=utcnow().tzinfo),
        payment_period_seconds=float(ext.get("paymentPeriod") or 0.0),
        k_im=k_im,
        k_mm=_num(config.get("kMM")),
        time_threshold_seconds=float(config.get("tThresh") or 0.0),
        taker_fee_rate=_num(config.get("takerFee")),
        settle_fee_rate=_num(ext.get("settleFeeRate")),
        rate_floor=rate_floor,
        max_leverage=max_leverage,
        funding_rate_symbol=meta.get("fundingRateSymbol"),
        collateral_symbol=raw.get("collateralSymbol") or meta.get("collateralSymbol"),
        is_matured=bool(raw.get("isMatured", False)),
    )


def get_market_info(market_id: int, http: Optional[HttpClient] = None) -> BorosMarketInfo:
    """Find one market in the live list, then in the matured list."""
    client = http or HttpClient()
    for matured in ("false", "true"):
        payload = client.get(f"{BOROS_API}/markets", params={"isMatured": matured, "limit": 200})
        for raw in _rows(payload):
            if int(raw.get("marketId", raw.get("id", -1))) == int(market_id):
                info = _parse_market(raw)
                return info if matured == "false" else BorosMarketInfo(**{**info.__dict__, "is_matured": True})
        _time.sleep(_REQUEST_SLEEP_SECONDS)
    raise BorosLoaderException(f"Boros market {market_id} not found")


# ------------------------------------------------------ market history
class BorosMarketLoader(Loader):
    """Implied-APR candles + realised settlements → :class:`BorosMarketHistory`.

    Args:
        market_id: Boros market id (integer).
        start_time / end_time: inclusive UTC window.
        time_frame: candle size (``"1h"`` default).
        maturity: market maturity; fetched via :func:`get_market_info` when omitted.
        include_settlements: also pull ``settlement-summary`` rows and join them.
        underlying: ``(asset_symbol, exchange)`` to fetch
            ``historical-underlying-apr`` for bars without a settlement.
        http: injectable client for offline tests.
    """

    def __init__(
        self,
        market_id: int,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        *,
        time_frame: str = "1h",
        maturity: Optional[datetime] = None,
        include_settlements: bool = True,
        underlying: Optional[tuple] = None,
        loader_type: LoaderType = LoaderType.CSV,
        http: Optional[HttpClient] = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if time_frame not in _TIME_FRAMES:
            raise ValueError(f"time_frame must be one of {sorted(_TIME_FRAMES)}, got {time_frame!r}")
        self.market_id = int(market_id)
        self.start_time = to_utc(start_time)
        self.end_time = to_utc(end_time) if end_time is not None else utcnow()
        if self.end_time < self.start_time:
            raise ValueError(f"end_time {self.end_time} precedes start_time {self.start_time}")
        self.time_frame = time_frame
        self.include_settlements = include_settlements
        self.underlying = underlying
        self._maturity: Optional[datetime] = to_utc(maturity) if maturity is not None else None
        self._http = http or HttpClient()
        self._candles: List[Dict[str, Any]] = []
        self._settlements: List[Dict[str, Any]] = []
        self._underlying: List[Dict[str, Any]] = []

    @property
    def maturity(self) -> datetime:
        if self._maturity is None:
            self._maturity = get_market_info(self.market_id, http=self._http).maturity
        return self._maturity

    def _cache_key(self) -> str:
        tail = "st" if self.include_settlements else "nost"
        if self.underlying:
            tail += f"-{self.underlying[0]}-{self.underlying[1]}".lower()
        return (
            f"m{self.market_id}-{self.time_frame}-{to_seconds(self.start_time)}-{to_seconds(self.end_time)}-"
            f"{to_seconds(self.maturity)}-{tail}"
        )

    # ----------------------------------------------------------- fetch
    def _fetch_candles(self) -> List[Dict[str, Any]]:
        window = _OHLCV_CAP * _TIME_FRAMES[self.time_frame]
        rows: List[Dict[str, Any]] = []
        cursor = to_seconds(self.start_time)
        end = to_seconds(self.end_time)
        while cursor <= end:
            payload = self._http.get(f"{BOROS_API}/markets/ohlcv", params={
                "marketId": self.market_id, "timeFrame": self.time_frame,
                "startTimestamp": cursor, "endTimestamp": min(cursor + window - 1, end),
            })
            rows.extend(_rows(payload))
            cursor += window
            _time.sleep(_REQUEST_SLEEP_SECONDS)
        return rows

    def _fetch_settlements(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        end = to_seconds(self.end_time)
        start = to_seconds(self.start_time)
        while end >= start:
            payload = self._http.post(f"{BOROS_API}/funding-rate/settlement-summary", json={
                "marketIds": [self.market_id], "fromTimestamp": start, "toTimestamp": end,
            })
            batch = _rows(payload)
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < _SETTLEMENT_CAP:
                break
            oldest = min(_epoch(r.get("periodTimestamp", r.get("timestamp"))) for r in batch)
            if oldest >= end:
                break
            end = oldest - 1
            _time.sleep(_REQUEST_SLEEP_SECONDS)
        return rows

    def _fetch_underlying(self) -> List[Dict[str, Any]]:
        asset, exchange = self.underlying
        payload = self._http.get(f"{BOROS_API}/markets/historical-underlying-apr", params={
            "assetSymbol": asset, "exchange": exchange, "timeFrame": _TIME_FRAMES[self.time_frame],
            "startTimestamp": to_seconds(self.start_time), "endTimestamp": to_seconds(self.end_time),
        })
        return _rows(payload)

    def extract(self) -> None:
        self._candles = self._fetch_candles()
        self._settlements = self._fetch_settlements() if self.include_settlements else []
        self._underlying = self._fetch_underlying() if self.underlying else []

    # ------------------------------------------------------- transform
    def transform(self) -> None:
        cols = ["time", *BorosMarketHistory.COLUMNS]
        if not self._candles:
            self._data = pd.DataFrame(columns=cols)
            return
        raw = pd.DataFrame(self._candles)
        df = pd.DataFrame({"time": raw["ts"].map(_epoch).astype("int64")})
        for col, key in (("mark_apr_open", "o"), ("mark_apr_high", "h"), ("mark_apr_low", "l"),
                         ("mark_apr_close", "c")):
            df[col] = pd.to_numeric(raw.get(key), errors="coerce")
        df["volume"] = pd.to_numeric(raw.get("v"), errors="coerce").fillna(0.0) if "v" in raw else 0.0
        df = df.sort_values("time").drop_duplicates("time", keep="last")
        df = df[(df["time"] >= to_seconds(self.start_time)) & (df["time"] <= to_seconds(self.end_time))]
        df = df.reset_index(drop=True)

        df["settlement_apr"] = float("nan")
        df["oi"] = float("nan")
        if self._settlements:
            st = pd.DataFrame(self._settlements)
            key = "periodTimestamp" if "periodTimestamp" in st.columns else "timestamp"
            st = pd.DataFrame({
                "time": st[key].map(_epoch).astype("int64"),
                "settlement_apr": pd.to_numeric(st.get("settlementApr"), errors="coerce"),
                "oi": pd.to_numeric(st.get("totalNotionalSize"), errors="coerce"),
            }).drop_duplicates("time", keep="last")
            df = df.drop(columns=["settlement_apr", "oi"]).merge(st, on="time", how="left")

        df["underlying_apr"] = df["settlement_apr"]
        if self._underlying:
            un = pd.DataFrame(self._underlying)
            key = "periodStartTimestamp" if "periodStartTimestamp" in un.columns else "timestamp"
            un = pd.DataFrame({
                "time": un[key].map(_epoch).astype("int64"),
                "u": pd.to_numeric(un.get("underlyingApr"), errors="coerce"),
            }).drop_duplicates("time", keep="last")
            df = df.merge(un, on="time", how="left")
            df["underlying_apr"] = df["underlying_apr"].fillna(df["u"])
            df = df.drop(columns=["u"])

        df["seconds_to_expiry"] = (to_seconds(self.maturity) - df["time"]).clip(lower=0).astype(float)
        df = df[cols]
        require_no_nan(df, ["mark_apr_close"])
        self._data = df

    def read(self, with_run: bool = False) -> BorosMarketHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return BorosMarketHistory(time=[], **{c: [] for c in BorosMarketHistory.COLUMNS})
        return BorosMarketHistory(
            time=self._utc_index(),
            **{c: self._data[c].astype(float).values for c in BorosMarketHistory.COLUMNS},
        )


def bar_seconds(time_frame: str) -> int:
    """Seconds per candle for a Boros ``time_frame``."""
    return _TIME_FRAMES[time_frame]
