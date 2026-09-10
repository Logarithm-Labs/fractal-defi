"""Pendle V2 loaders: market static info, market history, token OHLCV.

API (``https://api-v2.pendle.finance/core``, no key on the free tier —
100 CU/min, 200k CU/week, so requests are paced):

* ``GET /v1/{chainId}/markets/{address}`` — static market info
  (expiry, PT/YT/SY addresses, underlying and accounting asset).
* ``GET /v3/{chainId}/markets/{address}/historical-data?time_frame=hour|day
  &timestamp_start=ISO&timestamp_end=ISO&includeApyBreakdown=true`` —
  rows of ``timestamp, impliedApy, underlyingApy, tvl, ptPrice, syPrice,
  totalPt, totalSy``. **Hourly rows are only served for the last ~60
  days**; daily rows cover the market's whole life.
* ``GET /v4/{chainId}/prices/{address}/ohlcv?time_frame=…`` — USD
  candles for a PT/YT/LP address, ≤ 1440 rows per call.

The PT price used by the entity is derived from ``impliedApy`` with
Pendle's compounded convention (``(1 + apy) ** (-t)``, accounting asset
per PT); the API's USD marks are carried alongside. The AMM parameters
``scalarRoot`` / ``lnFeeRateRoot`` are not served by the API — use
:func:`read_market_state` (one ``eth_call`` to ``readState``) when the
AMM impact model is wanted.
"""
import time as _time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from fractal.loaders._dt import SECONDS_PER_YEAR, require_no_nan, to_seconds, to_utc, utcnow
from fractal.loaders._http import HttpClient
from fractal.loaders._rpc import as_signed, decode_words, eth_call
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import KlinesHistory, PendleMarketHistory
from fractal.loaders.thegraph.base_graph_loader import validate_evm_address

PENDLE_API = "https://api-v2.pendle.finance/core"
PENDLE_ROUTER = "0x888888888889758F76e7103c6CbF23ABbF58F946"  # same on every chain
_REQUEST_SLEEP_SECONDS = 0.65  # 100 CU/min on the free tier
_TIME_FRAMES = {"hour": 3600, "day": 86_400, "week": 7 * 86_400}
_OHLCV_TIME_FRAMES = {"hour": 3600, "day": 86_400, "week": 7 * 86_400}
_OHLCV_ROW_CAP = 1440
_READ_STATE_SELECTOR = "0x1a6e2a2c"  # readState(address)

__all__ = [
    "PENDLE_API",
    "PENDLE_ROUTER",
    "PendleLoaderException",
    "PendleMarketInfo",
    "PendleMarketState",
    "get_market_info",
    "read_market_state",
    "PendleMarketLoader",
    "PendleOHLCVLoader",
]


class PendleLoaderException(RuntimeError):
    """Malformed or missing Pendle API data."""


# ------------------------------------------------------------- helpers
def _rows(payload: Any) -> List[Dict[str, Any]]:
    """Unwrap ``results``/``data`` envelopes; accept a bare list."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "data"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return inner
    return []


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _epoch_seconds(values: Iterable[Any]) -> pd.Series:
    """ISO strings or numeric epochs (seconds or milliseconds) → int64 seconds."""
    series = pd.Series(list(values))
    if pd.api.types.is_numeric_dtype(series):
        numeric = series.astype("int64")
        return (numeric // 1000) if numeric.abs().max() > 10 ** 11 else numeric
    return (pd.to_datetime(series, utc=True).astype("int64") // 10 ** 9).astype("int64")


def _token(value: Any) -> Optional[str]:
    """``"1-0xabc…"`` or ``{"address": …}`` → lower-case address."""
    if isinstance(value, dict):
        value = value.get("address")
    if not isinstance(value, str):
        return None
    return value.split("-")[-1].lower()


# --------------------------------------------------------- market info
@dataclass(frozen=True)
class PendleMarketInfo:
    """Static market configuration from ``GET /v1/{chainId}/markets/{address}``."""
    chain_id: int
    address: str
    expiry: datetime
    pt: Optional[str]
    yt: Optional[str]
    sy: Optional[str]
    underlying_asset: Optional[str]
    accounting_asset: Optional[str]
    name: Optional[str] = None


def get_market_info(chain_id: int, market_address: str, http: Optional[HttpClient] = None) -> PendleMarketInfo:
    """Fetch expiry and token addresses so callers never type an expiry by hand."""
    address = validate_evm_address(market_address, field="market_address")
    client = http or HttpClient()
    payload = client.get(f"{PENDLE_API}/v1/{int(chain_id)}/markets/{address}")
    if not isinstance(payload, dict) or not payload.get("expiry"):
        raise PendleLoaderException(f"market {address} on chain {chain_id}: no expiry in {payload!r}")
    expiry = pd.Timestamp(payload["expiry"], tz="UTC").to_pydatetime()
    return PendleMarketInfo(
        chain_id=int(chain_id), address=address, expiry=expiry,
        pt=_token(payload.get("pt")), yt=_token(payload.get("yt")), sy=_token(payload.get("sy")),
        underlying_asset=_token(payload.get("underlyingAsset")),
        accounting_asset=_token(payload.get("accountingAsset")),
        name=payload.get("name"),
    )


@dataclass(frozen=True)
class PendleMarketState:
    """``PendleMarket.readState(router)`` — the AMM parameters the API does not serve."""
    total_pt: float
    total_sy: float
    scalar_root: float
    expiry: int
    ln_fee_rate_root: float
    reserve_fee_percent: int
    last_ln_implied_rate: float


def read_market_state(
    rpc_url: str,
    market_address: str,
    *,
    router: str = PENDLE_ROUTER,
    block: str = "latest",
    http: Optional[HttpClient] = None,
) -> PendleMarketState:
    """One ``eth_call`` to ``readState(router)``; token amounts are returned raw (18-dec WAD)."""
    market = validate_evm_address(market_address, field="market_address")
    data = _READ_STATE_SELECTOR + router.lower().replace("0x", "").rjust(64, "0")
    words = decode_words(eth_call(rpc_url, market, data, block=block, http=http), 9)
    return PendleMarketState(
        total_pt=as_signed(words[0]) / 1e18,
        total_sy=as_signed(words[1]) / 1e18,
        scalar_root=as_signed(words[4]) / 1e18,
        expiry=words[5],
        ln_fee_rate_root=words[6] / 1e18,
        reserve_fee_percent=words[7],
        last_ln_implied_rate=words[8] / 1e18,
    )


# ------------------------------------------------------ market history
class PendleMarketLoader(Loader):
    """Market history → :class:`PendleMarketHistory`.

    Args:
        market_address: Pendle market (LP) address.
        chain_id: EVM chain id (1, 42161, 8453, …).
        start_time / end_time: inclusive UTC window.
        time_frame: ``"hour"`` or ``"day"``.
        daily_fallback: with ``time_frame="hour"``, fill the part of the
            window older than the API's hourly retention with daily rows
            stretched to hourly (a warning says how many bars).
        expiry: market expiry; fetched via :func:`get_market_info` when
            omitted (offline tests pass it explicitly).
        http: injectable client for offline tests.
    """

    def __init__(
        self,
        market_address: str,
        chain_id: int,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        *,
        time_frame: str = "hour",
        daily_fallback: bool = True,
        expiry: Optional[datetime] = None,
        loader_type: LoaderType = LoaderType.CSV,
        http: Optional[HttpClient] = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if time_frame not in ("hour", "day"):
            raise ValueError(f"time_frame must be 'hour' or 'day', got {time_frame!r}")
        self.market_address = validate_evm_address(market_address, field="market_address")
        self.chain_id = int(chain_id)
        self.start_time = to_utc(start_time)
        self.end_time = to_utc(end_time) if end_time is not None else utcnow()
        if self.end_time < self.start_time:
            raise ValueError(f"end_time {self.end_time} precedes start_time {self.start_time}")
        self.time_frame = time_frame
        self.daily_fallback = daily_fallback
        self._expiry: Optional[datetime] = to_utc(expiry) if expiry is not None else None
        self._http = http or HttpClient()
        self._rows: List[Dict[str, Any]] = []

    @property
    def expiry(self) -> datetime:
        if self._expiry is None:
            self._expiry = get_market_info(self.chain_id, self.market_address, http=self._http).expiry
        return self._expiry

    def _cache_key(self) -> str:
        tail = "dfb" if self.daily_fallback else "nofb"
        return (
            f"{self.chain_id}-{self.market_address}-{self.time_frame}-"
            f"{to_seconds(self.start_time)}-{to_seconds(self.end_time)}-{to_seconds(self.expiry)}-{tail}"
        )

    # ---------------------------------------------------------- fetch
    def _fetch(self, time_frame: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Forward-paging fetch of one time frame over ``[start, end]``."""
        url = f"{PENDLE_API}/v3/{self.chain_id}/markets/{self.market_address}/historical-data"
        step_seconds = _TIME_FRAMES[time_frame]
        rows: List[Dict[str, Any]] = []
        cursor = start
        while cursor <= end:
            payload = self._http.get(url, params={
                "time_frame": time_frame, "timestamp_start": _iso(cursor), "timestamp_end": _iso(end),
                "includeApyBreakdown": "true",
            })
            batch = _rows(payload)
            if not batch:
                break
            rows.extend(batch)
            last = _epoch_seconds([r.get("timestamp") for r in batch]).max()
            next_cursor = datetime.fromtimestamp(int(last), tz=cursor.tzinfo) + timedelta(seconds=step_seconds)
            if next_cursor <= cursor or next_cursor > end:
                break
            cursor = next_cursor
            _time.sleep(_REQUEST_SLEEP_SECONDS)
        return rows

    def extract(self) -> None:
        rows = self._fetch(self.time_frame, self.start_time, self.end_time)
        if self.time_frame == "hour" and self.daily_fallback and rows:
            first = datetime.fromtimestamp(int(_epoch_seconds([r["timestamp"] for r in rows]).min()),
                                           tz=self.start_time.tzinfo)
            if first > self.start_time + timedelta(hours=2):
                daily = self._fetch("day", self.start_time, first - timedelta(seconds=1))
                if daily:
                    warnings.warn(
                        f"PendleMarketLoader: hourly history for {self.market_address} starts at {first}; "
                        f"{len(daily)} daily bars before it are stretched to hourly"
                    )
                    for row in daily:
                        row["_stretched"] = True
                    rows = daily + rows
        self._rows = rows

    # ------------------------------------------------------ transform
    def transform(self) -> None:
        if not self._rows:
            self._data = pd.DataFrame(columns=["time", *PendleMarketHistory.COLUMNS])
            return
        raw = pd.DataFrame(self._rows)
        df = pd.DataFrame({"time": _epoch_seconds(raw["timestamp"])})
        df["implied_apy"] = pd.to_numeric(raw.get("impliedApy"), errors="coerce")
        df["pt_price_usd"] = pd.to_numeric(raw.get("ptPrice"), errors="coerce")
        df["sy_price_usd"] = pd.to_numeric(raw.get("syPrice"), errors="coerce")
        df["underlying_apy"] = pd.to_numeric(raw.get("underlyingApy"), errors="coerce")
        df["tvl"] = pd.to_numeric(raw.get("tvl"), errors="coerce")
        df["total_pt"] = pd.to_numeric(raw.get("totalPt"), errors="coerce")
        df["total_sy"] = pd.to_numeric(raw.get("totalSy"), errors="coerce")
        stretched = raw["_stretched"].eq(True) if "_stretched" in raw.columns else None
        if stretched is not None and stretched.any():
            df = self._stretch_daily_prefix(df, stretched)
        df = df.sort_values("time").drop_duplicates("time", keep="last")
        df = df[(df["time"] >= to_seconds(self.start_time)) & (df["time"] <= to_seconds(self.end_time))]
        df["seconds_to_expiry"] = (to_seconds(self.expiry) - df["time"]).clip(lower=0).astype(float)
        years = df["seconds_to_expiry"] / SECONDS_PER_YEAR
        df["pt_price_asset"] = (1.0 + df["implied_apy"]) ** (-years)
        df["pt_price_sy"] = df["pt_price_usd"] / df["sy_price_usd"]
        df = df.reset_index(drop=True)[["time", *PendleMarketHistory.COLUMNS]]
        require_no_nan(df, ["implied_apy", "pt_price_usd", "sy_price_usd", "tvl", "seconds_to_expiry"])
        self._data = df

    @staticmethod
    def _stretch_daily_prefix(df: pd.DataFrame, stretched: pd.Series) -> pd.DataFrame:
        daily = df[stretched.values].sort_values("time")
        hourly = df[~stretched.values]
        grid = pd.DataFrame({"time": range(int(daily["time"].min()), int(daily["time"].max()) + 86_400, 3600)})
        filled = pd.merge_asof(grid, daily, on="time", direction="backward")
        if not hourly.empty:
            filled = filled[filled["time"] < hourly["time"].min()]
        return pd.concat([filled, hourly], ignore_index=True)

    # ----------------------------------------------------------- read
    def read(self, with_run: bool = False) -> PendleMarketHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return PendleMarketHistory(time=[], **{c: [] for c in PendleMarketHistory.COLUMNS})
        return PendleMarketHistory(
            time=self._utc_index(),
            **{c: self._data[c].astype(float).values for c in PendleMarketHistory.COLUMNS},
        )


# -------------------------------------------------------------- OHLCV
class PendleOHLCVLoader(Loader):
    """USD candles of a PT / YT / LP address → :class:`KlinesHistory`."""

    def __init__(
        self,
        token_address: str,
        chain_id: int,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        *,
        time_frame: str = "hour",
        loader_type: LoaderType = LoaderType.CSV,
        http: Optional[HttpClient] = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if time_frame not in _OHLCV_TIME_FRAMES:
            raise ValueError(f"time_frame must be one of {sorted(_OHLCV_TIME_FRAMES)}, got {time_frame!r}")
        self.token_address = validate_evm_address(token_address, field="token_address")
        self.chain_id = int(chain_id)
        self.start_time = to_utc(start_time)
        self.end_time = to_utc(end_time) if end_time is not None else utcnow()
        if self.end_time < self.start_time:
            raise ValueError(f"end_time {self.end_time} precedes start_time {self.start_time}")
        self.time_frame = time_frame
        self._http = http or HttpClient()
        self._rows: List[Dict[str, Any]] = []

    def _cache_key(self) -> str:
        return (
            f"{self.chain_id}-{self.token_address}-{self.time_frame}-"
            f"{to_seconds(self.start_time)}-{to_seconds(self.end_time)}"
        )

    @staticmethod
    def _parse(payload: Any) -> List[Dict[str, Any]]:
        """Rows as dicts, or the CSV-in-JSON form ``"time,open,high,low,close,volume\\n…"``."""
        rows = _rows(payload)
        if rows:
            return rows
        text = payload.get("results") if isinstance(payload, dict) else payload
        if isinstance(text, str) and "\n" in text:
            frame = pd.read_csv(pd.io.common.StringIO(text))
            return frame.to_dict("records")
        return []

    def extract(self) -> None:
        url = f"{PENDLE_API}/v4/{self.chain_id}/prices/{self.token_address}/ohlcv"
        step_seconds = _OHLCV_TIME_FRAMES[self.time_frame]
        rows: List[Dict[str, Any]] = []
        cursor = self.start_time
        while cursor <= self.end_time:
            payload = self._http.get(url, params={
                "time_frame": self.time_frame, "timestamp_start": _iso(cursor), "timestamp_end": _iso(self.end_time),
            })
            batch = self._parse(payload)
            if not batch:
                break
            rows.extend(batch)
            last = _epoch_seconds([r.get("time", r.get("timestamp")) for r in batch]).max()
            next_cursor = datetime.fromtimestamp(int(last), tz=cursor.tzinfo) + timedelta(seconds=step_seconds)
            if len(batch) < _OHLCV_ROW_CAP or next_cursor <= cursor or next_cursor > self.end_time:
                break
            cursor = next_cursor
            _time.sleep(_REQUEST_SLEEP_SECONDS)
        self._rows = rows

    def transform(self) -> None:
        cols = ["time", "open", "high", "low", "close", "volume"]
        if not self._rows:
            self._data = pd.DataFrame(columns=cols)
            return
        raw = pd.DataFrame(self._rows)
        time_col = "time" if "time" in raw.columns else "timestamp"
        df = pd.DataFrame({"time": _epoch_seconds(raw[time_col])})
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(raw.get(col), errors="coerce")
        df["volume"] = pd.to_numeric(raw.get("volume"), errors="coerce").fillna(0.0) if "volume" in raw else 0.0
        df = df.sort_values("time").drop_duplicates("time", keep="last")
        df = df[(df["time"] >= to_seconds(self.start_time)) & (df["time"] <= to_seconds(self.end_time))]
        df = df.reset_index(drop=True)[cols]
        require_no_nan(df, ["open", "high", "low", "close"])
        self._data = df

    def read(self, with_run: bool = False) -> KlinesHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return KlinesHistory(time=[], open=[], high=[], low=[], close=[], volume=[])
        return KlinesHistory(
            time=self._utc_index(),
            open=self._data["open"].astype(float).values, high=self._data["high"].astype(float).values,
            low=self._data["low"].astype(float).values, close=self._data["close"].astype(float).values,
            volume=self._data["volume"].astype(float).values,
        )
