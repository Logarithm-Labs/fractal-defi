"""Pendle public-REST OHLCV loader (PT / YT / SY token prices).

Endpoint:
    GET https://api-v2.pendle.finance/core/v4/{chain_id}/prices/{address}/ohlcv
        ?time_frame=<1m|5m|15m|1h|4h|1d|1w>&from=<unix_seconds>&to=<unix_seconds>

The endpoint returns OHLCV bars for any Pendle-tracked asset address —
PT, YT, SY or the underlying — denominated in USD. Bars are produced
from Pendle's own oracle so they are consistent with the AMM prices
exposed by the historical-data endpoint that backs
:class:`PendleMarketLoader`.

Use this loader when you want a candlestick view of the PT / YT mark
through time (drawdown analysis, vol estimation, plotting). For
implied-yield / liquidity timeseries keyed off the *market* contract
rather than the *token* address, use :class:`PendleMarketLoader`.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from fractal.loaders._dt import to_seconds, to_utc
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import KlinesHistory
from fractal.loaders.thegraph.base_graph_loader import validate_evm_address

PENDLE_REST_BASE: str = "https://api-v2.pendle.finance/core"

_SUPPORTED_CHAIN_IDS: Set[int] = {1, 42161, 8453}

_ALLOWED_TIME_FRAMES: Set[str] = {"1m", "5m", "15m", "1h", "4h", "1d", "1w"}


class PendleOHLCVLoader(Loader):
    """OHLCV bars for a Pendle-tracked token (PT / YT / SY / underlying).

    Pipeline: ``extract`` GETs the OHLCV endpoint via the shared
    :class:`HttpClient`; ``transform`` parses the bar list into a typed
    :class:`KlinesHistory`; ``read`` returns the same type on both
    cache-hit and cache-miss paths.

    Args:
        token_address: 20-byte EVM address of the PT / YT / SY token.
        chain_id: ``1`` (Ethereum), ``42161`` (Arbitrum), or ``8453``
            (Base).
        start_time: Inclusive start of the historical window, UTC.
        end_time: Inclusive end of the historical window, UTC.
        time_frame: Bar resolution; one of ``{1m, 5m, 15m, 1h, 4h, 1d, 1w}``.
        api_key: Reserved for future authenticated endpoints; the
            public Pendle REST endpoints do NOT require a key.
        loader_type: Cache backend; default CSV.
    """

    def __init__(
        self,
        token_address: str,
        start_time: datetime,
        end_time: datetime,
        chain_id: int = 1,
        *,
        time_frame: str = "1h",
        api_key: Optional[str] = None,
        loader_type: LoaderType = LoaderType.CSV,
    ) -> None:
        super().__init__(loader_type=loader_type)
        self.token_address: str = validate_evm_address(
            token_address, field="token_address"
        )
        chain_id_int = int(chain_id)
        if chain_id_int not in _SUPPORTED_CHAIN_IDS:
            raise ValueError(
                f"chain_id {chain_id_int} not supported; "
                f"expected one of {sorted(_SUPPORTED_CHAIN_IDS)}"
            )
        if time_frame not in _ALLOWED_TIME_FRAMES:
            raise ValueError(
                f"time_frame must be one of {sorted(_ALLOWED_TIME_FRAMES)}, "
                f"got {time_frame!r}"
            )
        self.chain_id: int = chain_id_int
        self.start_time: datetime = to_utc(start_time)
        self.end_time: datetime = to_utc(end_time)
        if self.end_time < self.start_time:
            raise ValueError(
                f"end_time {self.end_time} precedes start_time {self.start_time}"
            )
        self.time_frame: str = time_frame
        self._api_key: Optional[str] = api_key
        self._raw: Any = None
        self._http: HttpClient = HttpClient()

    def _cache_key(self) -> str:
        return (
            f"{self.chain_id}-{self.token_address}-{self.time_frame}-"
            f"{to_seconds(self.start_time)}-{to_seconds(self.end_time)}"
        )

    def _url(self) -> str:
        return (
            f"{PENDLE_REST_BASE}/v4/{self.chain_id}/prices/"
            f"{self.token_address}/ohlcv"
        )

    def _params(self) -> Dict[str, Any]:
        return {
            "time_frame": self.time_frame,
            "from": to_seconds(self.start_time),
            "to": to_seconds(self.end_time),
        }

    def extract(self) -> None:
        payload = self._http.get(self._url(), params=self._params())
        self._raw = payload

    def transform(self) -> None:
        bars = _extract_bars(self._raw)
        if not bars:
            self._data = _empty_klines()
            return
        time: List[int] = []
        opens: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []
        volumes: List[float] = []
        for bar in bars:
            try:
                t = int(bar.get("time") or bar.get("t") or bar.get("timestamp"))
                o = float(bar.get("open", bar.get("o", 0.0)))
                h = float(bar.get("high", bar.get("h", 0.0)))
                low = float(bar.get("low", bar.get("l", 0.0)))
                c = float(bar.get("close", bar.get("c", 0.0)))
                v = float(bar.get("volume", bar.get("v", 0.0)))
            except (TypeError, ValueError, KeyError):
                continue
            time.append(t)
            opens.append(o)
            highs.append(h)
            lows.append(low)
            closes.append(c)
            volumes.append(v)
        self._data = KlinesHistory(
            time=time,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            volume=volumes,
        )

    def read(self, with_run: bool = False) -> KlinesHistory:
        """Return the OHLCV ``KlinesHistory`` on both cache paths."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
            self._data = _rebuild_from_cache(self._data)
        return self._data


# ---------------------------------------------------------------- helpers


def _extract_bars(raw: Any) -> List[Dict[str, Any]]:
    """Locate the bar list within a Pendle OHLCV response payload.

    Pendle's OHLCV endpoint has evolved across minor versions; this
    helper tolerates the two shapes observed in production (top-level
    ``list`` and ``{"results": [...]}`` / ``{"data": [...]}`` wrappers).
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict):
        for key in ("results", "data", "ohlcv", "bars"):
            v = raw.get(key)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
    return []


def _empty_klines() -> KlinesHistory:
    return KlinesHistory(
        time=[], open=[], high=[], low=[], close=[], volume=[]
    )


def _rebuild_from_cache(data: Any) -> KlinesHistory:
    """Coerce a cached DataFrame (or ``KlinesHistory``) back to typed form."""
    import pandas as pd

    if isinstance(data, KlinesHistory):
        return data
    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        return _empty_klines()
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
    idx = pd.to_datetime(df.index, utc=True)
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    idx.name = "time"
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = 0.0
    return KlinesHistory(
        time=idx,
        open=df["open"].to_numpy(),
        high=df["high"].to_numpy(),
        low=df["low"].to_numpy(),
        close=df["close"].to_numpy(),
        volume=df["volume"].to_numpy(),
    )
