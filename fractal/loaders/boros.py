"""Pendle Boros market loader.

Boros is Pendle's funding-rate tokenisation layer launched August 2025.
A Boros market represents a forward on the *average funding rate* of a
specific perpetual futures market (e.g. Hyperliquid ETH-USDC) up to a
fixed maturity. Going long a Boros market = receiving funding payments
between now and maturity; going short = paying.

API
---
Public, keyless:

* List markets: ``GET https://api.boros.finance/core/v1/markets``
  Returns a list of objects with ``marketId``, ``imData.symbol``,
  ``imData.maturity``, ``metadata.fundingRateSymbol``,
  ``data.markApr`` (current annualised mark rate), and so on.

* Historical bars: ``GET https://api.boros.finance/core/v1/markets/chart``
  ``?marketId=<N>&timeFrame=<5m|1h|1d|1w>``. The endpoint **ignores
  ``from``/``to`` parameters and always returns the most recent ~500
  bars** (verified empirically May 2026). For longer history we would
  need to scrape on-chain via ``eth_getLogs`` on the AMM contract —
  out of scope for the current research project.

Output
------
:class:`BorosMarketHistory` indexed by UTC datetime with columns:

* ``mark_apr``         — close mark rate (annualised, decimal).
* ``observed_funding`` — close observed funding rate (annualised).
* ``mark_apr_7d_ma``   — 7-day moving average of observed funding.
* ``mark_apr_30d_ma``  — 30-day moving average of observed funding.

The ``mark_apr`` column is what a long-Boros holder receives in funding
per unit of time. It is the analogue of the per-hour funding-rate
loaders (Hyperliquid, Binance) and can be swapped in directly in any
strategy that consumes funding-rate observations.

Recent-only coverage is a known endpoint limitation, not a loader bug.
For long-history backtests, calibrate against a per-hour funding feed
(e.g. :class:`HyperliquidFundingRatesLoader`) on the overlap window.
"""
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd

from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import BorosMarketHistory

BOROS_API_BASE: str = "https://api.boros.finance/core/v1"

_ALLOWED_TIME_FRAMES: FrozenSet[str] = frozenset({"5m", "1h", "1d", "1w"})

_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "mark_apr",
    "observed_funding",
    "mark_apr_7d_ma",
    "mark_apr_30d_ma",
)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class BorosMarketLoader(Loader):
    """Recent funding-rate bars for one Boros market.

    Pipeline: ``extract`` issues a single REST GET to the Boros chart
    endpoint via the shared :class:`HttpClient` (retry + backoff) and
    stashes the JSON in ``self._raw``; ``transform`` parses the bar
    list into a typed :class:`BorosMarketHistory`; ``read`` returns the
    same type on both cache-hit and cache-miss paths.
    """

    def __init__(
        self,
        market_id: int,
        start_time: datetime,
        end_time: datetime,
        *,
        time_frame: str = "1h",
        api_key: Optional[str] = None,
        loader_type: LoaderType = LoaderType.CSV,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if time_frame not in _ALLOWED_TIME_FRAMES:
            raise ValueError(
                f"time_frame must be one of {sorted(_ALLOWED_TIME_FRAMES)}, "
                f"got {time_frame!r}"
            )
        self.market_id: int = int(market_id)
        self.start_time: datetime = _to_utc(start_time)
        self.end_time: datetime = _to_utc(end_time)
        if self.end_time < self.start_time:
            raise ValueError(
                f"end_time {self.end_time} precedes start_time {self.start_time}"
            )
        self.time_frame: str = time_frame
        self._api_key: Optional[str] = api_key
        self._raw: Dict[str, Any] = {}
        self._http: HttpClient = HttpClient()

    def _cache_key(self) -> str:
        s = int(self.start_time.timestamp())
        e = int(self.end_time.timestamp())
        return f"market{self.market_id}-{self.time_frame}-{s}-{e}"

    def _url(self) -> str:
        return f"{BOROS_API_BASE}/markets/chart"

    def _params(self) -> Dict[str, Any]:
        return {"marketId": self.market_id, "timeFrame": self.time_frame}

    def extract(self) -> None:
        payload = self._http.get(self._url(), params=self._params())
        self._raw = payload if isinstance(payload, dict) else {}

    def transform(self) -> None:
        if not self._raw:
            self._data = _empty_history()
            return
        rows = _parse_bars(self._raw)
        rows = [r for r in rows if self.start_time <= r["dt"] <= self.end_time]
        if not rows:
            self._data = _empty_history()
            return
        self._data = BorosMarketHistory(
            mark_apr=[r["mark_apr"] for r in rows],
            observed_funding=[r["observed_funding"] for r in rows],
            mark_apr_7d_ma=[r["mark_apr_7d_ma"] for r in rows],
            mark_apr_30d_ma=[r["mark_apr_30d_ma"] for r in rows],
            time=[int(r["dt"].timestamp()) for r in rows],
        )

    def read(self, with_run: bool = False) -> BorosMarketHistory:
        """Return the typed ``BorosMarketHistory`` on both cache paths."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
            self._data = _rebuild_from_cache(self._data)
        return self._data


# ---------------------------------------------------------------- helpers


def _empty_history() -> BorosMarketHistory:
    return BorosMarketHistory(
        mark_apr=[],
        observed_funding=[],
        mark_apr_7d_ma=[],
        mark_apr_30d_ma=[],
        time=[],
    )


def _parse_bars(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse Boros chart payload — list of bars under ``results``."""
    results = payload.get("results") or []
    if not isinstance(results, list) or not results:
        return []
    parsed: List[Dict[str, Any]] = []
    seen_ts: Set[int] = set()
    for row in results:
        try:
            epoch = int(row["ts"])
            mark_apr = float(row.get("c", row.get("mr", 0.0)))
            observed = float(row.get("u", row.get("ofr", 0.0)))
            ma7 = float(row.get("b7dmafr", 0.0))
            ma30 = float(row.get("b30dmafr", 0.0))
        except (TypeError, ValueError, KeyError):
            continue
        if epoch in seen_ts:
            continue
        seen_ts.add(epoch)
        parsed.append(
            {
                "dt": datetime.fromtimestamp(epoch, tz=timezone.utc),
                "mark_apr": mark_apr,
                "observed_funding": observed,
                "mark_apr_7d_ma": ma7,
                "mark_apr_30d_ma": ma30,
            }
        )
    parsed.sort(key=lambda r: r["dt"])
    return parsed


def _rebuild_from_cache(data: Any) -> BorosMarketHistory:
    """Coerce a cached ``pd.DataFrame`` (or ``BorosMarketHistory``) back to typed form."""
    if isinstance(data, BorosMarketHistory):
        return data
    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        return _empty_history()
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
    idx = pd.to_datetime(df.index, utc=True)
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    idx.name = "time"
    for col in _OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return BorosMarketHistory(
        mark_apr=df["mark_apr"].to_numpy(),
        observed_funding=df["observed_funding"].to_numpy(),
        mark_apr_7d_ma=df["mark_apr_7d_ma"].to_numpy(),
        mark_apr_30d_ma=df["mark_apr_30d_ma"].to_numpy(),
        time=idx,
    )


__all__ = ["BorosMarketLoader", "BOROS_API_BASE"]
