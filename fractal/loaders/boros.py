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
DataFrame indexed by UTC datetime, columns:

* ``mark_apr``         — close mark rate (annualised, decimal).
* ``observed_funding`` — close observed funding rate (annualised).
* ``mark_apr_7d_ma``   — 7-day moving average of observed funding.
* ``mark_apr_30d_ma``  — 30-day moving average of observed funding.

The ``mark_apr`` column is what a long-Boros holder receives in funding
per unit of time. It is the analogue of
``funding_rate_annualised`` in the ``FundingHedgeLoader`` (Hyperliquid
proxy) and can be swapped in directly in any strategy that consumes
funding-rate observations.

Recent-only coverage is a known limitation. We use the Hyperliquid
funding-rate loader as a long-history proxy and validate empirically
on the overlap window that the two series track each other closely.
See ``scripts/validate_boros_proxy.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import BorosMarketHistory

BOROS_API_BASE: str = "https://api.boros.finance/core/v1"

_REQUEST_TIMEOUT_S: float = 30.0
_OUTPUT_COLUMNS: tuple[str, ...] = (
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
    endpoint and stashes the JSON in ``self._raw``; ``transform`` parses
    the bar list into a UTC-indexed DataFrame.
    """

    def __init__(
        self,
        market_id: int,
        start_time: datetime,
        end_time: datetime,
        *,
        time_frame: str = "1h",
        api_key: str | None = None,
        loader_type: LoaderType = LoaderType.CSV,
    ) -> None:
        super().__init__(loader_type=loader_type)
        self.market_id: int = int(market_id)
        self.start_time: datetime = _to_utc(start_time)
        self.end_time: datetime = _to_utc(end_time)
        self.time_frame: str = time_frame
        self._api_key: str | None = api_key
        self._raw: dict[str, Any] = {}

    def _cache_key(self) -> str:
        s = self.start_time.strftime("%Y%m%d")
        e = self.end_time.strftime("%Y%m%d")
        return f"market{self.market_id}-{self.time_frame}-{s}-{e}"

    def _url(self) -> str:
        return f"{BOROS_API_BASE}/markets/chart"

    def _params(self) -> dict[str, Any]:
        return {"marketId": self.market_id, "timeFrame": self.time_frame}

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def extract(self) -> None:
        response = requests.get(
            self._url(),
            params=self._params(),
            headers=self._headers(),
            timeout=_REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json() if callable(getattr(response, "json", None)) else {}
        self._raw = payload if isinstance(payload, dict) else {}

    def transform(self) -> None:
        if not self._raw:
            self._data = BorosMarketHistory([], [], [], [], [])
            return
        rows = _parse_bars(self._raw)
        rows = [r for r in rows if self.start_time <= r["dt"] <= self.end_time]
        if not rows:
            self._data = BorosMarketHistory([], [], [], [], [])
            return
        self._data = BorosMarketHistory(
            mark_apr=[r["mark_apr"] for r in rows],
            observed_funding=[r["observed_funding"] for r in rows],
            mark_apr_7d_ma=[r["mark_apr_7d_ma"] for r in rows],
            mark_apr_30d_ma=[r["mark_apr_30d_ma"] for r in rows],
            time=[int(r["dt"].timestamp()) for r in rows],
        )

    def read(self, with_run: bool = False) -> BorosMarketHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        return self._data


# ---------------------------------------------------------------- helpers


def _parse_bars(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse Boros chart payload — list of bars under ``results``."""
    results = payload.get("results") or []
    if not isinstance(results, list) or not results:
        return []
    parsed: list[dict[str, Any]] = []
    seen_ts: set[int] = set()
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


__all__ = ["BorosMarketLoader", "BOROS_API_BASE"]
