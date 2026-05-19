"""Morpho Blue market loader — hourly borrow rate and utilization.

Endpoint
--------
Morpho exposes market data through a keyless GraphQL endpoint at
``https://blue-api.morpho.org/graphql``. The relevant root field is
``marketByUniqueKey(uniqueKey, chainId)``; it returns a ``Market`` whose
``historicalState`` selection produces timeseries with the requested
``interval`` (``HOUR`` / ``DAY`` / …) and ``[startTimestamp, endTimestamp]``
window. Each timeseries entry is a ``FloatDataPoint`` of shape
``{x: <unix_seconds>, y: <decimal>}``.

The fields we read are already **annualised decimals**:

* ``borrowApy`` — annualised borrow APY paid by debtors. Equivalent to
  the ``borrowing_rate`` column of :class:`LendingHistory`.
* ``supplyApy`` — annualised supply APY earned by lenders. Equivalent
  to the ``lending_rate`` column of :class:`LendingHistory`.
* ``utilization`` — fraction of supplied debt asset currently borrowed,
  in ``[0, 1]``. Optional column of :class:`LendingHistory` populated
  by this loader (Morpho exposes it; other lending sources may not).

Chains
------
Morpho is deployed on Ethereum mainnet, Arbitrum and Base. This loader
accepts a chain name and maps it to the API's ``chainId`` parameter.

Output contract
---------------
:meth:`MorphoMarketLoader.read` returns a :class:`LendingHistory` with
the three columns ``lending_rate``, ``borrowing_rate``, ``utilization``
indexed by a UTC-aware ``DatetimeIndex``. Empty windows return a
same-shape :class:`LendingHistory` with zero rows.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import LendingHistory


MORPHO_GRAPHQL_URL: str = "https://blue-api.morpho.org/graphql"

# Chain-name → numeric chainId used by the Morpho GraphQL API.
_CHAIN_ID_BY_NAME: Dict[str, int] = {
    "ethereum": 1,
    "arbitrum": 42161,
    "base": 8453,
}

# Default interval. Hourly matches the rest of the backtest's granularity.
_INTERVAL: str = "HOUR"

# Morpho market ids are 32-byte hex strings ("0x" + 64 hex chars).
_MARKET_ID_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")

# Output column order is part of the public contract for downstream joins.
_OUTPUT_COLUMNS: Tuple[str, ...] = ("lending_rate", "borrowing_rate", "utilization")

_HISTORICAL_QUERY: str = """
query MarketHistory($id: String!, $cid: Int!, $opts: TimeseriesOptions) {
  marketByUniqueKey(uniqueKey: $id, chainId: $cid) {
    historicalState {
      borrowApy(options: $opts) { x y }
      supplyApy(options: $opts) { x y }
      utilization(options: $opts) { x y }
    }
  }
}
"""


def _chain_id(chain: str) -> int:
    """Map a chain name to the Morpho API's integer chainId."""
    key = chain.lower()
    if key not in _CHAIN_ID_BY_NAME:
        raise ValueError(
            f"unsupported chain {chain!r}; expected one of {sorted(_CHAIN_ID_BY_NAME)}"
        )
    return _CHAIN_ID_BY_NAME[key]


def _validate_market_id(value: str) -> str:
    """Reject anything that is not a 32-byte hex ``0x``-prefixed string."""
    if not isinstance(value, str) or not _MARKET_ID_RE.match(value):
        raise ValueError(
            f"market_id must match ^0x[a-fA-F0-9]{{64}}$, got {value!r}"
        )
    return value.lower()


def _series_to_df(points: Optional[List[Dict[str, Any]]], col: str) -> pd.DataFrame:
    """Coerce a ``[{x, y}, …]`` timeseries to a 2-col DataFrame keyed by ``x``."""
    if not points:
        return pd.DataFrame(columns=["x", col])
    df = pd.DataFrame(points)
    df["x"] = df["x"].astype("int64")
    df[col] = df["y"].astype(float)
    return df[["x", col]]


class MorphoMarketLoader(Loader):
    """Hourly historical state for a single Morpho Blue isolated market.

    The loader uses Morpho's GraphQL endpoint (no API key needed). On
    cache miss it issues one POST request covering
    ``[start_time, end_time]`` at hourly resolution; on cache hit it
    deserialises the on-disk CSV. Cache keys include chain, market id,
    and both timestamps as Unix seconds so different windows do not
    collide.

    Args:
        market_id: 32-byte Morpho market identifier as a ``0x``-hex
            string (``0x`` + 64 hex chars). Validated at construction.
        chain: Lower-case chain name (``"ethereum"``, ``"arbitrum"``,
            ``"base"``); resolved to ``chainId`` internally.
        start_time: Inclusive start of the historical window, UTC.
        end_time: Inclusive end of the historical window, UTC.
        api_key: Reserved for future authenticated endpoints; the
            public ``blue-api.morpho.org`` endpoint does NOT require a
            key. Pass ``None`` for the public API.
        loader_type: Cache backend; default CSV.
    """

    def __init__(
        self,
        market_id: str,
        chain: str,
        start_time: datetime,
        end_time: datetime,
        *,
        api_key: Optional[str] = None,
        loader_type: LoaderType = LoaderType.CSV,
    ) -> None:
        super().__init__(loader_type=loader_type)
        self.market_id: str = _validate_market_id(market_id)
        self.chain: str = chain.lower()
        self._chain_id: int = _chain_id(chain)
        self.start_time: datetime = to_utc(start_time)  # type: ignore[assignment]
        self.end_time: datetime = to_utc(end_time)  # type: ignore[assignment]
        if self.start_time is None or self.end_time is None:
            raise ValueError("start_time and end_time are required")
        if self.end_time < self.start_time:
            raise ValueError(
                f"end_time {self.end_time} precedes start_time {self.start_time}"
            )
        self._api_key: Optional[str] = api_key
        self._http: HttpClient = HttpClient()

    # ------------------------------------------------------------------
    # Cache identity
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        return (
            f"{self.chain}-{self.market_id}-"
            f"{to_seconds(self.start_time)}-{to_seconds(self.end_time)}"
        )

    # ------------------------------------------------------------------
    # extract / transform
    # ------------------------------------------------------------------

    def _post(self) -> Dict[str, Any]:
        """POST the historical query and unwrap GraphQL ``data``."""
        variables = {
            "id": self.market_id,
            "cid": self._chain_id,
            "opts": {
                "startTimestamp": to_seconds(self.start_time),
                "endTimestamp": to_seconds(self.end_time),
                "interval": _INTERVAL,
            },
        }
        headers: Optional[Dict[str, str]] = None
        if self._api_key:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }
        payload = self._http.post(
            MORPHO_GRAPHQL_URL,
            json={"query": _HISTORICAL_QUERY, "variables": variables},
            headers=headers,
        )
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Morpho GraphQL: expected JSON object, got {type(payload).__name__}"
            )
        if "errors" in payload:
            raise RuntimeError(f"Morpho GraphQL errors: {payload['errors']}")
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            raise RuntimeError("Morpho GraphQL: missing 'data' object")
        return data

    def extract(self) -> None:
        """Fetch the timeseries triple and stash an unmerged staging frame."""
        data = self._post()
        market = data.get("marketByUniqueKey")
        if not market:
            self._data = pd.DataFrame(columns=["x"] + list(_OUTPUT_COLUMNS))
            return
        hist = market.get("historicalState") or {}
        borrow = _series_to_df(hist.get("borrowApy"), "borrowing_rate")
        supply = _series_to_df(hist.get("supplyApy"), "lending_rate")
        util = _series_to_df(hist.get("utilization"), "utilization")
        merged = borrow.merge(util, on="x", how="outer").merge(
            supply, on="x", how="outer"
        )
        self._data = merged.sort_values("x").reset_index(drop=True)

    def transform(self) -> None:
        """Convert the staging frame to a typed :class:`LendingHistory`."""
        if self._data is None or self._data.empty:
            self._data = LendingHistory([], [], [], utilization=[])
            return
        df = self._data.copy()
        epoch = df["x"].astype("int64").to_numpy()
        for c in _OUTPUT_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        df = df[list(_OUTPUT_COLUMNS)].astype(float)
        self._data = LendingHistory(
            lending_rates=df["lending_rate"].to_numpy(),
            borrowing_rates=df["borrowing_rate"].to_numpy(),
            utilization=df["utilization"].to_numpy(),
            time=epoch,
        )

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def read(self, with_run: bool = False) -> LendingHistory:
        """Return the hourly ``LendingHistory``; run pipeline on cache miss."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
            self._data = _rebuild_from_cache(self._data)
        return self._data


# ---------------------------------------------------------------- helpers


def _rebuild_from_cache(data: Any) -> LendingHistory:
    """Coerce a cached ``pd.DataFrame`` (or ``LendingHistory``) back to typed form."""
    if isinstance(data, LendingHistory):
        return data
    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        return LendingHistory([], [], [], utilization=[])
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
    return LendingHistory(
        lending_rates=df["lending_rate"].to_numpy(),
        borrowing_rates=df["borrowing_rate"].to_numpy(),
        utilization=df["utilization"].to_numpy(),
        time=idx,
    )
