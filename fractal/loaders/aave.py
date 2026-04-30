"""Aave V3 lending/borrowing-rate loaders.

Aave migrated their public REST endpoint (``aave-api-v2.aave.com/data/rates-history``)
to a GraphQL service at ``https://api.v3.aave.com/graphql`` in 2025. The old
URL now serves an HTML landing page, so we go through GraphQL.

The GraphQL API exposes APY history through fixed-window enums
(``LAST_DAY`` / ``LAST_WEEK`` / ``LAST_MONTH`` / ``LAST_SIX_MONTHS`` / ``LAST_YEAR``).
The loader picks the smallest window that covers the user's
``start_time``…``end_time`` range and filters client-side.

Rates returned by the API are **annualized**. We convert to per-period
rates (``annual / (365 * 24 / resolution)``) so they match the legacy
``LendingHistory`` semantics. Borrowing rates are stored with a negative
sign (you pay), supply rates with a positive sign (you receive).
"""
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc, utcnow
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import LendingHistory

DEFAULT_URL = "https://api.v3.aave.com/graphql"

# Aave V3 market (PoolAddressesProvider) addresses by chain.
ARBITRUM_V3_MARKET = "0x794a61358D6845594F94dc1DB02A252b5b4814aD"
ETHEREUM_V3_MARKET = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"

# Time-window enum → upper-bound timedelta. We pick the smallest window that
# still covers the user's start_time.
_WINDOWS = [
    ("LAST_DAY", timedelta(days=1)),
    ("LAST_WEEK", timedelta(days=7)),
    ("LAST_MONTH", timedelta(days=31)),
    ("LAST_SIX_MONTHS", timedelta(days=183)),
    ("LAST_YEAR", timedelta(days=366)),
]


def _window_for(start_time: Optional[datetime], end_time: Optional[datetime]) -> str:
    """Pick the smallest TimeWindow enum covering ``[start_time, end_time]``."""
    end = to_utc(end_time) or utcnow()
    start = to_utc(start_time)
    if start is None:
        return "LAST_MONTH"
    span = end - start
    for name, td in _WINDOWS:
        if span <= td:
            return name
    return "LAST_YEAR"


class AaveV3RatesLoader(Loader):
    """Fetch supply+borrow APY history for a single Aave V3 market+asset."""

    _QUERY = """
    query Rates($req: BorrowAPYHistoryRequest!, $sup: SupplyAPYHistoryRequest!) {
      borrowAPYHistory(request: $req) { date avgRate { value } }
      supplyAPYHistory(request: $sup) { date avgRate { value } }
    }
    """

    def __init__(
        self,
        asset_address: str,
        chain_id: int,
        market_address: str,
        loader_type: LoaderType = LoaderType.CSV,
        url: str = DEFAULT_URL,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        resolution: int = 1,
    ) -> None:
        super().__init__(loader_type)
        self.asset_address: str = asset_address.lower()
        self.market_address: str = market_address
        self.chain_id: int = int(chain_id)
        self._url: str = url
        self.start_time: Optional[datetime] = to_utc(start_time)
        self.end_time: Optional[datetime] = to_utc(end_time)
        self._resolution: int = int(resolution)
        self._http = HttpClient()

    def _cache_key(self) -> str:
        s = to_seconds(self.start_time) if self.start_time is not None else "open"
        e = to_seconds(self.end_time) if self.end_time is not None else "now"
        return f"{self.chain_id}-{self.asset_address}-{s}-{e}-{self._resolution}"

    def _request(self) -> List[Dict[str, Any]]:
        window = _window_for(self.start_time, self.end_time)
        variables = {
            "req": {
                "market": self.market_address,
                "underlyingToken": self.asset_address,
                "window": window,
                "chainId": self.chain_id,
            },
            "sup": {
                "market": self.market_address,
                "underlyingToken": self.asset_address,
                "window": window,
                "chainId": self.chain_id,
            },
        }
        payload = self._http.post(self._url, json={"query": self._QUERY, "variables": variables})
        if "errors" in (payload or {}):
            raise RuntimeError(f"Aave GraphQL errors: {payload['errors']}")
        return payload.get("data", {})

    def extract(self) -> None:
        data = self._request()
        borrow = pd.DataFrame(data.get("borrowAPYHistory") or [])
        supply = pd.DataFrame(data.get("supplyAPYHistory") or [])
        if borrow.empty and supply.empty:
            self._data = pd.DataFrame(columns=["date", "lending_rate", "borrowing_rate"])
            return

        def _flatten(df: pd.DataFrame, col: str) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame(columns=["date", col])
            df = df.copy()
            df[col] = df["avgRate"].apply(lambda r: float(r["value"]) if r else float("nan"))
            df["date"] = pd.to_datetime(df["date"], utc=True)
            return df[["date", col]]

        borrow_f = _flatten(borrow, "borrowing_rate")
        supply_f = _flatten(supply, "lending_rate")
        self._data = (
            supply_f.merge(borrow_f, on="date", how="outer")
            .sort_values("date")
            .reset_index(drop=True)
        )

    def transform(self) -> None:
        cols = ["date", "lending_rate", "borrowing_rate"]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        df = self._data
        # Convert annual APY to per-period rate; the legacy convention divides
        # the annual rate by (year_hours / resolution_hours).
        scale = (365 * 24) / max(self._resolution, 1)
        df["lending_rate"] = df["lending_rate"].astype(float) / scale
        df["borrowing_rate"] = -df["borrowing_rate"].astype(float) / scale
        # Filter to user-requested window
        if self.start_time is not None:
            df = df[df["date"] >= self.start_time]
        if self.end_time is not None:
            df = df[df["date"] <= self.end_time]
        self._data = df[cols].reset_index(drop=True)

    def load(self) -> None:
        self._load(self._cache_key())

    def read(self, with_run: bool = False) -> LendingHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return LendingHistory(lending_rates=[], borrowing_rates=[], time=[])
        return LendingHistory(
            lending_rates=self._data["lending_rate"].astype(float).fillna(0.0).values,
            borrowing_rates=self._data["borrowing_rate"].astype(float).fillna(0.0).values,
            time=pd.to_datetime(self._data["date"], utc=True).values,
        )


class AaveV3ArbitrumLoader(AaveV3RatesLoader):
    """Aave V3 on Arbitrum One (chain id 42161)."""

    def __init__(
        self,
        asset_address: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        resolution: int = 1,
    ) -> None:
        super().__init__(
            asset_address=asset_address,
            chain_id=42161,
            market_address=ARBITRUM_V3_MARKET,
            loader_type=loader_type,
            start_time=start_time,
            end_time=end_time,
            resolution=resolution,
        )


class AaveV3EthereumLoader(AaveV3RatesLoader):
    """Aave V3 on Ethereum mainnet (chain id 1)."""

    def __init__(
        self,
        asset_address: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        resolution: int = 1,
    ) -> None:
        super().__init__(
            asset_address=asset_address,
            chain_id=1,
            market_address=ETHEREUM_V3_MARKET,
            loader_type=loader_type,
            start_time=start_time,
            end_time=end_time,
            resolution=resolution,
        )


class AaveV2EthereumLoader(AaveV3EthereumLoader):
    """Aave V2 Ethereum loader.

    .. deprecated::
        The Aave V2 ``rates-history`` REST endpoint was sunset in 2025. This
        class is kept as a thin alias of :class:`AaveV3EthereumLoader` so
        existing call sites keep working, but it now returns V3 mainnet data.
        Migrate to :class:`AaveV3EthereumLoader` directly.
    """

    def __init__(
        self,
        asset_address: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        resolution: int = 1,
    ) -> None:
        warnings.warn(
            "AaveV2EthereumLoader is deprecated; the Aave V2 rates API was "
            "sunset in 2025. This class now returns Aave V3 Ethereum data.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            asset_address=asset_address,
            loader_type=loader_type,
            start_time=start_time,
            end_time=end_time,
            resolution=resolution,
        )
