"""GMX V1 funding-rate loader.

.. warning::
    The legacy ``subgraph.satsuma-prod.com/.../gmx-arbitrum-stats/api`` URL was
    shut down in 2024 along with the original ``api.thegraph.com`` hosted
    service. There is currently **no public replacement** subgraph that
    exposes the V1 ``fundingRates`` entity in the same shape. The loader is
    kept for back-compat: pass your own ``url`` (e.g. a self-hosted graph
    node) to use it again.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import FundingHistory

LEGACY_URL = "https://subgraph.satsuma-prod.com/3b2ced13c8d9/gmx/gmx-arbitrum-stats/api"


class GMXV1FundingLoader(Loader):
    """Funding-rate history for a single GMX V1 token."""

    _BATCH_LIMIT = 1000

    def __init__(
        self,
        token_address: str,
        loader_type: LoaderType = LoaderType.CSV,
        url: str = LEGACY_URL,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        period: str = "daily",
    ) -> None:
        super().__init__(loader_type)
        self.token_address: str = token_address.lower()
        self._url: str = url
        self._period: str = period
        self.start_time: Optional[datetime] = to_utc(start_time)
        self.end_time: Optional[datetime] = to_utc(end_time)
        self._http = HttpClient()

    def _cache_key(self) -> str:
        s = to_seconds(self.start_time) if self.start_time is not None else "open"
        e = to_seconds(self.end_time) if self.end_time is not None else "now"
        return f"{self.token_address}-{self._period}-{s}-{e}"

    def _query(self, last_ts: Optional[int]) -> str:
        clauses = [f'period: "{self._period}"', f'token: "{self.token_address}"']
        if last_ts is not None:
            clauses.append(f"timestamp_lt: {last_ts}")
        if self.start_time is not None:
            clauses.append(f"timestamp_gte: {to_seconds(self.start_time)}")
        where = ", ".join(clauses)
        return (
            "{ fundingRates("
            f"first: {self._BATCH_LIMIT}, "
            "orderBy: timestamp, orderDirection: desc, "
            f"where: {{{where}}}, subgraphError: allow"
            ") { token timestamp startFundingRate startTimestamp endFundingRate endTimestamp } }"
        )

    def extract(self) -> None:
        all_rows: List[Dict[str, Any]] = []
        cursor: Optional[int] = (
            to_seconds(self.end_time) if self.end_time is not None else None
        )
        while True:
            payload = self._http.post(self._url, json={"query": self._query(cursor)})
            if "errors" in (payload or {}):
                raise RuntimeError(f"GraphQL errors from GMX V1: {payload['errors']}")
            batch = payload.get("data", {}).get("fundingRates") or []
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < self._BATCH_LIMIT:
                break
            cursor = int(batch[-1]["timestamp"])
        self._data = pd.DataFrame(all_rows)

    def transform(self) -> None:
        cols = ["time", "rate"]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        df = self._data
        df["rate"] = (df["endFundingRate"].astype(float) - df["startFundingRate"].astype(float)) / 1e6
        df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df = df[cols].sort_values("time").drop_duplicates("time").reset_index(drop=True)
        self._data = df

    def load(self) -> None:
        self._load(self._cache_key())

    def read(self, with_run: bool = False) -> FundingHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return FundingHistory(rates=[], time=[])
        # Inverted sign matches the historical convention: longs pay shorts =>
        # negative rate for long-side accounting.
        return FundingHistory(
            rates=(-1) * self._data["rate"].astype(float).values,
            time=pd.to_datetime(self._data["time"], utc=True).values,
        )
