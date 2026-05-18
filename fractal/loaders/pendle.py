"""Pendle public-REST loader.

Endpoint:
    GET https://api-v2.pendle.finance/core/v1/{chain_id}/markets/{market}/historical-data
        ?time_frame=hour&from=<ISO>&to=<ISO>

Returns parallel arrays (timestamp, impliedApy, tvl). PT mid-price is
not exposed by the API, so we compute it from the implied yield and
the time-to-expiry using the linear convention that Pendle's own
oracle uses (``pt_price = 1 - implied_yield * tau``, clamped to
``[0, 1]``). The linear approximation matches what Morpho's PT oracle
reads on-chain — keeps pricing consistent across the stack.

The endpoint caps returned rows (~60-day window at hourly granularity).
For longer histories the loader would have to paginate via successive
``from`` slices; we leave that as a follow-up when a concrete
multi-cycle use case arrives upstream.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from fractal.loaders._dt import to_utc
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import PendleMarketHistory

PENDLE_REST_BASE = "https://api-v2.pendle.finance/core/v1"
_SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
_REQUEST_TIMEOUT_S = 30.0


def _compute_pt_price_linear(implied_yield: float, seconds_to_expiry: float) -> float:
    """``pt_price = 1 - implied_yield * tau``, clamped to ``[0, 1]``."""
    if seconds_to_expiry <= 0:
        return 1.0
    tau = seconds_to_expiry / _SECONDS_PER_YEAR
    price = 1.0 - implied_yield * tau
    if price < 0.0:
        return 0.0
    if price > 1.0:
        return 1.0
    return price


class PendleMarketLoader(Loader):
    """Hourly Pendle PT market history for a single (chain, market) pair.

    Pipeline: ``extract`` GETs the historical-data endpoint and stashes
    the JSON in ``self._raw``; ``transform`` parses the parallel-array
    payload into a :class:`PendleMarketHistory`; ``read(with_run=True)``
    runs the pipeline and writes a CSV cache.
    """

    def __init__(
        self,
        market_address: str,
        expiry_timestamp: int,
        start_time: datetime,
        end_time: datetime,
        chain_id: int = 1,
        loader_type: LoaderType = LoaderType.CSV,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        self.market_address: str = market_address.lower()
        self.expiry_timestamp: int = int(expiry_timestamp)
        self.start_time: datetime = to_utc(start_time)
        self.end_time: datetime = to_utc(end_time)
        self.chain_id: int = int(chain_id)
        self._api_key: Optional[str] = api_key
        self._raw: Dict[str, Any] = {}

    def _cache_key(self) -> str:
        s = self.start_time.strftime("%Y%m%d")
        e = self.end_time.strftime("%Y%m%d")
        return f"{self.chain_id}-{self.market_address}-{s}-{e}"

    def _url(self) -> str:
        return (
            f"{PENDLE_REST_BASE}/{self.chain_id}/markets/"
            f"{self.market_address}/historical-data"
        )

    def _params(self) -> Dict[str, str]:
        return {
            "time_frame": "hour",
            "from": self.start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": self.end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def _headers(self) -> Dict[str, str]:
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
        payload = self._raw
        timestamps = payload.get("timestamp") or []
        implied = payload.get("impliedApy") or []
        tvl = payload.get("tvl") or []
        if not timestamps:
            self._data = PendleMarketHistory(
                pt_prices=[],
                implied_yields=[],
                seconds_to_expiry=[],
                pool_liquidity=[],
                time=[],
            )
            return

        n = min(len(timestamps), len(implied), len(tvl))
        out_ts: List[int] = []
        out_pt: List[float] = []
        out_iy: List[float] = []
        out_se: List[float] = []
        out_pl: List[float] = []
        for i in range(n):
            try:
                epoch = int(timestamps[i])
                implied_y = float(implied[i])
                liquidity = float(tvl[i])
            except (TypeError, ValueError):
                continue
            seconds_to_expiry = float(max(self.expiry_timestamp - epoch, 0))
            pt = _compute_pt_price_linear(implied_y, seconds_to_expiry)
            out_ts.append(epoch)
            out_pt.append(pt)
            out_iy.append(implied_y)
            out_se.append(seconds_to_expiry)
            out_pl.append(liquidity)
        self._data = PendleMarketHistory(
            pt_prices=out_pt,
            implied_yields=out_iy,
            seconds_to_expiry=out_se,
            pool_liquidity=out_pl,
            time=out_ts,
        )

    def read(self, with_run: bool = False) -> PendleMarketHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        return self._data
