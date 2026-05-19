"""Pendle public-REST loader.

Endpoint:
    GET https://api-v2.pendle.finance/core/v1/{chain_id}/markets/{market}/historical-data
        ?time_frame=hour&from=<ISO>&to=<ISO>

Returns parallel arrays (``timestamp``, ``impliedApy``, ``tvl``, plus
optional ``baseApy``, ``underlyingApy``, ``maxApy``). PT mid-price is
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
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import PendleMarketHistory
from fractal.loaders.thegraph.base_graph_loader import validate_evm_address

PENDLE_REST_BASE = "https://api-v2.pendle.finance/core/v1"
_SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0

_SUPPORTED_CHAIN_IDS: Set[int] = {1, 42161, 8453}

_OUTPUT_COLUMNS = (
    "pt_price",
    "implied_yield",
    "seconds_to_expiry",
    "pool_liquidity",
    "base_apy",
    "underlying_apy",
    "max_apy",
)


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
    runs the pipeline and writes a CSV cache; ``read(with_run=False)``
    rehydrates the cached CSV back into a :class:`PendleMarketHistory`
    so both paths return the same typed contract.
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
        self.market_address: str = validate_evm_address(
            market_address, field="market_address"
        )
        self.expiry_timestamp: int = int(expiry_timestamp)
        self.start_time: datetime = to_utc(start_time)
        self.end_time: datetime = to_utc(end_time)
        if self.end_time < self.start_time:
            raise ValueError(
                f"end_time {self.end_time} precedes start_time {self.start_time}"
            )
        chain_id_int = int(chain_id)
        if chain_id_int not in _SUPPORTED_CHAIN_IDS:
            raise ValueError(
                f"chain_id {chain_id_int} not supported; "
                f"expected one of {sorted(_SUPPORTED_CHAIN_IDS)}"
            )
        self.chain_id: int = chain_id_int
        self._api_key: Optional[str] = api_key
        self._raw: Dict[str, Any] = {}
        self._http: HttpClient = HttpClient()

    def _cache_key(self) -> str:
        # Use full epoch seconds (not just dates) so two windows that share
        # a date but differ by hours don't collide on disk.
        return (
            f"{self.chain_id}-{self.market_address}-"
            f"{to_seconds(self.start_time)}-{to_seconds(self.end_time)}"
        )

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

    def extract(self) -> None:
        payload = self._http.get(self._url(), params=self._params())
        self._raw = payload if isinstance(payload, dict) else {}

    def transform(self) -> None:
        payload = self._raw
        timestamps = payload.get("timestamp") or []
        implied = payload.get("impliedApy") or []
        tvl = payload.get("tvl") or []
        base = payload.get("baseApy") or []
        underlying = payload.get("underlyingApy") or []
        maxapy = payload.get("maxApy") or []
        if not timestamps:
            self._data = _empty_history()
            return

        n = min(len(timestamps), len(implied), len(tvl))
        out_ts: List[int] = []
        out_pt: List[float] = []
        out_iy: List[float] = []
        out_se: List[float] = []
        out_pl: List[float] = []
        out_b: List[float] = []
        out_u: List[float] = []
        out_m: List[float] = []
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
            out_b.append(_safe_float(base, i))
            out_u.append(_safe_float(underlying, i))
            out_m.append(_safe_float(maxapy, i))

        self._data = PendleMarketHistory(
            pt_prices=out_pt,
            implied_yields=out_iy,
            seconds_to_expiry=out_se,
            pool_liquidity=out_pl,
            time=out_ts,
            base_apy=out_b,
            underlying_apy=out_u,
            max_apy=out_m,
        )

    def read(self, with_run: bool = False) -> PendleMarketHistory:
        """Return the hourly ``PendleMarketHistory``.

        Cache-hit path rehydrates the CSV into the typed history so both
        ``with_run=True`` and ``with_run=False`` paths return the same
        type and the same UTC ``DatetimeIndex``.
        """
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
            self._data = _rebuild_from_cache(self._data)
        return self._data


# ---------------------------------------------------------------- helpers


def _safe_float(seq: Any, idx: int) -> float:
    """``float(seq[idx])`` or ``NaN`` if missing / not coercible."""
    try:
        return float(seq[idx])
    except (TypeError, ValueError, IndexError, KeyError):
        return float("nan")


def _empty_history() -> PendleMarketHistory:
    return PendleMarketHistory(
        pt_prices=[],
        implied_yields=[],
        seconds_to_expiry=[],
        pool_liquidity=[],
        time=[],
        base_apy=[],
        underlying_apy=[],
        max_apy=[],
    )


def _rebuild_from_cache(data: Any) -> PendleMarketHistory:
    """Coerce a cached ``pd.DataFrame`` (or ``PendleMarketHistory``) back to typed form.

    The CSV cache loses the typed-class identity and the index name; we
    restore both here so cache reads behave identically to fresh runs.
    """
    if isinstance(data, PendleMarketHistory):
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
    return PendleMarketHistory(
        pt_prices=df["pt_price"].to_numpy(),
        implied_yields=df["implied_yield"].to_numpy(),
        seconds_to_expiry=df["seconds_to_expiry"].to_numpy(),
        pool_liquidity=df["pool_liquidity"].to_numpy(),
        time=idx,
        base_apy=df["base_apy"].to_numpy(),
        underlying_apy=df["underlying_apy"].to_numpy(),
        max_apy=df["max_apy"].to_numpy(),
    )
