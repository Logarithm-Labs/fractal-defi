"""Morpho Blue loaders: market static info and hourly market history.

API: ``https://api.morpho.org/graphql`` (no key; 750 req/min). Since
2026-05 markets are addressed by ``marketById(marketId, chainId)``;
``marketByUniqueKey`` and ``Market.id`` were removed. History comes
from ``historicalState { field(options: {startTimestamp, endTimestamp,
interval}) { x y } }`` with points **descending** and APYs as
fractions; the newest point is the live one (not interval-aligned) and
is dropped.

Rates: the API reports effective annual ``borrowApy = exp(r·YEAR) − 1``.
The loader converts to the **per-bar** exponent the entities apply
(``ln(1 + apy) · Δt / YEAR``, ``compounding="continuous"``) or Aave's
linear ``apy / bars_per_year`` (``"linear"``), and keeps the annual
values in the optional ``borrow_apy`` / ``supply_apy`` columns.

Oracle price: the API has no oracle-price history. For PT collateral
on a linear-discount feed, :func:`read_linear_discount` reads
``baseDiscountPerYear`` once and the observation builder computes
``1 − d · t`` per bar.
"""
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from fractal.loaders._dt import SECONDS_PER_YEAR, require_no_nan, to_seconds, to_utc, utcnow
from fractal.loaders._http import HttpClient
from fractal.loaders._rpc import decode_words, eth_call
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import LendingHistory, PriceHistory

MORPHO_GRAPHQL_URL = "https://api.morpho.org/graphql"
_REQUEST_SLEEP_SECONDS = 0.1
_MARKET_ID_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")
_INTERVAL_SECONDS = {"HOUR": 3600, "DAY": 86_400}
_BASE_DISCOUNT_SELECTOR = "0x61d5a1f7"  # baseDiscountPerYear()

_INFO_QUERY = """
query MarketInfo($id: String!, $cid: Int!) {
  marketById(marketId: $id, chainId: $cid) {
    marketId lltv irmAddress oracleAddress
    oracle { address type }
    loanAsset { address symbol decimals }
    collateralAsset { address symbol decimals }
    state { price fee borrowApy supplyApy utilization rateAtTarget }
  }
}
"""

_HISTORY_QUERY = """
query MarketHistory($id: String!, $cid: Int!, $o: TimeseriesOptions) {
  marketById(marketId: $id, chainId: $cid) {
    historicalState {
      borrowApy(options: $o) { x y }
      supplyApy(options: $o) { x y }
      utilization(options: $o) { x y }
      rateAtTarget(options: $o) { x y }
    }
    collateralAsset { historicalPriceUsd(options: $o) { x y } }
  }
}
"""

__all__ = [
    "MORPHO_GRAPHQL_URL",
    "MorphoLoaderException",
    "MorphoMarketInfo",
    "get_market_info",
    "read_linear_discount",
    "MorphoMarketLoader",
]


class MorphoLoaderException(RuntimeError):
    """GraphQL errors or malformed Morpho API data."""


def _validate_market_id(value: str) -> str:
    if not isinstance(value, str) or not _MARKET_ID_RE.match(value):
        raise ValueError(f"market_id must match ^0x[a-fA-F0-9]{{64}}$, got {value!r}")
    return value.lower()


def _graphql(http: HttpClient, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    payload = http.post(MORPHO_GRAPHQL_URL, json={"query": query, "variables": variables})
    if not isinstance(payload, dict):
        raise MorphoLoaderException(f"Morpho GraphQL: expected a JSON object, got {type(payload).__name__}")
    if payload.get("errors"):
        raise MorphoLoaderException(f"Morpho GraphQL errors: {payload['errors']}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise MorphoLoaderException("Morpho GraphQL: missing 'data'")
    return data


# --------------------------------------------------------- market info
@dataclass(frozen=True)
class MorphoMarketInfo:
    """Static market config from ``marketById``; ``lltv`` as a fraction."""
    chain_id: int
    market_id: str
    lltv: float
    loan_asset: str
    loan_symbol: str
    loan_decimals: int
    collateral_asset: str
    collateral_symbol: str
    collateral_decimals: int
    oracle_address: Optional[str]
    oracle_type: Optional[str]
    irm_address: Optional[str]
    fee: float
    borrow_apy: Optional[float] = None
    supply_apy: Optional[float] = None
    utilization: Optional[float] = None
    oracle_price: Optional[float] = None

    @property
    def price_scale(self) -> float:
        """Divisor turning ``state.price`` into loan-per-collateral units."""
        return 10.0 ** (36 + self.loan_decimals - self.collateral_decimals)


def _wad(value: Any) -> float:
    number = float(value)
    return number / 1e18 if number > 1.0 else number


def get_market_info(market_id: str, chain_id: int, http: Optional[HttpClient] = None) -> MorphoMarketInfo:
    """Fetch LLTV, assets, oracle and the current state of one market."""
    client = http or HttpClient()
    data = _graphql(client, _INFO_QUERY, {"id": _validate_market_id(market_id), "cid": int(chain_id)})
    market = data.get("marketById")
    if not market:
        raise MorphoLoaderException(f"market {market_id} not found on chain {chain_id}")
    loan, coll, state = market["loanAsset"], market["collateralAsset"], market.get("state") or {}
    oracle = market.get("oracle") or {}
    loan_dec, coll_dec = int(loan["decimals"]), int(coll["decimals"])
    raw_price = state.get("price")
    oracle_price = float(raw_price) / 10.0 ** (36 + loan_dec - coll_dec) if raw_price is not None else None
    return MorphoMarketInfo(
        chain_id=int(chain_id), market_id=market["marketId"].lower(), lltv=_wad(market["lltv"]),
        loan_asset=loan["address"].lower(), loan_symbol=loan["symbol"], loan_decimals=loan_dec,
        collateral_asset=coll["address"].lower(), collateral_symbol=coll["symbol"], collateral_decimals=coll_dec,
        oracle_address=(oracle.get("address") or market.get("oracleAddress") or None),
        oracle_type=oracle.get("type"), irm_address=market.get("irmAddress"),
        fee=_wad(state.get("fee") or 0.0),
        borrow_apy=state.get("borrowApy"), supply_apy=state.get("supplyApy"),
        utilization=state.get("utilization"), oracle_price=oracle_price,
    )


def read_linear_discount(
    rpc_url: str,
    feed_address: str,
    *,
    block: str = "latest",
    http: Optional[HttpClient] = None,
) -> float:
    """``baseDiscountPerYear()`` of a ``PendleSparkLinearDiscountOracle`` feed, as a fraction."""
    result = eth_call(rpc_url, feed_address, _BASE_DISCOUNT_SELECTOR, block=block, http=http)
    return decode_words(result, 1)[0] / 1e18


# ------------------------------------------------------- market history
def _series_to_df(points: Optional[List[Dict[str, Any]]], col: str) -> pd.DataFrame:
    if not points:
        return pd.DataFrame({"x": pd.Series(dtype="int64"), col: pd.Series(dtype=float)})
    df = pd.DataFrame(points)
    return pd.DataFrame({"x": df["x"].astype("int64"), col: df["y"].astype(float)})


class MorphoMarketLoader(Loader):
    """Hourly (or daily) market history → :class:`LendingHistory` with per-bar rates.

    Args:
        market_id: 32-byte market id (``0x…``).
        chain_id: EVM chain id.
        start_time / end_time: inclusive UTC window.
        resolution: bar length in hours the rates are converted to.
        interval: API interval, ``"HOUR"`` or ``"DAY"``.
        compounding: ``"continuous"`` (per-bar exponent ``ln(1+apy)·Δt/YEAR``)
            or ``"linear"`` (``apy · Δt / YEAR``, Aave parity).
        http: injectable client for offline tests.
    """

    def __init__(
        self,
        market_id: str,
        chain_id: int,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        *,
        resolution: int = 1,
        interval: str = "HOUR",
        compounding: str = "continuous",
        loader_type: LoaderType = LoaderType.CSV,
        http: Optional[HttpClient] = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if interval not in _INTERVAL_SECONDS:
            raise ValueError(f"interval must be one of {sorted(_INTERVAL_SECONDS)}, got {interval!r}")
        if compounding not in ("continuous", "linear"):
            raise ValueError(f"compounding must be 'continuous' or 'linear', got {compounding!r}")
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0 hours, got {resolution}")
        self.market_id = _validate_market_id(market_id)
        self.chain_id = int(chain_id)
        self.start_time = to_utc(start_time)
        self.end_time = to_utc(end_time) if end_time is not None else utcnow()
        if self.end_time < self.start_time:
            raise ValueError(f"end_time {self.end_time} precedes start_time {self.start_time}")
        self.resolution = int(resolution)
        self.interval = interval
        self.compounding = compounding
        self._http = http or HttpClient()

    def _cache_key(self) -> str:
        return (
            f"{self.chain_id}-{self.market_id}-{self.interval}-{to_seconds(self.start_time)}-"
            f"{to_seconds(self.end_time)}-{self.resolution}-{self.compounding}"
        )

    def extract(self) -> None:
        data = _graphql(self._http, _HISTORY_QUERY, {
            "id": self.market_id, "cid": self.chain_id,
            "o": {"startTimestamp": to_seconds(self.start_time), "endTimestamp": to_seconds(self.end_time),
                  "interval": self.interval},
        })
        market = data.get("marketById")
        if not market:
            raise MorphoLoaderException(f"market {self.market_id} not found on chain {self.chain_id}")
        hist = market.get("historicalState") or {}
        frames = [
            _series_to_df(hist.get("borrowApy"), "borrow_apy"),
            _series_to_df(hist.get("supplyApy"), "supply_apy"),
            _series_to_df(hist.get("utilization"), "utilization"),
            _series_to_df(hist.get("rateAtTarget"), "rate_at_target"),
            _series_to_df((market.get("collateralAsset") or {}).get("historicalPriceUsd"), "collateral_price_usd"),
        ]
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="x", how="outer")
        self._data = merged.sort_values("x").reset_index(drop=True)

    def transform(self) -> None:
        cols = ["time", "lending_rate", "borrowing_rate", "utilization", "borrow_apy", "supply_apy",
                "rate_at_target", "collateral_price_usd"]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        df = self._data.rename(columns={"x": "time"}).copy()
        df["time"] = df["time"].astype("int64")
        step = _INTERVAL_SECONDS[self.interval]
        aligned = df["time"] % step == 0
        if (~aligned).any():
            df = df[aligned]  # drop the live, non-aligned newest point
        df = df[(df["time"] >= to_seconds(self.start_time)) & (df["time"] <= to_seconds(self.end_time))]
        df = df.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
        gaps = df["time"].diff().dropna() != step
        if gaps.any():
            raise ValueError(
                f"Morpho history for {self.market_id} has {int(gaps.sum())} gap(s) in its {self.interval} grid; "
                f"narrow the window instead of filling missing bars"
            )
        bar_seconds = self.resolution * 3600
        if self.compounding == "continuous":
            df["borrowing_rate"] = df["borrow_apy"].apply(lambda apy: math.log1p(apy) * bar_seconds / SECONDS_PER_YEAR)
        else:
            df["borrowing_rate"] = df["borrow_apy"] * bar_seconds / SECONDS_PER_YEAR
        # Morpho Blue collateral never earns the supply rate (only loan-token
        # suppliers do), and ``LendingHistory.lending_rates`` is credited to
        # collateral by the entities: emit 0 there and keep the market's
        # supplier APY in the optional ``supply_apy`` column.
        df["lending_rate"] = 0.0
        for col in cols:
            if col not in df.columns:
                df[col] = float("nan")
        df = df[cols]
        require_no_nan(df, ["lending_rate", "borrowing_rate", "utilization"])
        self._data = df

    def read(self, with_run: bool = False) -> LendingHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return LendingHistory(lending_rates=[], borrowing_rates=[], time=[])
        return LendingHistory(
            lending_rates=self._data["lending_rate"].astype(float).values,
            borrowing_rates=self._data["borrowing_rate"].astype(float).values,
            time=self._utc_index(),
            utilization=self._data["utilization"].astype(float).values,
            borrow_apy=self._data["borrow_apy"].astype(float).values,
            supply_apy=self._data["supply_apy"].astype(float).values,
            rate_at_target=self._data["rate_at_target"].astype(float).values,
        )

    def read_collateral_price(self) -> PriceHistory:
        """The API's USD price of the collateral asset over the same window (from the current ``_data``)."""
        if self._data is None or self._data.empty:
            return PriceHistory(prices=[], time=[])
        series = self._data[["time", "collateral_price_usd"]].dropna()
        return PriceHistory(
            prices=series["collateral_price_usd"].astype(float).values,
            time=pd.to_datetime(series["time"].astype("int64"), unit="s", utc=True),
        )
