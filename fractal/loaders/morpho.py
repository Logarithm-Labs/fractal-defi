"""Morpho Blue loaders: market static info and hourly market history.

API: ``https://api.morpho.org/graphql`` (no key; 750 req/min). Since
2026-05 markets are addressed by ``marketById(marketId, chainId)``;
``marketByUniqueKey`` and ``Market.id`` were removed. History comes
from ``historicalState { field(options: {startTimestamp, endTimestamp,
interval}) { x y } }`` with points **descending** and APYs as
fractions; the newest point is the live one (not interval-aligned) and
is dropped. ``max_window_days`` optionally splits a long window into consecutive
requests whose payloads are concatenated and de-duplicated by timestamp;
by default the whole window is one request.

Rates: the API reports effective annual ``borrowApy = exp(r·YEAR) − 1``.
The loader converts to the **per-bar** exponent the entities apply
(``ln(1 + apy) · Δt / YEAR``, ``compounding="continuous"``) or Aave's
linear ``apy / bars_per_year`` (``"linear"``), and keeps the annual
values in the optional ``borrow_apy`` / ``supply_apy`` columns.

Oracle price: the API has no oracle-price history. For PT collateral
on a linear-discount feed, :func:`read_linear_discount` reads
``baseDiscountPerYear`` once and the observation builder computes
``1 − d · t`` per bar.

Sizes: ``historicalState { supplyAssetsUsd, borrowAssetsUsd,
liquidityAssetsUsd }`` come as USD notional and ``liquidityAssetsUsd`` is
the withdrawable amount the exit gate is measured against. With
``include_market_totals=True``, raw supply/borrow assets are converted by
the loan token's decimals and shares by ``decimals + 6`` (Morpho Blue's
virtual-share offset), so ``supply_assets / supply_shares`` is the supply
share price (about 1 at market creation). BigInts are parsed as exact
integers from JSON numbers or decimal/hex strings; a null point is NaN.
``loanAsset.historicalPriceUsd`` is left-joined onto the bars and read
with :meth:`MorphoMarketLoader.read_loan_price`. ``netSupplyApy`` (supply
APY including the market's reward campaigns) is carried as an optional
column; the reward rate is ``net_supply_apy − supply_apy``. ``fee`` is **not** exposed by the API's
``MarketHistory`` type (verified live 2026-10-12: schema validation
error), so it is not a column here — read the current value from
:func:`get_market_info` and pass it to the entity. ``rateAtTarget`` is
a per-second WAD BigInt; the loader converts it to the annual APY its
column documents.
"""
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from fractal.core.entities.models.morpho_math import WAD, borrow_apy_from_per_second_rate
from fractal.loaders._dt import SECONDS_PER_YEAR, require_no_nan, to_seconds, to_utc, utcnow
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.rpc import decode_words, eth_call, function_selector
from fractal.loaders.structs import LendingHistory, PriceHistory

MORPHO_GRAPHQL_URL = "https://api.morpho.org/graphql"
_REQUEST_SLEEP_SECONDS = 0.1
_MARKET_ID_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")
_INTERVAL_SECONDS = {"HOUR": 3600, "DAY": 86_400}
# Bumped whenever the emitted column set changes, so a cached CSV written by
# an older loader cannot be read back as if it had today's columns.
_CACHE_SCHEMA = "v4marketstate"
_BASE_DISCOUNT_SELECTOR = function_selector("baseDiscountPerYear()")

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
      supplyAssetsUsd(options: $o) { x y }
      borrowAssetsUsd(options: $o) { x y }
      liquidityAssetsUsd(options: $o) { x y }
      netSupplyApy(options: $o) { x y }%s
    }
    loanAsset { decimals historicalPriceUsd(options: $o) { x y } }
    collateralAsset { historicalPriceUsd(options: $o) { x y } }
  }
}
"""
# Raw BigInt market totals, requested only with ``include_market_totals=True``.
_MARKET_TOTALS_FIELDS = ("supplyAssets", "supplyShares", "borrowAssets", "borrowShares")
_MARKET_TOTALS_QUERY = "".join(f"\n      {field}(options: $o) {{ x y }}" for field in _MARKET_TOTALS_FIELDS)
# Morpho Blue mints ``VIRTUAL_SHARES = 1e6`` shares per asset unit at creation.
_VIRTUAL_SHARES_DECIMALS = 6

__all__ = [
    "MORPHO_GRAPHQL_URL",
    "MorphoLoaderException",
    "MorphoMarketInfo",
    "MorphoMarketLoader",
    "get_market_info",
    "graphql_query",
    "read_linear_discount",
]


class MorphoLoaderException(RuntimeError):
    """GraphQL errors or malformed Morpho API data."""


def _validate_market_id(value: str) -> str:
    if not isinstance(value, str) or not _MARKET_ID_RE.match(value):
        raise ValueError(f"market_id must match ^0x[a-fA-F0-9]{{64}}$, got {value!r}")
    return value.lower()


def graphql_query(http: HttpClient, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """POST one GraphQL query to the Morpho API and return its ``data`` object, raising on API errors."""
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
    oracle_address: str | None
    oracle_type: str | None
    irm_address: str | None
    fee: float
    borrow_apy: float | None = None
    supply_apy: float | None = None
    utilization: float | None = None
    oracle_price: float | None = None

    @property
    def price_scale(self) -> float:
        """Divisor turning ``state.price`` into loan-per-collateral units."""
        return 10.0 ** (36 + self.loan_decimals - self.collateral_decimals)


def _wad(value: Any) -> float:
    number = float(value)
    return number / WAD if number > 1.0 else number


def get_market_info(market_id: str, chain_id: int, http: HttpClient | None = None) -> MorphoMarketInfo:
    """Fetch LLTV, assets, oracle and the current state of one market."""
    client = http or HttpClient()
    data = graphql_query(client, _INFO_QUERY, {"id": _validate_market_id(market_id), "cid": int(chain_id)})
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
    http: HttpClient | None = None,
) -> float:
    """``baseDiscountPerYear()`` of a ``PendleSparkLinearDiscountOracle`` feed, as a fraction."""
    result = eth_call(rpc_url, feed_address, _BASE_DISCOUNT_SELECTOR, block=block, http=http)
    return decode_words(result, 1)[0] / WAD


# ------------------------------------------------------- market history
def _series_to_df(points: list[dict[str, Any]] | None, col: str) -> pd.DataFrame:
    if not points:
        return pd.DataFrame({"x": pd.Series(dtype="int64"), col: pd.Series(dtype=float)})
    df = pd.DataFrame(points)
    return pd.DataFrame({"x": df["x"].astype("int64"), col: df["y"].astype(float)})


def _scaled_bigint(value: Any, decimals: int, col: str) -> float:
    """``value / 10**decimals`` from a GraphQL BigInt (JSON number, decimal/hex string); ``None`` → NaN."""
    if value is None:
        return float("nan")
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise MorphoLoaderException(f"{col}: expected a BigInt, got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        try:
            value = int(text, 16) if text.lower().startswith(("0x", "-0x")) else int(text)
        except ValueError:
            try:
                value = float(text)  # scientific notation such as "1.2e+21"
            except ValueError as exc:
                raise MorphoLoaderException(f"{col}: expected a BigInt, got {text!r}") from exc
    # int / int is correctly rounded in Python, so exact raw integers lose no
    # precision beyond the final float.
    return value / 10 ** int(decimals)


def _scaled_bigint_series_to_df(
    points: list[dict[str, Any]] | None,
    col: str,
    decimals: int,
) -> pd.DataFrame:
    points = [point for point in points or [] if point.get("x") is not None]
    if not points:
        return pd.DataFrame({"x": pd.Series(dtype="int64"), col: pd.Series(dtype=float)})
    return pd.DataFrame({
        "x": pd.Series((int(point["x"]) for point in points), dtype="int64"),
        col: pd.Series((_scaled_bigint(point.get("y"), decimals, col) for point in points), dtype=float),
    })


def _merge_points(series: list[list[dict[str, Any]] | None]) -> list[dict[str, Any]]:
    """Concatenate chunked ``{x, y}`` series; a timestamp seen twice (a shared boundary) keeps its last value."""
    by_x: dict[Any, dict[str, Any]] = {}
    for points in series:
        for point in points or []:
            by_x[point.get("x")] = point
    return list(by_x.values())


def _merge_market_payloads(markets: list[dict[str, Any]]) -> dict[str, Any]:
    """One ``marketById`` payload from the payloads of consecutive request windows."""
    hist_keys = dict.fromkeys(key for market in markets for key in (market.get("historicalState") or {}))
    merged: dict[str, Any] = {"historicalState": {
        key: _merge_points([(market.get("historicalState") or {}).get(key) for market in markets])
        for key in hist_keys
    }}
    for asset in ("loanAsset", "collateralAsset"):
        parts = [market.get(asset) or {} for market in markets]
        combined = {key: value for part in parts for key, value in part.items() if value is not None}
        if "historicalPriceUsd" in combined:
            combined["historicalPriceUsd"] = _merge_points([part.get("historicalPriceUsd") for part in parts])
        merged[asset] = combined
    return merged


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
        include_market_totals: include token-unit supply/borrow assets and
            normalized supply/borrow shares. Defaults to ``False`` so the
            historical output columns are unchanged.
        max_window_days: split ``[start_time, end_time]`` into consecutive
            request windows of at most this many days and concatenate them
            (points on a shared boundary are de-duplicated by timestamp).
            ``None`` (default) sends one request for the whole window. The
            output does not depend on it, so it is not part of the cache key.
        http: injectable client for offline tests.
    """

    def __init__(
        self,
        market_id: str,
        chain_id: int,
        start_time: datetime,
        end_time: datetime | None = None,
        *,
        resolution: int = 1,
        interval: str = "HOUR",
        compounding: str = "continuous",
        include_market_totals: bool = False,
        max_window_days: float | None = None,
        loader_type: LoaderType = LoaderType.CSV,
        http: HttpClient | None = None,
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
        self.include_market_totals = bool(include_market_totals)
        if max_window_days is not None and max_window_days * 86_400 < _INTERVAL_SECONDS[interval]:
            raise ValueError(f"max_window_days must cover at least one {interval} bar, got {max_window_days}")
        self.max_window_days = max_window_days
        self._http = http or HttpClient()

    def _cache_key(self) -> str:
        return (
            f"{self.chain_id}-{self.market_id}-{self.interval}-{to_seconds(self.start_time)}-"
            f"{to_seconds(self.end_time)}-{self.resolution}-{self.compounding}-"
            f"{int(self.include_market_totals)}-{_CACHE_SCHEMA}"
        )

    def _request_windows(self) -> list[tuple[int, int]]:
        """Inclusive ``(start, end)`` request windows; consecutive windows share their boundary second."""
        start, end = to_seconds(self.start_time), to_seconds(self.end_time)
        if self.max_window_days is None:
            return [(start, end)]
        span = int(self.max_window_days * 86_400)
        windows = []
        lo = start
        while True:
            hi = min(lo + span, end)
            windows.append((lo, hi))
            if hi >= end:
                return windows
            lo = hi

    def _fetch_market(self) -> dict[str, Any]:
        query = _HISTORY_QUERY % (_MARKET_TOTALS_QUERY if self.include_market_totals else "")
        payloads = []
        for lo, hi in self._request_windows():
            data = graphql_query(self._http, query, {
                "id": self.market_id, "cid": self.chain_id,
                "o": {"startTimestamp": lo, "endTimestamp": hi, "interval": self.interval},
            })
            market = data.get("marketById")
            if not market:
                raise MorphoLoaderException(f"market {self.market_id} not found on chain {self.chain_id}")
            payloads.append(market)
        return payloads[0] if len(payloads) == 1 else _merge_market_payloads(payloads)

    def extract(self) -> None:
        market = self._fetch_market()
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
        # Sizes are left-joined onto the rate spine: a size series that is not
        # backfilled must not add rows (or NaNs) to the rate grid.
        optional_frames = [
            _series_to_df(hist.get("supplyAssetsUsd"), "supply_assets_usd"),
            _series_to_df(hist.get("borrowAssetsUsd"), "borrow_assets_usd"),
            _series_to_df(hist.get("liquidityAssetsUsd"), "liquidity_assets_usd"),
            _series_to_df(hist.get("netSupplyApy"), "net_supply_apy"),
            _series_to_df((market.get("loanAsset") or {}).get("historicalPriceUsd"), "loan_price_usd"),
        ]
        if self.include_market_totals:
            raw_decimals = (market.get("loanAsset") or {}).get("decimals")
            if raw_decimals is None:
                raise MorphoLoaderException(
                    f"market {self.market_id} on chain {self.chain_id}: loanAsset.decimals missing"
                )
            decimals = int(raw_decimals)
            share_decimals = decimals + _VIRTUAL_SHARES_DECIMALS
            optional_frames.extend((
                _scaled_bigint_series_to_df(hist.get("supplyAssets"), "supply_assets", decimals),
                _scaled_bigint_series_to_df(hist.get("supplyShares"), "supply_shares", share_decimals),
                _scaled_bigint_series_to_df(hist.get("borrowAssets"), "borrow_assets", decimals),
                _scaled_bigint_series_to_df(hist.get("borrowShares"), "borrow_shares", share_decimals),
            ))
        for frame in optional_frames:
            if not frame.empty:
                merged = merged.merge(frame, on="x", how="left")
        self._data = merged.sort_values("x").reset_index(drop=True)

    def transform(self) -> None:
        cols = ["time", "lending_rate", "borrowing_rate", "utilization", "borrow_apy", "supply_apy",
                "rate_at_target", "collateral_price_usd", "supply_rate", "supply_assets_usd",
                "borrow_assets_usd", "liquidity_assets_usd", "net_supply_apy", "loan_price_usd"]
        if self.include_market_totals:
            cols.extend(("supply_assets", "supply_shares", "borrow_assets", "borrow_shares"))
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
            df["supply_rate"] = df["supply_apy"].apply(
                lambda apy: math.log1p(apy) * bar_seconds / SECONDS_PER_YEAR)
        else:
            df["borrowing_rate"] = df["borrow_apy"] * bar_seconds / SECONDS_PER_YEAR
            df["supply_rate"] = df["supply_apy"] * bar_seconds / SECONDS_PER_YEAR
        # The API reports ``rateAtTarget`` as a per-second WAD BigInt while the
        # column is documented as an annual APY; convert it here.
        df["rate_at_target"] = df["rate_at_target"].apply(
            lambda raw: borrow_apy_from_per_second_rate(_wad(raw)) if pd.notna(raw) else raw)
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
            if self.include_market_totals:
                return LendingHistory(
                    lending_rates=[], borrowing_rates=[], time=[], supply_assets=[], supply_shares=[],
                    borrow_assets=[], borrow_shares=[],
                )
            return LendingHistory(lending_rates=[], borrowing_rates=[], time=[])
        return LendingHistory(
            lending_rates=self._data["lending_rate"].astype(float).values,
            borrowing_rates=self._data["borrowing_rate"].astype(float).values,
            time=self._utc_index(),
            utilization=self._data["utilization"].astype(float).values,
            borrow_apy=self._data["borrow_apy"].astype(float).values,
            supply_apy=self._data["supply_apy"].astype(float).values,
            rate_at_target=self._data["rate_at_target"].astype(float).values,
            supply_rate=self._data["supply_rate"].astype(float).values,
            supply_assets_usd=self._data["supply_assets_usd"].astype(float).values,
            borrow_assets_usd=self._data["borrow_assets_usd"].astype(float).values,
            liquidity_assets_usd=self._data["liquidity_assets_usd"].astype(float).values,
            net_supply_apy=self._data["net_supply_apy"].astype(float).values,
            supply_assets=(self._data["supply_assets"].astype(float).values
                           if self.include_market_totals else None),
            supply_shares=(self._data["supply_shares"].astype(float).values
                           if self.include_market_totals else None),
            borrow_assets=(self._data["borrow_assets"].astype(float).values
                           if self.include_market_totals else None),
            borrow_shares=(self._data["borrow_shares"].astype(float).values
                           if self.include_market_totals else None),
        )

    def _price_history(self, col: str) -> PriceHistory:
        if self._data is None or self._data.empty:
            return PriceHistory(prices=[], time=[])
        series = self._data[["time", col]].dropna()
        return PriceHistory(
            prices=series[col].astype(float).values,
            time=pd.to_datetime(series["time"].astype("int64"), unit="s", utc=True),
        )

    def read_collateral_price(self) -> PriceHistory:
        """The API's USD price of the collateral asset over the same window (from the current ``_data``)."""
        return self._price_history("collateral_price_usd")

    def read_loan_price(self) -> PriceHistory:
        """The API's USD price of the loan asset on the market's own bars (from the current ``_data``).

        Converts the token-unit ``supply_assets`` / ``borrow_assets`` totals to
        USD; a loan-price point never adds a bar to the rate grid.
        """
        return self._price_history("loan_price_usd")
