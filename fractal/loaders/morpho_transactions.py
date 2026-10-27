"""Morpho Blue market events from the ``marketTransactions`` query.

The loader returns one row per market event (``Supply``, ``Withdraw``,
``Borrow``, ``Repay``, ``SupplyCollateral``, ``WithdrawCollateral`` and
``Liquidation``) on one chain over a UTC ``[start_time, end_time)`` window,
optionally restricted to a set of markets and event types.

The API's ``data`` field is a union with one member per event family, and each
family is selected with its own fragment:

* ``MarketTransactionTransferData { assets shares }`` for the four loan-token
  events;
* ``MarketTransactionCollateralTransferData { assets }`` for the two collateral
  events, which carry no shares;
* ``MarketTransactionLiquidationData { liquidator repaidAssets repaidShares
  seizedAssets badDebtAssets badDebtShares }`` for liquidations.

Selecting only the first fragment returns an empty ``data`` object for the
other two families, so collateral and liquidation rows would lose their
amounts.

Units: ``assets`` is unsigned and scaled by the decimals of the token named in
``asset_side``: the loan token for loan events and for a liquidation (the
repaid debt), the collateral token for collateral events. ``seized_assets`` is
in collateral-token units and ``bad_debt_assets`` in loan-token units; both are
``0.0`` outside liquidations. ``shares`` and ``bad_debt_shares`` stay raw
integers because the API does not expose share decimals; ``shares`` is ``0``
for collateral events and holds the repaid shares for a liquidation.

Raw amounts arrive as JSON numbers when small and as decimal strings when
large; both are parsed as exact integers before scaling.
"""
import hashlib
import math
import warnings
from collections.abc import Hashable, Iterable, Mapping
from datetime import datetime
from typing import Any, ClassVar

import pandas as pd

from fractal.loaders._dt import to_utc, utcnow
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.morpho import MorphoLoaderException, _validate_market_id, graphql_query

__all__ = [
    "COLLATERAL_TRANSACTION_TYPES",
    "LOAN_TRANSACTION_TYPES",
    "MARKET_TRANSACTION_TYPES",
    "MorphoMarketTransactionsLoader",
]

LOAN_TRANSACTION_TYPES = ("Supply", "Withdraw", "Borrow", "Repay")
COLLATERAL_TRANSACTION_TYPES = ("SupplyCollateral", "WithdrawCollateral")
MARKET_TRANSACTION_TYPES: tuple[str, ...] = (
    *LOAN_TRANSACTION_TYPES,
    *COLLATERAL_TRANSACTION_TYPES,
    "Liquidation",
)

_PAGE_SIZE_MAX = 1000
_SKIP_CAP = 10_000
_SLICE_SECONDS = 24 * 60 * 60
_CACHE_SCHEMA = "v1"

_TRANSACTIONS_QUERY = """
query MarketTransactions($first: Int!, $skip: Int!, $where: MarketTransactionFilters) {
  marketTransactions(first: $first, skip: $skip, orderBy: Timestamp, orderDirection: Asc, where: $where) {
    items {
      txHash timestamp blockNumber txIndex logIndex type
      chain { id }
      user { address }
      market {
        marketId
        loanAsset { symbol decimals }
        collateralAsset { symbol decimals }
      }
      data {
        __typename
        ... on MarketTransactionTransferData { assets shares }
        ... on MarketTransactionCollateralTransferData { assets }
        ... on MarketTransactionLiquidationData {
          liquidator repaidAssets repaidShares seizedAssets badDebtAssets badDebtShares
        }
      }
    }
    pageInfo { count countTotal }
  }
}
"""

_STRING_COLUMNS = (
    "transaction_hash",
    "market_id",
    "event_type",
    "user",
    "asset_side",
    "loan_symbol",
    "collateral_symbol",
    "liquidator",
)
_INT_COLUMNS = ("timestamp", "chain_id", "block_number", "tx_index", "log_index")
_FLOAT_COLUMNS = ("assets", "seized_assets", "bad_debt_assets")
_RAW_INT_COLUMNS = ("shares", "bad_debt_shares")


def _event_key(item: Mapping[str, Any]) -> Hashable:
    return (str(item.get("txHash") or "").lower(), item.get("logIndex"))


def _epoch_ceil(moment: datetime) -> int:
    """Smallest whole epoch second not before ``moment``.

    Events carry whole-second timestamps, so a fractional window bound
    ``t`` admits exactly the seconds ``>= ceil(t)``.
    """
    return math.ceil(moment.timestamp())


def _raw_int(value: Any) -> int:
    """Exact integer from a GraphQL BigInt delivered as a number or a string."""
    if isinstance(value, bool) or value is None:
        raise MorphoLoaderException(f"Morpho GraphQL: expected an integer amount, got {value!r}")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        try:
            number = float(text)
        except ValueError:
            number = math.nan
        if not math.isfinite(number) or not number.is_integer():
            raise MorphoLoaderException(f"Morpho GraphQL: expected an integer amount, got {value!r}") from None
        return int(number)


def _decimals(asset: Mapping[str, Any] | None, side: str, event_type: str) -> int:
    decimals = (asset or {}).get("decimals")
    if decimals is None:
        raise MorphoLoaderException(f"Morpho GraphQL: {side}-asset decimals missing for a {event_type} event")
    return int(decimals)


def _field(data: Mapping[str, Any], name: str, event_type: str) -> Any:
    if data.get(name) is None:
        raise MorphoLoaderException(f"Morpho GraphQL: {event_type} event data lacks {name!r}")
    return data[name]


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    dtypes = dict.fromkeys(_INT_COLUMNS, "int64")
    dtypes.update(dict.fromkeys(_FLOAT_COLUMNS, "float64"))
    dtypes.update(dict.fromkeys(_STRING_COLUMNS + _RAW_INT_COLUMNS, "object"))
    return pd.DataFrame({column: pd.Series(dtype=dtypes[column]) for column in columns})


def _normalise_types(types: Iterable[str] | None) -> tuple[str, ...] | None:
    if types is None:
        return None
    if isinstance(types, str):
        types = (types,)
    chosen = set()
    for event_type in types:
        if event_type not in MARKET_TRANSACTION_TYPES:
            raise ValueError(
                f"unknown market transaction type {event_type!r}; expected one of {MARKET_TRANSACTION_TYPES}"
            )
        chosen.add(event_type)
    if not chosen:
        raise ValueError("types must name at least one market transaction type, or be None for all")
    return tuple(event_type for event_type in MARKET_TRANSACTION_TYPES if event_type in chosen)


def _normalise_market_ids(market_ids: Iterable[str] | None) -> tuple[str, ...] | None:
    if market_ids is None:
        return None
    if isinstance(market_ids, str):
        market_ids = (market_ids,)
    chosen = tuple(sorted({_validate_market_id(market_id) for market_id in market_ids}))
    if not chosen:
        raise ValueError("market_ids must name at least one market, or be None for every market on the chain")
    return chosen


class MorphoMarketTransactionsLoader(Loader):
    """Load Morpho Blue market events for one chain and UTC window.

    The window is walked in 1-day slices, and a slice whose ``countTotal``
    exceeds ``page_size`` is bisected until each part fits in one page.
    Offset paging is avoided because ``skip`` pages of this query can come
    back short of ``first`` and miss events across page boundaries; only a
    single second that holds more events than one page is read with
    ``skip``, up to the API cap. A slice that yields fewer distinct events
    than its ``countTotal`` emits a ``UserWarning`` instead of failing. Events are deduplicated by
    transaction hash and log index and sorted by timestamp, block and log
    index.

    Args:
        chain_id: EVM chain id.
        start_time: inclusive beginning of the UTC window (fractional seconds
            round up to the next whole second).
        end_time: exclusive end of the UTC window; current UTC time when omitted.
        market_ids: Morpho Blue market ids (``0x`` + 64 hex) to keep; every
            market on the chain when omitted.
        types: event types to keep, a subset of
            :data:`MARKET_TRANSACTION_TYPES`; every type when omitted.
        page_size: GraphQL page size, at most 1000; also the bisection
            target, so a smaller page means more, smaller slices.
        loader_type: on-disk cache format.
        http: injectable client for offline tests.
    """

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "timestamp",
        "chain_id",
        "block_number",
        "tx_index",
        "log_index",
        "transaction_hash",
        "market_id",
        "event_type",
        "user",
        "asset_side",
        "assets",
        "shares",
        "seized_assets",
        "bad_debt_assets",
        "bad_debt_shares",
        "liquidator",
        "loan_symbol",
        "collateral_symbol",
    )

    def __init__(
        self,
        chain_id: int,
        start_time: datetime,
        end_time: datetime | None = None,
        *,
        market_ids: Iterable[str] | None = None,
        types: Iterable[str] | None = None,
        page_size: int = _PAGE_SIZE_MAX,
        loader_type: LoaderType = LoaderType.CSV,
        http: HttpClient | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if not 0 < page_size <= _PAGE_SIZE_MAX:
            raise ValueError(f"page_size must be in [1, {_PAGE_SIZE_MAX}], got {page_size}")
        self.chain_id = int(chain_id)
        self.start_time = to_utc(start_time)
        self.end_time = to_utc(end_time) if end_time is not None else utcnow()
        if self.end_time < self.start_time:
            raise ValueError(f"end_time {self.end_time} precedes start_time {self.start_time}")
        self.market_ids = _normalise_market_ids(market_ids)
        self.types = _normalise_types(types)
        self.page_size = int(page_size)
        self._http = http or HttpClient()

    def _cache_key(self) -> str:
        filters = "|".join((
            ",".join(self.market_ids) if self.market_ids is not None else "*",
            ",".join(self.types) if self.types is not None else "*",
        ))
        filter_hash = hashlib.sha256(filters.encode()).hexdigest()[:12]
        return (
            f"schema-{_CACHE_SCHEMA}-chain-{self.chain_id}-"
            f"{_epoch_ceil(self.start_time)}-{_epoch_ceil(self.end_time)}-"
            f"page-{self.page_size}-filters-{filter_hash}"
        )

    def _where(self, lower: int, upper: int) -> dict[str, Any]:
        where: dict[str, Any] = {
            "timestamp_gte": lower,
            "timestamp_lte": upper,
            "chainId_in": [self.chain_id],
        }
        if self.market_ids is not None:
            where["marketUniqueKey_in"] = list(self.market_ids)
        if self.types is not None:
            where["type_in"] = list(self.types)
        return where

    def _page(self, lower: int, upper: int, skip: int) -> tuple[list[dict[str, Any]], int | None]:
        data = graphql_query(self._http, _TRANSACTIONS_QUERY, {
            "first": self.page_size,
            "skip": skip,
            "where": self._where(lower, upper),
        })
        block = data.get("marketTransactions")
        if not isinstance(block, dict):
            raise MorphoLoaderException("Morpho GraphQL: 'marketTransactions' missing from the response")
        items = block.get("items") or []
        if not isinstance(items, list):
            raise MorphoLoaderException("Morpho GraphQL: 'marketTransactions.items' must be a list")
        page_info = block.get("pageInfo") or {}
        total = page_info.get("countTotal")
        return items, None if total is None else int(total)

    def _extract_slice(self, lower: int, upper: int) -> list[dict[str, Any]]:
        page, total = self._page(lower, upper, 0)
        if total is not None and total > self.page_size and lower < upper:
            middle = (lower + upper) // 2
            return self._extract_slice(lower, middle) + self._extract_slice(middle + 1, upper)
        if total is not None and total > _SKIP_CAP:
            raise MorphoLoaderException(
                f"{total} market transactions in second {lower}; cannot paginate below the API skip cap"
            )

        rows: list[dict[str, Any]] = []
        seen: set[Hashable] = set()
        skip = 0
        while page:
            new_rows = [item for item in page if _event_key(item) not in seen]
            if not new_rows:
                break
            rows.extend(new_rows)
            seen.update(_event_key(item) for item in page)
            skip += self.page_size
            # Pages can come back short of ``first`` while more events remain,
            # so a known ``countTotal`` decides when to stop.
            done = len(page) < self.page_size if total is None else skip >= total
            if done:
                break
            if skip > _SKIP_CAP:
                raise MorphoLoaderException(
                    f"market-transaction slice [{lower}, {upper}] requires skip {skip} above the API cap"
                )
            page, _ = self._page(lower, upper, skip)
        if total is not None and len(rows) < total:
            warnings.warn(
                f"MorphoMarketTransactionsLoader: slice [{lower}, {upper}] on chain {self.chain_id} returned "
                f"{len(rows)} distinct events of countTotal {total}",
                stacklevel=3,
            )
        return rows

    def extract(self) -> None:
        start = _epoch_ceil(self.start_time)
        end = _epoch_ceil(self.end_time)
        items: list[dict[str, Any]] = []
        for lower in range(start, end, _SLICE_SECONDS):
            upper = min(lower + _SLICE_SECONDS, end) - 1
            items.extend(self._extract_slice(lower, upper))
        self._data = items

    def _row(self, item: Mapping[str, Any]) -> dict[str, Any]:
        event_type = str(item.get("type"))
        if event_type not in MARKET_TRANSACTION_TYPES:
            raise MorphoLoaderException(f"Morpho GraphQL: unsupported market transaction type {event_type!r}")
        market = item.get("market") or {}
        loan_asset = market.get("loanAsset") or {}
        collateral_asset = market.get("collateralAsset") or {}
        data = item.get("data") or {}
        row: dict[str, Any] = {
            "shares": 0,
            "seized_assets": 0.0,
            "bad_debt_assets": 0.0,
            "bad_debt_shares": 0,
            "liquidator": "",
        }
        if event_type in COLLATERAL_TRANSACTION_TYPES:
            decimals = _decimals(collateral_asset, "collateral", event_type)
            row["asset_side"] = "collateral"
            row["assets"] = abs(_raw_int(_field(data, "assets", event_type))) / 10**decimals
        elif event_type == "Liquidation":
            loan_decimals = _decimals(loan_asset, "loan", event_type)
            collateral_decimals = _decimals(collateral_asset, "collateral", event_type)
            row["asset_side"] = "loan"
            row["assets"] = abs(_raw_int(_field(data, "repaidAssets", event_type))) / 10**loan_decimals
            row["shares"] = abs(_raw_int(_field(data, "repaidShares", event_type)))
            row["seized_assets"] = abs(_raw_int(_field(data, "seizedAssets", event_type))) / 10**collateral_decimals
            row["bad_debt_assets"] = abs(_raw_int(_field(data, "badDebtAssets", event_type))) / 10**loan_decimals
            row["bad_debt_shares"] = abs(_raw_int(_field(data, "badDebtShares", event_type)))
            row["liquidator"] = str(_field(data, "liquidator", event_type)).lower()
        else:
            decimals = _decimals(loan_asset, "loan", event_type)
            row["asset_side"] = "loan"
            row["assets"] = abs(_raw_int(_field(data, "assets", event_type))) / 10**decimals
            row["shares"] = abs(_raw_int(_field(data, "shares", event_type)))
        chain = item.get("chain") or {}
        row.update({
            "timestamp": _raw_int(item.get("timestamp")),
            "chain_id": int(chain.get("id", self.chain_id)),
            "block_number": _raw_int(item.get("blockNumber")),
            "tx_index": int(item.get("txIndex") or 0),
            "log_index": int(item.get("logIndex") or 0),
            "transaction_hash": str(item.get("txHash") or "").lower(),
            "market_id": str(market.get("marketId") or "").lower(),
            "event_type": event_type,
            "user": str((item.get("user") or {}).get("address") or "").lower(),
            "loan_symbol": str(loan_asset.get("symbol") or ""),
            "collateral_symbol": str(collateral_asset.get("symbol") or ""),
            "_event_key": _event_key(item),
        })
        return row

    def _normalise_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return _empty_frame(self.COLUMNS)
        frame = frame.copy()
        for column in _INT_COLUMNS:
            frame[column] = frame[column].astype("int64")
        for column in _FLOAT_COLUMNS:
            frame[column] = frame[column].astype("float64")
        for column in _RAW_INT_COLUMNS:
            frame[column] = frame[column].map(int).astype("object")
        for column in _STRING_COLUMNS:
            frame[column] = frame[column].fillna("").astype(str).astype("object")
        return frame[list(self.COLUMNS)]

    def transform(self) -> None:
        if self._data is None or len(self._data) == 0:
            self._data = _empty_frame(self.COLUMNS)
            return
        items = self._data.to_dict("records") if isinstance(self._data, pd.DataFrame) else self._data
        frame = pd.DataFrame([self._row(item) for item in items])
        frame = frame.drop_duplicates("_event_key", keep="first")
        frame = frame.sort_values(
            ["timestamp", "block_number", "log_index"], kind="stable"
        ).reset_index(drop=True)
        self._data = self._normalise_frame(frame)

    def read(self, with_run: bool = False) -> pd.DataFrame:
        """Return the tidy market-event frame, running extraction when requested."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        frame = self._data.drop(columns=["Unnamed: 0"], errors="ignore")
        self._data = self._normalise_frame(frame)
        return self._data.copy()
