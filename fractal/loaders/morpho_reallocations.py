"""Morpho vault reallocation events from the ``vaultReallocates`` query.

The loader returns one row per event over a UTC ``[start_time, end_time)``
window. ``assets`` is scaled by the market loan token's decimals and signed
from the vault's perspective: ``ReallocateSupply`` is positive and
``ReallocateWithdraw`` is negative. ``shares`` remains the unsigned raw event
quantity because the API does not expose share decimals. ``caller`` is the
``msg.sender`` of ``MetaMorpho.reallocate``; ``is_public_allocator`` marks
callers equal to the configured PublicAllocator of the event's chain.
``log_index`` keeps events with identical payloads in one transaction
distinguishable.

Raw amounts arrive as JSON numbers when small and as decimal strings when
large; both are parsed as exact integers before scaling.

USD values are opt-in (``include_usd=True``) and point-in-time: each event is
priced at the loan asset's hourly ``historicalPriceUsd`` point at or before its
timestamp (``loan_price_usd``), and ``assets_usd = assets * loan_price_usd``
keeps the sign of ``assets``. An event with no price point in the hour up to
it gets ``NaN``. The loan asset's ``priceUsd`` field is deliberately not used:
it is the price at retrieval time, not at the event.
"""
import hashlib
import math
import warnings
from collections.abc import Hashable, Mapping
from datetime import datetime
from typing import Any, ClassVar

import pandas as pd

from fractal.loaders._dt import to_utc, utcnow
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.morpho import MorphoLoaderException, graphql_query

__all__ = ["PUBLIC_ALLOCATORS", "MorphoReallocationsLoader"]

PUBLIC_ALLOCATORS: dict[int, str] = {
    1: "0xfd32fa2ca22c76dd6e550706ad913fc6ce91c75d",
    8453: "0xa090dd1a701408df1d4d0b85b716c87565f90467",
}

_PAGE_SIZE_MAX = 1000
_SKIP_CAP = 10_000
_SLICE_SECONDS = 6 * 60 * 60
_CACHE_SCHEMA = "v2"
_EVENT_TYPES = {"ReallocateSupply", "ReallocateWithdraw"}
_USD_COLUMNS = ("loan_price_usd", "assets_usd")
_HOUR = 3600
_PRICE_CHUNK_SECONDS = 30 * 24 * _HOUR
_PRICE_ASSETS_PER_REQUEST = 50

_REALLOCATIONS_QUERY = """
query VaultReallocations($first: Int!, $skip: Int!, $where: VaultReallocateFilters) {
  vaultReallocates(first: $first, skip: $skip, orderBy: Timestamp, orderDirection: Asc, where: $where) {
    items {
      id timestamp type assets shares caller hash logIndex
      market { marketId loanAsset { address decimals } }
      vault { address chain { id } }
    }
    pageInfo { count countTotal }
  }
}
"""


def _price_query(count: int) -> str:
    """Aliased ``assetByAddress`` hourly USD price series for ``count`` addresses."""
    params = ", ".join(f"$a{i}: String!" for i in range(count))
    fields = "\n".join(
        f"  p{i}: assetByAddress(address: $a{i}, chainId: $chain) {{ historicalPriceUsd(options: $o) {{ x y }} }}"
        for i in range(count)
    )
    return f"query LoanPrices($chain: Int!, $o: TimeseriesOptions, {params}) {{\n{fields}\n}}\n"


def _event_key(item: Mapping[str, Any]) -> Hashable:
    return item.get("id") or (item.get("hash"), item.get("logIndex"))


def _epoch_ceil(moment: datetime) -> int:
    """Smallest whole epoch second not before ``moment``.

    Events carry whole-second timestamps, so a fractional window bound
    ``t`` admits exactly the seconds ``>= ceil(t)``.
    """
    return math.ceil(moment.timestamp())


def _raw_int(value: Any) -> int:
    """Exact integer from a GraphQL BigInt delivered as a number or a string."""
    if isinstance(value, bool):
        raise MorphoLoaderException(f"Morpho GraphQL: expected an integer amount, got {value!r}")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        number = float(text)
        if not math.isfinite(number) or not number.is_integer():
            raise MorphoLoaderException(f"Morpho GraphQL: expected an integer amount, got {value!r}") from None
        return int(number)


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="int64"),
        "chain_id": pd.Series(dtype="int64"),
        "market_id": pd.Series(dtype="object"),
        "vault_address": pd.Series(dtype="object"),
        "event_type": pd.Series(dtype="object"),
        "assets": pd.Series(dtype="float64"),
        "shares": pd.Series(dtype="object"),
        "caller": pd.Series(dtype="object"),
        "transaction_hash": pd.Series(dtype="object"),
        "log_index": pd.Series(dtype="int64"),
        "is_public_allocator": pd.Series(dtype=bool),
        "loan_price_usd": pd.Series(dtype="float64"),
        "assets_usd": pd.Series(dtype="float64"),
    })[list(columns)]


class MorphoReallocationsLoader(Loader):
    """Load MetaMorpho reallocation events for one chain and UTC window.

    The window is walked in 6-hour slices, and a slice whose ``countTotal``
    exceeds one page is bisected in time until each part fits in a single
    page. Offset paging is avoided because ``skip`` pages of Morpho API event
    queries can come back short of ``first`` and miss events across page
    boundaries, and ``orderBy: Timestamp`` does not order events that share a
    second; only a single second that holds more events than one page is read
    with ``skip``, up to the API cap. A slice that yields fewer distinct
    events than its ``countTotal`` emits a ``UserWarning`` instead of failing.

    Args:
        chain_id: EVM chain id.
        start_time: inclusive beginning of the UTC window (fractional seconds
            round up to the next whole second).
        end_time: exclusive end of the UTC window; current UTC time when omitted.
        page_size: GraphQL page size, at most 1000.
        public_allocators: chain-to-address mapping used for
            ``is_public_allocator``; defaults to the verified Ethereum and
            Base PublicAllocator contracts.
        include_usd: add the point-in-time ``loan_price_usd`` and
            ``assets_usd`` columns (extra price requests per loan asset).
        loader_type: on-disk cache format.
        http: injectable client for offline tests.
    """

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "timestamp",
        "chain_id",
        "market_id",
        "vault_address",
        "event_type",
        "assets",
        "shares",
        "caller",
        "transaction_hash",
        "log_index",
        "is_public_allocator",
    )

    def __init__(
        self,
        chain_id: int,
        start_time: datetime,
        end_time: datetime | None = None,
        *,
        page_size: int = _PAGE_SIZE_MAX,
        public_allocators: Mapping[int, str] | None = None,
        include_usd: bool = False,
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
        self.page_size = int(page_size)
        allocators = PUBLIC_ALLOCATORS if public_allocators is None else public_allocators
        self.public_allocators = {int(chain): str(address).lower() for chain, address in allocators.items()}
        self.include_usd = bool(include_usd)
        self._loan_prices: pd.DataFrame | None = None
        self._http = http or HttpClient()

    @property
    def columns(self) -> tuple[str, ...]:
        """Output columns: :attr:`COLUMNS`, plus the USD columns when ``include_usd``."""
        return self.COLUMNS + _USD_COLUMNS if self.include_usd else self.COLUMNS

    def _cache_key(self) -> str:
        allocator_config = ";".join(
            f"{chain}:{address}" for chain, address in sorted(self.public_allocators.items())
        )
        allocator_hash = hashlib.sha256(allocator_config.encode()).hexdigest()[:12]
        return (
            f"schema-{_CACHE_SCHEMA}-chain-{self.chain_id}-"
            f"{_epoch_ceil(self.start_time)}-{_epoch_ceil(self.end_time)}-"
            f"page-{self.page_size}-allocators-{allocator_hash}{'-usd' if self.include_usd else ''}"
        )

    def _page(self, lower: int, upper: int, skip: int) -> tuple[list[dict[str, Any]], int | None]:
        data = graphql_query(self._http, _REALLOCATIONS_QUERY, {
            "first": self.page_size,
            "skip": skip,
            "where": {
                "timestamp_gte": lower,
                "timestamp_lte": upper,
                "chainId_in": [self.chain_id],
            },
        })
        block = data.get("vaultReallocates")
        if not isinstance(block, dict):
            raise MorphoLoaderException("Morpho GraphQL: 'vaultReallocates' missing from the response")
        items = block.get("items") or []
        if not isinstance(items, list):
            raise MorphoLoaderException("Morpho GraphQL: 'vaultReallocates.items' must be a list")
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
                f"{total} reallocations in second {lower}; cannot paginate below the API skip cap"
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
                    f"reallocation slice [{lower}, {upper}] requires skip {skip} above the API cap"
                )
            page, _ = self._page(lower, upper, skip)
        if total is not None and len(rows) < total:
            warnings.warn(
                f"MorphoReallocationsLoader: slice [{lower}, {upper}] on chain {self.chain_id} returned "
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
        if self.include_usd:
            assets = sorted({
                str(((item.get("market") or {}).get("loanAsset") or {}).get("address") or "").lower()
                for item in items
            } - {""})
            self._loan_prices = self._fetch_loan_prices(assets, start, end - 1)

    def _fetch_loan_prices(self, assets: list[str], lower: int, upper: int) -> pd.DataFrame:
        """Hourly USD price points of ``assets`` covering ``[floor_hour(lower), upper]``."""
        rows: list[dict[str, Any]] = []
        first_hour = lower - lower % _HOUR
        for offset in range(0, len(assets), _PRICE_ASSETS_PER_REQUEST):
            batch = assets[offset:offset + _PRICE_ASSETS_PER_REQUEST]
            for chunk_lower in range(first_hour, upper + 1, _PRICE_CHUNK_SECONDS):
                chunk_upper = min(chunk_lower + _PRICE_CHUNK_SECONDS - _HOUR, upper)
                variables: dict[str, Any] = {
                    "chain": self.chain_id,
                    "o": {"startTimestamp": chunk_lower, "endTimestamp": chunk_upper, "interval": "HOUR"},
                }
                variables.update({f"a{i}": address for i, address in enumerate(batch)})
                data = graphql_query(self._http, _price_query(len(batch)), variables)
                for i, address in enumerate(batch):
                    points = (data.get(f"p{i}") or {}).get("historicalPriceUsd") or []
                    rows.extend(
                        {"_loan_asset": address, "timestamp": int(point["x"]), "loan_price_usd": float(point["y"])}
                        for point in points
                        if point.get("x") is not None and point.get("y") is not None
                    )
        prices = pd.DataFrame(rows, columns=["_loan_asset", "timestamp", "loan_price_usd"])
        # Only interval boundaries are snapshots; a non-aligned point is the live value.
        prices = prices[prices["timestamp"] % _HOUR == 0]
        return prices.drop_duplicates(["_loan_asset", "timestamp"], keep="last").reset_index(drop=True)

    def _attach_usd(self, frame: pd.DataFrame) -> pd.DataFrame:
        prices = self._loan_prices
        if prices is None or prices.empty:
            return frame.assign(loan_price_usd=float("nan"), assets_usd=float("nan"))
        prices = prices.astype({"timestamp": "int64", "loan_price_usd": "float64"}).sort_values("timestamp")
        order = frame.assign(_order=range(len(frame))).sort_values("timestamp", kind="stable")
        merged = pd.merge_asof(
            order, prices, on="timestamp", by="_loan_asset", direction="backward", tolerance=_HOUR - 1,
        ).sort_values("_order", kind="stable").drop(columns="_order")
        merged["assets_usd"] = merged["assets"] * merged["loan_price_usd"]
        return merged.reset_index(drop=True)

    def _row(self, item: Mapping[str, Any]) -> dict[str, Any]:
        event_type = str(item.get("type"))
        if event_type not in _EVENT_TYPES:
            raise MorphoLoaderException(f"Morpho GraphQL: unsupported reallocation type {event_type!r}")
        market = item.get("market") or {}
        loan_asset = market.get("loanAsset") or {}
        if loan_asset.get("decimals") is None:
            raise MorphoLoaderException("Morpho GraphQL: reallocation loan-asset decimals missing")
        vault = item.get("vault") or {}
        chain = vault.get("chain") or {}
        chain_id = int(chain.get("id", self.chain_id))
        assets = abs(_raw_int(item["assets"])) / 10 ** int(loan_asset["decimals"])
        if event_type == "ReallocateWithdraw":
            assets = -assets
        caller = str(item.get("caller") or "").lower()
        return {
            "timestamp": int(item["timestamp"]),
            "chain_id": chain_id,
            "market_id": str(market.get("marketId") or "").lower(),
            "vault_address": str(vault.get("address") or "").lower(),
            "event_type": event_type,
            "assets": assets,
            "shares": abs(_raw_int(item["shares"])),
            "caller": caller,
            "transaction_hash": str(item.get("hash") or "").lower(),
            "log_index": int(item.get("logIndex") or 0),
            "is_public_allocator": bool(caller) and caller == self.public_allocators.get(chain_id),
            "_event_key": _event_key(item),
            "_loan_asset": str(loan_asset.get("address") or "").lower(),
        }

    def _normalise_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return _empty_frame(self.columns)
        frame = frame.copy()
        frame["timestamp"] = frame["timestamp"].astype("int64")
        frame["chain_id"] = frame["chain_id"].astype("int64")
        frame["assets"] = frame["assets"].astype("float64")
        frame["shares"] = frame["shares"].map(int).astype("object")
        frame["log_index"] = frame["log_index"].astype("int64")
        for column in ("market_id", "vault_address", "caller", "transaction_hash"):
            frame[column] = frame[column].fillna("").astype(str)
        if frame["is_public_allocator"].dtype != bool:
            frame["is_public_allocator"] = frame["is_public_allocator"].map(
                lambda value: str(value).lower() == "true"
            )
        frame["is_public_allocator"] = frame["is_public_allocator"].astype(bool)
        if self.include_usd:
            for column in _USD_COLUMNS:
                frame[column] = frame[column].astype("float64")
        return frame[list(self.columns)]

    def transform(self) -> None:
        if self._data is None or len(self._data) == 0:
            self._data = _empty_frame(self.columns)
            return
        items = self._data.to_dict("records") if isinstance(self._data, pd.DataFrame) else self._data
        rows = [self._row(item) for item in items]
        frame = pd.DataFrame(rows)
        frame = frame.drop_duplicates("_event_key", keep="first")
        frame = frame.sort_values(
            ["timestamp", "chain_id", "transaction_hash", "log_index"], kind="stable"
        ).reset_index(drop=True)
        if self.include_usd:
            frame = self._attach_usd(frame)
        self._data = self._normalise_frame(frame)

    def read(self, with_run: bool = False) -> pd.DataFrame:
        """Return the tidy reallocation-event frame, running extraction when requested."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        frame = self._data.drop(columns=["Unnamed: 0"], errors="ignore")
        self._data = self._normalise_frame(frame)
        return self._data.copy()
