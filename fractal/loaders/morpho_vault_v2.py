"""Morpho Vault V2 discovery and allocation/depositor activity.

The loader uses the public Morpho GraphQL API to discover Vault V2 vaults on
one chain, fetch their current adapter positions, and load allocation and
vault transactions over an inclusive UTC time window. Allocation rows are
mapped to Morpho Blue markets only when the event's adapter exposes exactly
one market position through the API. Ambiguous or absent mappings are retained
with ``market_id=None`` and counted in the attached coverage summary.

Amounts are returned in native vault-asset units. ``assets_delta_native`` is
positive for allocations/deposits, negative for deallocations/withdrawals,
and zero for share transfers. ``change_native`` preserves the allocation
API's interest-inclusive adapter change without using it as a flow sign.
"""
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc, utcnow
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.morpho import MorphoLoaderException, graphql_query

__all__ = [
    "MorphoVaultV2Coverage",
    "MorphoVaultV2History",
    "MorphoVaultV2Listing",
    "MorphoVaultV2Loader",
]

_DEFAULT_TVL_FLOOR_USD = 1_000_000.0
_ALLOCATION_SLICE_SECONDS = 7 * 86_400
_TRANSACTION_SLICE_SECONDS = 86_400
_TRANSACTION_SKIP_CAP = 10_000
_CACHE_SCHEMA = "v1"

_VAULTS_QUERY = """
query VaultV2Vaults($first: Int!, $skip: Int!, $where: VaultV2sFilters) {
  vaultV2s(first: $first, skip: $skip, where: $where) {
    items {
      address chain { id } asset { address symbol decimals } totalAssetsUsd
      liquidityAdapter { address }
      allocators { allocator { address } }
    }
  }
}
"""

_ADAPTERS_QUERY = """
query VaultV2Adapters($address: String!, $chainId: Int!) {
  vaultV2ByAddress(address: $address, chainId: $chainId) {
    adapters(first: 20) {
      items {
        address type
        ... on MorphoMarketV1Adapter {
          positions(first: 200) { items { market { marketId } } }
        }
      }
    }
  }
}
"""

_ALLOCATIONS_QUERY = """
query VaultV2Allocations(
  $first: Int!, $skip: Int!, $chainId: Int!, $vaultAddress: String!,
  $where: VaultV2AllocationTransactionFilters
) {
  vaultV2AllocationTransactions(
    first: $first, skip: $skip, chainId: $chainId, vaultAddress: $vaultAddress,
    orderBy: Timestamp, orderDirection: Asc, where: $where
  ) {
    items { txHash logIndex timestamp type adapter sender assets change ids }
    pageInfo { countTotal }
  }
}
"""

_TRANSACTIONS_QUERY = """
query VaultV2Transactions($first: Int!, $skip: Int!, $where: VaultV2TransactionFilters) {
  vaultV2transactions(first: $first, skip: $skip, orderBy: Time, orderDirection: Asc, where: $where) {
    items { txHash logIndex timestamp type assets vault { address chain { id } } }
    pageInfo { countTotal }
  }
}
"""

_ALLOCATION_COLUMNS = [
    "time",
    "chain_id",
    "vault_address",
    "asset_address",
    "asset_symbol",
    "type",
    "adapter_address",
    "sender_address",
    "assets_native",
    "assets_delta_native",
    "change_native",
    "market_id",
    "map_method",
    "ids",
    "tx_hash",
    "log_index",
]
_TRANSACTION_COLUMNS = [
    "time",
    "chain_id",
    "vault_address",
    "asset_address",
    "asset_symbol",
    "type",
    "assets_native",
    "assets_delta_native",
    "tx_hash",
    "log_index",
]


@dataclass(frozen=True)
class MorphoVaultV2Listing:
    """One Vault V2 vault and its current asset/configuration.

    Attributes:
        chain_id: EVM chain id.
        vault_address: Vault V2 contract address.
        asset_address: Underlying vault asset address.
        asset_symbol: Symbol reported by the Morpho API.
        asset_decimals: Decimal precision used to scale event amounts.
        total_assets_usd: Current API TVL in USD, or ``None`` when absent.
        liquidity_adapter_address: Current liquidity adapter, if configured.
        allocators: Current allocator addresses reported by the API.
    """

    chain_id: int
    vault_address: str
    asset_address: str
    asset_symbol: str
    asset_decimals: int
    total_assets_usd: float | None
    liquidity_adapter_address: str | None
    allocators: tuple[str, ...]


@dataclass(frozen=True)
class MorphoVaultV2Coverage:
    """Coverage of allocation-to-Morpho-Blue market mapping.

    ``mapped_assets_fraction`` weights rows by absolute native-asset flow;
    fractions are ``None`` when there are no allocation rows or no volume.
    """

    total_allocations: int
    mapped_allocations: int
    unmapped_allocations: int
    mapped_fraction: float | None
    total_assets_native: float
    mapped_assets_native: float
    mapped_assets_fraction: float | None


@dataclass(frozen=True)
class MorphoVaultV2History:
    """Vault listings, allocation events, depositor transactions and mapping coverage."""

    vaults: tuple[MorphoVaultV2Listing, ...]
    allocations: pd.DataFrame
    transactions: pd.DataFrame
    coverage: MorphoVaultV2Coverage


def _lower(value: Any) -> str:
    return str(value or "").lower()


def _scale(value: Any, decimals: int) -> float | None:
    if value is None:
        return None
    return int(value) / 10**decimals


def _event_key(item: dict[str, Any]) -> tuple[str, int]:
    return _lower(item.get("txHash")), int(item.get("logIndex") or 0)


def _query_page(
    http: HttpClient,
    query: str,
    root: str,
    variables: dict[str, Any],
    page_size: int,
    skip: int,
) -> tuple[list[dict[str, Any]], int | None]:
    data = graphql_query(http, query, {**variables, "first": page_size, "skip": skip})
    block = data.get(root)
    if not isinstance(block, dict):
        raise MorphoLoaderException(f"Morpho GraphQL: '{root}' missing from the response")
    page = block.get("items")
    if not isinstance(page, list):
        raise MorphoLoaderException(f"Morpho GraphQL: '{root}.items' missing from the response")
    raw_total = (block.get("pageInfo") or {}).get("countTotal")
    return page, None if raw_total is None else int(raw_total)


def _paged_items(
    http: HttpClient,
    query: str,
    root: str,
    variables: dict[str, Any],
    page_size: int,
    identity: Callable[[dict[str, Any]], Hashable],
    *,
    initial_page: list[dict[str, Any]] | None = None,
    total: int | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[Hashable] = set()
    skip = 0
    while True:
        if skip == 0 and initial_page is not None:
            page = initial_page
        else:
            page, page_total = _query_page(http, query, root, variables, page_size, skip)
            total = total if total is not None else page_total
        new_items = []
        for item in page:
            key = identity(item)
            if key not in seen:
                seen.add(key)
                new_items.append(item)
        items.extend(new_items)
        skip += page_size
        if len(page) < page_size or not new_items or (total is not None and skip >= total):
            return items


def _allocation_slice(
    http: HttpClient,
    variables: dict[str, Any],
    page_size: int,
    lo: int,
    hi: int,
) -> list[dict[str, Any]]:
    bounded = {**variables, "where": {"timestamp_gte": lo, "timestamp_lte": hi}}
    page, total = _query_page(
        http, _ALLOCATIONS_QUERY, "vaultV2AllocationTransactions", bounded, page_size, 0
    )
    if total is not None and total >= page_size:
        if hi <= lo:
            raise MorphoLoaderException(
                f"{total} Vault V2 allocations in one second for {variables['vaultAddress']}"
            )
        mid = (lo + hi) // 2
        return (
            _allocation_slice(http, variables, page_size, lo, mid)
            + _allocation_slice(http, variables, page_size, mid + 1, hi)
        )
    if total is not None or len(page) < page_size:
        return page
    return _paged_items(
        http,
        _ALLOCATIONS_QUERY,
        "vaultV2AllocationTransactions",
        bounded,
        page_size,
        _event_key,
        initial_page=page,
        total=total,
    )


def _transaction_slice(
    http: HttpClient,
    variables: dict[str, Any],
    page_size: int,
    lo: int,
    hi: int,
) -> list[dict[str, Any]]:
    bounded = {
        **variables,
        "where": {**variables["where"], "timestamp_gte": lo, "timestamp_lte": hi},
    }
    page, total = _query_page(http, _TRANSACTIONS_QUERY, "vaultV2transactions", bounded, page_size, 0)
    if total is not None and total > _TRANSACTION_SKIP_CAP:
        if hi <= lo:
            raise MorphoLoaderException(f"{total} Vault V2 transactions in one second")
        mid = (lo + hi) // 2
        return (
            _transaction_slice(http, variables, page_size, lo, mid)
            + _transaction_slice(http, variables, page_size, mid + 1, hi)
        )
    return _paged_items(
        http,
        _TRANSACTIONS_QUERY,
        "vaultV2transactions",
        bounded,
        page_size,
        _event_key,
        initial_page=page,
        total=total,
    )


def _slices(start: int, end: int, width: int) -> list[tuple[int, int]]:
    return [(lo, min(lo + width - 1, end)) for lo in range(start, end + 1, width)]


def _listing(item: dict[str, Any]) -> MorphoVaultV2Listing:
    asset = item.get("asset") or {}
    total_assets_usd = item.get("totalAssetsUsd")
    allocators = {
        _lower((entry.get("allocator") or {}).get("address"))
        for entry in item.get("allocators") or []
        if (entry.get("allocator") or {}).get("address")
    }
    return MorphoVaultV2Listing(
        chain_id=int((item.get("chain") or {})["id"]),
        vault_address=_lower(item["address"]),
        asset_address=_lower(asset["address"]),
        asset_symbol=str(asset.get("symbol") or ""),
        asset_decimals=int(asset["decimals"]),
        total_assets_usd=None if total_assets_usd is None else float(total_assets_usd),
        liquidity_adapter_address=_lower((item.get("liquidityAdapter") or {}).get("address")) or None,
        allocators=tuple(sorted(allocators)),
    )


def _coverage(allocations: pd.DataFrame) -> MorphoVaultV2Coverage:
    total = len(allocations)
    mapped_mask = allocations["market_id"].notna() if total else pd.Series(dtype=bool)
    mapped = int(mapped_mask.sum())
    amounts = (
        pd.to_numeric(allocations["assets_delta_native"], errors="coerce").abs()
        if total
        else pd.Series(dtype=float)
    )
    volume = float(amounts.sum())
    mapped_volume = float(amounts[mapped_mask].sum()) if total else 0.0
    return MorphoVaultV2Coverage(
        total_allocations=total,
        mapped_allocations=mapped,
        unmapped_allocations=total - mapped,
        mapped_fraction=mapped / total if total else None,
        total_assets_native=volume,
        mapped_assets_native=mapped_volume,
        mapped_assets_fraction=mapped_volume / volume if volume else None,
    )


class MorphoVaultV2Loader(Loader):
    """Load Vault V2 listings, allocation events and depositor transactions.

    Args:
        chain_id: EVM chain id to scan.
        start_time / end_time: Inclusive UTC event window. ``end_time``
            defaults to construction time.
        tvl_floor_usd: Minimum current vault TVL requested server-side.
            Defaults to ``1_000_000`` USD, matching the research collector.
        vault_page_size: Number of vaults requested per discovery page.
        event_page_size: Number of allocation/transaction rows requested per
            page; the Morpho API accepts at most 1,000.
        loader_type: Cache format. Only ``PICKLE`` is supported because the
            result contains dataclasses and multiple data frames.
        http: Injectable HTTP client for offline tests.
    """

    def __init__(
        self,
        chain_id: int,
        start_time: datetime,
        end_time: datetime | None = None,
        *,
        tvl_floor_usd: float = _DEFAULT_TVL_FLOOR_USD,
        vault_page_size: int = 200,
        event_page_size: int = 1000,
        loader_type: LoaderType = LoaderType.PICKLE,
        http: HttpClient | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if loader_type is not LoaderType.PICKLE:
            raise ValueError("MorphoVaultV2Loader requires LoaderType.PICKLE")
        if tvl_floor_usd < 0:
            raise ValueError(f"tvl_floor_usd must be >= 0, got {tvl_floor_usd}")
        if vault_page_size <= 0:
            raise ValueError(f"vault_page_size must be > 0, got {vault_page_size}")
        if not 0 < event_page_size <= 1000:
            raise ValueError(f"event_page_size must be in [1, 1000], got {event_page_size}")
        self.chain_id = int(chain_id)
        self.start_time = to_utc(start_time)
        self.end_time = to_utc(end_time) if end_time is not None else utcnow()
        if self.end_time < self.start_time:
            raise ValueError(f"end_time {self.end_time} precedes start_time {self.start_time}")
        self.tvl_floor_usd = float(tvl_floor_usd)
        self.vault_page_size = int(vault_page_size)
        self.event_page_size = int(event_page_size)
        self._http = http or HttpClient()

    def _cache_key(self) -> str:
        return (
            f"schema-{_CACHE_SCHEMA}-chain-{self.chain_id}-start-{to_seconds(self.start_time)}"
            f"-end-{to_seconds(self.end_time)}-tvl-{self.tvl_floor_usd:g}"
            f"-vault-page-{self.vault_page_size}-event-page-{self.event_page_size}"
        )

    def extract(self) -> None:
        """Fetch vaults, adapter positions, allocations and vault transactions."""
        vaults = _paged_items(
            self._http,
            _VAULTS_QUERY,
            "vaultV2s",
            {"where": {"chainId_in": [self.chain_id], "totalAssetsUsd_gte": self.tvl_floor_usd}},
            self.vault_page_size,
            lambda item: (int((item.get("chain") or {}).get("id", self.chain_id)), _lower(item.get("address"))),
        )
        adapters: list[dict[str, Any]] = []
        allocations: list[dict[str, Any]] = []
        transactions: list[dict[str, Any]] = []
        start = to_seconds(self.start_time)
        end = to_seconds(self.end_time)
        for item in vaults:
            vault_address = _lower(item["address"])
            adapter_data = graphql_query(
                self._http,
                _ADAPTERS_QUERY,
                {"address": vault_address, "chainId": self.chain_id},
            )
            vault = adapter_data.get("vaultV2ByAddress") or {}
            adapter_block = vault.get("adapters") or {}
            adapter_items = adapter_block.get("items") or []
            if not isinstance(adapter_items, list):
                raise MorphoLoaderException("Morpho GraphQL: 'vaultV2ByAddress.adapters.items' is malformed")
            adapters.extend({"vault_address": vault_address, **adapter} for adapter in adapter_items)
            allocation_variables = {"chainId": self.chain_id, "vaultAddress": vault_address}
            for lo, hi in _slices(start, end, _ALLOCATION_SLICE_SECONDS):
                allocations.extend(
                    {"vault_address": vault_address, **event}
                    for event in _allocation_slice(
                        self._http, allocation_variables, self.event_page_size, lo, hi
                    )
                )
            transaction_variables = {
                "where": {
                    "chainId_in": [self.chain_id],
                    "vaultAddress_in": [vault_address],
                    "type_in": ["Deposit", "Withdraw", "Transfer"],
                }
            }
            for lo, hi in _slices(start, end, _TRANSACTION_SLICE_SECONDS):
                transactions.extend(
                    _transaction_slice(
                        self._http, transaction_variables, self.event_page_size, lo, hi
                    )
                )
        self._data = {
            "vaults": vaults,
            "adapters": adapters,
            "allocations": allocations,
            "transactions": transactions,
        }

    def transform(self) -> None:
        """Scale amounts, sign flows and attach unambiguous adapter-position market ids."""
        raw = self._data or {}
        listings = tuple(sorted((_listing(item) for item in raw.get("vaults", [])), key=lambda row: row.vault_address))
        by_vault = {listing.vault_address: listing for listing in listings}
        positions: dict[tuple[str, str], set[str]] = {}
        for adapter in raw.get("adapters", []):
            key = (_lower(adapter.get("vault_address")), _lower(adapter.get("address")))
            market_ids = {
                _lower((position.get("market") or {}).get("marketId"))
                for position in ((adapter.get("positions") or {}).get("items") or [])
                if (position.get("market") or {}).get("marketId")
            }
            positions.setdefault(key, set()).update(market_ids)

        allocation_rows = []
        for item in raw.get("allocations", []):
            vault_address = _lower(item.get("vault_address"))
            listing = by_vault[vault_address]
            event_type = str(item.get("type") or "")
            if event_type not in {"Allocate", "Deallocate"}:
                raise MorphoLoaderException(f"unsupported Vault V2 allocation type {event_type!r}")
            assets_native = _scale(item.get("assets"), listing.asset_decimals)
            change_native = _scale(item.get("change"), listing.asset_decimals)
            adapter_address = _lower(item.get("adapter"))
            candidates = positions.get((vault_address, adapter_address), set())
            market_id = next(iter(candidates)) if len(candidates) == 1 else None
            sign = 1.0 if event_type == "Allocate" else -1.0
            allocation_rows.append(
                {
                    "time": pd.to_datetime(int(item["timestamp"]), unit="s", utc=True),
                    "chain_id": self.chain_id,
                    "vault_address": vault_address,
                    "asset_address": listing.asset_address,
                    "asset_symbol": listing.asset_symbol,
                    "type": event_type,
                    "adapter_address": adapter_address,
                    "sender_address": _lower(item.get("sender")),
                    "assets_native": assets_native,
                    "assets_delta_native": None if assets_native is None else sign * assets_native,
                    "change_native": change_native,
                    "market_id": market_id,
                    "map_method": "adapter_position" if market_id is not None else "none",
                    "ids": tuple(_lower(value) for value in item.get("ids") or []),
                    "tx_hash": _lower(item.get("txHash")),
                    "log_index": int(item.get("logIndex") or 0),
                }
            )
        allocations = pd.DataFrame(allocation_rows, columns=_ALLOCATION_COLUMNS)
        if not allocations.empty:
            allocations = (
                allocations.drop_duplicates(["chain_id", "vault_address", "tx_hash", "log_index"])
                .sort_values(["time", "vault_address", "tx_hash", "log_index"], kind="stable")
                .reset_index(drop=True)
            )

        transaction_rows = []
        for item in raw.get("transactions", []):
            vault_address = _lower((item.get("vault") or {}).get("address"))
            listing = by_vault.get(vault_address)
            if listing is None:
                continue
            event_type = str(item.get("type") or "")
            if event_type not in {"Deposit", "Withdraw", "Transfer"}:
                raise MorphoLoaderException(f"unsupported Vault V2 transaction type {event_type!r}")
            assets_native = _scale(item.get("assets"), listing.asset_decimals)
            sign = {"Deposit": 1.0, "Withdraw": -1.0, "Transfer": 0.0}[event_type]
            transaction_rows.append(
                {
                    "time": pd.to_datetime(int(item["timestamp"]), unit="s", utc=True),
                    "chain_id": self.chain_id,
                    "vault_address": vault_address,
                    "asset_address": listing.asset_address,
                    "asset_symbol": listing.asset_symbol,
                    "type": event_type,
                    "assets_native": assets_native,
                    "assets_delta_native": None if assets_native is None else sign * assets_native,
                    "tx_hash": _lower(item.get("txHash")),
                    "log_index": int(item.get("logIndex") or 0),
                }
            )
        transactions = pd.DataFrame(transaction_rows, columns=_TRANSACTION_COLUMNS)
        if not transactions.empty:
            transactions = (
                transactions.drop_duplicates(["chain_id", "vault_address", "tx_hash", "log_index"])
                .sort_values(["time", "vault_address", "tx_hash", "log_index"], kind="stable")
                .reset_index(drop=True)
            )
        self._data = MorphoVaultV2History(
            vaults=listings,
            allocations=allocations,
            transactions=transactions,
            coverage=_coverage(allocations),
        )

    def read(self, with_run: bool = False) -> MorphoVaultV2History:
        """Run the loader or read its cached :class:`MorphoVaultV2History`."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if not isinstance(self._data, MorphoVaultV2History):
            raise TypeError(f"cached Morpho Vault V2 data has unexpected type {type(self._data).__name__}")
        return self._data
