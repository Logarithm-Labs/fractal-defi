"""Morpho Vault V2 discovery and allocation/depositor activity.

The loader uses the public Morpho GraphQL API to discover Vault V2 vaults
(optionally restricted to chains, vault assets and a current TVL floor) and to
load their allocation events and vault transactions over an inclusive UTC time
window.

Market mapping. A ``MorphoMarketV1`` adapter tags every ``Allocate`` /
``Deallocate`` event with the ids ``keccak256(abi.encode("this", adapter))``,
``keccak256(abi.encode("collateralToken", collateral))`` and
``keccak256(abi.encode("this/marketParams", adapter, marketParams))`` with
``marketParams = (loanToken, collateralToken, oracle, irm, lltv)``. The loader
fetches the ``MarketParams`` (:func:`fetch_market_params`) of every Morpho
Blue market whose loan token is a loaded vault's asset and maps a row to the market whose market-params id is
among the row's ``ids`` (``map_method="id"``). The match is a hash of the
event's own ids, so it holds for markets the adapter has since exited, and it
needs no ABI library: the fixed-layout encodings are built here and hashed with
:func:`fractal.loaders.rpc.keccak256`. Rows of other adapter types, or without
exactly one match, keep ``market_id=None`` (``map_method="none"``) and are
counted in the attached coverage summary.

Amounts are returned in native vault-asset units. ``assets_delta_native`` is
positive for allocations/deposits, negative for deallocations/withdrawals,
and zero for share transfers. ``change_native`` preserves the allocation
API's interest-inclusive adapter change without using it as a flow sign.

:func:`classify_allocations` labels allocation rows by who triggered them:
``depositor_driven`` when the same transaction carries a deposit/withdrawal of
the same vault, ``allocator`` when sent by a current vault allocator, else
``other``. :func:`hour_end`, :func:`hourly_supply_changes` and
:func:`attribute_supply_changes` attribute a market's hourly supply change to
the source (e.g. that label) whose signed flows dominate the hour.
"""
import hashlib
import warnings
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc, utcnow
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.morpho import MorphoLoaderException, graphql_query
from fractal.loaders.rpc import keccak256

__all__ = [
    "MorphoMarketParams",
    "MorphoVaultV2Coverage",
    "MorphoVaultV2History",
    "MorphoVaultV2Listing",
    "MorphoVaultV2Loader",
    "attribute_supply_changes",
    "classify_allocations",
    "fetch_market_params",
    "hour_end",
    "hourly_supply_changes",
    "morpho_market_id",
    "vault_v2_market_params_id",
]

_DEFAULT_TVL_FLOOR_USD = 1_000_000.0
_ALLOCATION_SLICE_SECONDS = 7 * 86_400
_TRANSACTION_SLICE_SECONDS = 86_400
_EVENT_SKIP_CAP = 10_000
_MARKET_PAGE_SIZE = 1000
_CACHE_SCHEMA = "v3"
_ZERO_ADDRESS = "0x" + "00" * 20

_VAULTS_QUERY = """
query VaultV2Vaults($first: Int!, $skip: Int!, $where: VaultV2sFilters) {
  vaultV2s(first: $first, skip: $skip, where: $where) {
    items {
      address chain { id } asset { address symbol decimals } totalAssetsUsd
      liquidityAdapter { address }
      allocators { allocator { address } }
    }
    pageInfo { countTotal }
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

_MARKETS_QUERY = """
query VaultV2Markets($first: Int!, $skip: Int!, $where: MarketFilters) {
  markets(first: $first, skip: $skip, where: $where) {
    items { marketId lltv irmAddress oracleAddress loanAsset { address } collateralAsset { address } }
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
_ALLOCATION_SIGNS = {"Allocate": 1.0, "Deallocate": -1.0}
_TRANSACTION_SIGNS = {"Deposit": 1.0, "Withdraw": -1.0, "Transfer": 0.0}


@dataclass(frozen=True)
class MorphoVaultV2Listing:
    """One Vault V2 vault and its current asset/configuration.

    Attributes:
        chain_id: EVM chain id.
        vault_address: Vault V2 contract address.
        asset_address: Underlying vault asset address.
        asset_symbol: Symbol reported by the Morpho API.
        asset_decimals: Decimal precision used to scale event amounts, or
            ``None`` when the API omits it (amounts are then ``None``).
        total_assets_usd: Current API TVL in USD, or ``None`` when absent.
        liquidity_adapter_address: Current liquidity adapter, if configured.
        allocators: Current allocator addresses reported by the API.
    """

    chain_id: int
    vault_address: str
    asset_address: str
    asset_symbol: str
    asset_decimals: int | None
    total_assets_usd: float | None
    liquidity_adapter_address: str | None
    allocators: tuple[str, ...]


@dataclass(frozen=True)
class MorphoVaultV2Coverage:
    """Coverage of allocation-to-Morpho-Blue market mapping.

    ``mapped_assets_fraction`` weights rows by absolute native-asset flow;
    fractions are ``None`` when there are no allocation rows or no volume.
    Native amounts are summed across vaults as-is, so the volume figures are
    only meaningful for a single vault asset.
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


class MorphoMarketParams(NamedTuple):
    """Morpho Blue ``MarketParams``: lowercase addresses (zero address when absent) and the WAD ``lltv``.

    ``morpho_market_id(*params)`` returns the market's id.
    """

    loan_token: str
    collateral_token: str
    oracle: str
    irm: str
    lltv: int


# ------------------------------------------------------------------ ids
def _word(value: int) -> bytes:
    return int(value).to_bytes(32, "big")


def _address_word(address: str) -> bytes:
    return _word(int(address or _ZERO_ADDRESS, 16))


def _string_tail(text: str) -> bytes:
    data = text.encode()
    return _word(len(data)) + data + b"\0" * (-len(data) % 32)


def _hex_keccak(data: bytes) -> str:
    return "0x" + keccak256(data).hex()


def _collateral_id(token: str) -> str:
    """``keccak256(abi.encode("collateralToken", token))``."""
    return _hex_keccak(_word(64) + _address_word(token) + _string_tail("collateralToken"))


def _market_params_words(loan_token: str, collateral_token: str, oracle: str, irm: str, lltv: int) -> bytes:
    """``abi.encode(marketParams)``: five static words (four addresses, the WAD ``lltv``)."""
    return b"".join(
        [
            _address_word(loan_token),
            _address_word(collateral_token),
            _address_word(oracle),
            _address_word(irm),
            _word(lltv),
        ]
    )


def morpho_market_id(loan_token: str, collateral_token: str, oracle: str, irm: str, lltv_wad: int) -> str:
    """Morpho Blue market id: ``keccak256(abi.encode(MarketParams))``.

    ``MarketParams = (loanToken, collateralToken, oracle, irm, lltv)`` with
    addresses as ``0x`` hex (case-insensitive) and ``lltv_wad`` the exact WAD
    integer (e.g. ``860000000000000000`` for 86%), not a float fraction.
    Returns the lowercase ``0x`` hex id the Morpho API reports as ``marketId``.
    """
    return _hex_keccak(_market_params_words(loan_token, collateral_token, oracle, irm, lltv_wad))


def vault_v2_market_params_id(
    adapter: str,
    loan_token: str,
    collateral_token: str,
    oracle: str,
    irm: str,
    lltv: int,
) -> str:
    """Id a ``MorphoMarketV1`` adapter emits for one Blue market.

    ``keccak256(abi.encode("this/marketParams", adapter, marketParams))`` with
    ``marketParams = (loanToken, collateralToken, oracle, irm, lltv)``; ``lltv``
    is the exact WAD integer. Returns a lowercase ``0x`` hex string.
    """
    params = _market_params_words(loan_token, collateral_token, oracle, irm, lltv)
    head = _word(7 * 32) + _address_word(adapter) + params
    return _hex_keccak(head + _string_tail("this/marketParams"))


# ------------------------------------------------------------------ paging
def _lower(value: Any) -> str:
    return str(value or "").lower()


def _raw_int(value: Any) -> int:
    """Exact integer from a GraphQL BigInt sent as an int, a digit string or scientific notation."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, bool):
        try:
            number = Decimal(str(value).strip())
        except InvalidOperation:
            number = None
        if number is not None and number.is_finite() and number == number.to_integral_value():
            return int(number)
    raise MorphoLoaderException(f"Morpho GraphQL: expected an integer amount, got {value!r}")


def _scale(value: Any, decimals: int | None) -> float | None:
    if value is None or decimals is None:
        return None
    return _raw_int(value) / 10**decimals


def _event_key(item: dict[str, Any]) -> tuple[str, int, str]:
    return (
        _lower(item.get("txHash")),
        int(item.get("logIndex") or 0),
        _lower((item.get("vault") or {}).get("address")),
    )


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
) -> list[dict[str, Any]]:
    """Skip-paginate a listing until a short page, a page without new rows, or ``countTotal`` rows."""
    items: list[dict[str, Any]] = []
    seen: set[Hashable] = set()
    skip = 0
    while True:
        page, total = _query_page(http, query, root, variables, page_size, skip)
        new_items = []
        for item in page:
            key = identity(item)
            if key not in seen:
                seen.add(key)
                new_items.append(item)
        items.extend(new_items)
        skip += page_size
        done = skip >= total if total is not None else len(page) < page_size
        if done or not new_items:
            return items


def _event_slice(
    http: HttpClient,
    query: str,
    root: str,
    variables: Callable[[int, int], dict[str, Any]],
    page_size: int,
    lo: int,
    hi: int,
    label: str,
) -> list[dict[str, Any]]:
    """Events in ``[lo, hi]``, bisecting the window until its first page holds ``countTotal`` rows.

    ``skip`` pages of Morpho API event queries can come back short of ``first``
    while more rows remain, so offset paging is used only for a single second
    that holds more events than one page; it stops on ``countTotal`` (on a
    short page when the API omits it) and a result short of ``countTotal``
    emits a ``UserWarning`` instead of failing.
    """
    bounded = variables(lo, hi)
    page, total = _query_page(http, query, root, bounded, page_size, 0)
    if total is not None and total > len(page) and lo < hi:
        mid = (lo + hi) // 2
        return _event_slice(http, query, root, variables, page_size, lo, mid, label) + _event_slice(
            http, query, root, variables, page_size, mid + 1, hi, label
        )
    if total is not None and total > _EVENT_SKIP_CAP:
        raise MorphoLoaderException(f"{total} {label} in second {lo}; cannot page past the API skip cap")
    rows: list[dict[str, Any]] = []
    seen: set[Hashable] = set()
    skip = 0
    while page:
        new_rows = [item for item in page if _event_key(item) not in seen]
        if not new_rows:
            break
        rows.extend(new_rows)
        seen.update(_event_key(item) for item in new_rows)
        skip += page_size
        if (len(page) < page_size if total is None else skip >= total) or skip > _EVENT_SKIP_CAP:
            break
        page, _ = _query_page(http, query, root, bounded, page_size, skip)
    if total is not None and len(rows) < total:
        warnings.warn(
            f"MorphoVaultV2Loader: {label} in [{lo}, {hi}] returned {len(rows)} distinct events of countTotal {total}",
            stacklevel=2,
        )
    return rows


def _allocation_slice(
    http: HttpClient,
    variables: dict[str, Any],
    page_size: int,
    lo: int,
    hi: int,
) -> list[dict[str, Any]]:
    """Allocations of one vault in ``[lo, hi]``."""
    return _event_slice(
        http,
        _ALLOCATIONS_QUERY,
        "vaultV2AllocationTransactions",
        lambda a, b: {**variables, "where": {"timestamp_gte": a, "timestamp_lte": b}},
        page_size,
        lo,
        hi,
        f"Vault V2 allocations of {variables['vaultAddress']}",
    )


def _transaction_slice(
    http: HttpClient,
    where: dict[str, Any],
    page_size: int,
    lo: int,
    hi: int,
) -> list[dict[str, Any]]:
    """Vault transactions of a vault batch in ``[lo, hi]``."""
    return _event_slice(
        http,
        _TRANSACTIONS_QUERY,
        "vaultV2transactions",
        lambda a, b: {"where": {**where, "timestamp_gte": a, "timestamp_lte": b}},
        page_size,
        lo,
        hi,
        "Vault V2 transactions",
    )


def _slices(start: int, end: int, width: int) -> list[tuple[int, int]]:
    return [(lo, min(lo + width - 1, end)) for lo in range(start, end + 1, width)]


# ------------------------------------------------------------------ parsing
def _listing(item: dict[str, Any]) -> MorphoVaultV2Listing:
    asset = item.get("asset") or {}
    total_assets_usd = item.get("totalAssetsUsd")
    decimals = asset.get("decimals")
    allocators = {
        _lower((entry.get("allocator") or {}).get("address"))
        for entry in item.get("allocators") or []
        if (entry.get("allocator") or {}).get("address")
    }
    return MorphoVaultV2Listing(
        chain_id=int((item.get("chain") or {})["id"]),
        vault_address=_lower(item["address"]),
        asset_address=_lower(asset.get("address")),
        asset_symbol=str(asset.get("symbol") or ""),
        asset_decimals=None if decimals is None else int(decimals),
        total_assets_usd=None if total_assets_usd is None else float(total_assets_usd),
        liquidity_adapter_address=_lower((item.get("liquidityAdapter") or {}).get("address")) or None,
        allocators=tuple(sorted(allocators)),
    )


def _market_params(item: dict[str, Any]) -> MorphoMarketParams:
    return MorphoMarketParams(
        loan_token=_lower((item.get("loanAsset") or {}).get("address")) or _ZERO_ADDRESS,
        collateral_token=_lower((item.get("collateralAsset") or {}).get("address")) or _ZERO_ADDRESS,
        oracle=_lower(item.get("oracleAddress")) or _ZERO_ADDRESS,
        irm=_lower(item.get("irmAddress")) or _ZERO_ADDRESS,
        lltv=_raw_int(item.get("lltv") or 0),
    )


def fetch_market_params(
    chain_id: int,
    *,
    market_ids: Iterable[str] = (),
    loan_addresses: Iterable[str] = (),
    http: HttpClient | None = None,
    batch_size: int = 100,
) -> dict[str, MorphoMarketParams]:
    """Exact Morpho Blue ``MarketParams`` from the public Morpho API.

    Returns ``{market_id: MorphoMarketParams}`` for the markets in
    ``market_ids`` (queried in batches of ``batch_size``) and every market on
    ``chain_id`` whose loan token is in ``loan_addresses``, listed or not.
    Ids and addresses are matched case-insensitively and returned lowercase.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    client = http or HttpClient()
    ids = sorted({_lower(market_id) for market_id in market_ids})
    wheres = [
        {"chainId_in": [int(chain_id)], "uniqueKey_in": ids[i:i + batch_size]} for i in range(0, len(ids), batch_size)
    ]
    loans = sorted({_lower(address) for address in loan_addresses})
    if loans:
        wheres.append({"chainId_in": [int(chain_id)], "loanAssetAddress_in": loans})
    out: dict[str, MorphoMarketParams] = {}
    for where in wheres:
        for item in _paged_items(
            client,
            _MARKETS_QUERY,
            "markets",
            {"where": where},
            _MARKET_PAGE_SIZE,
            lambda item: _lower(item.get("marketId")),
        ):
            out[_lower(item.get("marketId"))] = _market_params(item)
    return out


class _MarketIndex:
    """Resolve allocation ``ids`` to a Blue market id through the market-params id."""

    def __init__(self, markets: dict[int, dict[str, MorphoMarketParams]]) -> None:
        self._by_collateral: dict[tuple[int, str, str], list[tuple[str, MorphoMarketParams]]] = {}
        self._params_ids: dict[tuple[str, int, str], str] = {}
        collateral_ids: dict[str, str] = {}
        for chain_id, chain_markets in markets.items():
            for market_id, params in chain_markets.items():
                token = params.collateral_token
                if token not in collateral_ids:
                    collateral_ids[token] = _collateral_id(token)
                key = (int(chain_id), params.loan_token, collateral_ids[token])
                self._by_collateral.setdefault(key, []).append((market_id, params))

    def resolve(self, chain_id: int, loan_token: str, adapter: str, ids: Iterable[str]) -> str | None:
        """The unique market whose market-params id for ``adapter`` is in ``ids``, else ``None``."""
        seen = set(ids)
        hits = set()
        for value in seen:
            for market_id, params in self._by_collateral.get((chain_id, loan_token, value), []):
                cache_key = (adapter, chain_id, market_id)
                if cache_key not in self._params_ids:
                    self._params_ids[cache_key] = vault_v2_market_params_id(adapter, *params)
                if self._params_ids[cache_key] in seen:
                    hits.add(market_id)
        return hits.pop() if len(hits) == 1 else None


def classify_allocations(
    allocations: pd.DataFrame,
    transactions: pd.DataFrame,
    vaults: Iterable[MorphoVaultV2Listing] = (),
    *,
    join_types: Iterable[str] = ("Deposit", "Withdraw"),
) -> pd.Series:
    """Label each allocation row ``depositor_driven``, ``allocator`` or ``other``.

    A row is ``depositor_driven`` when ``transactions`` holds a row of a
    ``join_types`` type with the same ``chain_id``, ``vault_address`` and
    ``tx_hash`` (the vault moved liquidity inside a depositor's own deposit or
    withdrawal, e.g. through its liquidity adapter). Otherwise it is
    ``allocator`` when ``sender_address`` is among that vault's
    :attr:`MorphoVaultV2Listing.allocators` (the *current* allocator set, so
    a since-removed allocator falls to ``other``), else ``other``.

    Args:
        allocations: :attr:`MorphoVaultV2History.allocations`.
        transactions: :attr:`MorphoVaultV2History.transactions`.
        vaults: Listings providing the allocator sets, normally
            :attr:`MorphoVaultV2History.vaults`.
        join_types: Transaction types that mark a depositor transaction.

    Returns:
        A string series aligned with ``allocations.index``, named ``"class"``.
    """
    joined = transactions[transactions["type"].isin(tuple(join_types))]
    depositor_txs = set(
        zip(
            joined["chain_id"].astype(int),
            joined["vault_address"].map(_lower),
            joined["tx_hash"].map(_lower),
            strict=True,
        )
    )
    allocators = {(vault.chain_id, vault.vault_address): set(vault.allocators) for vault in vaults}
    labels = []
    for chain_id, vault_address, tx_hash, sender in zip(
        allocations["chain_id"].astype(int),
        allocations["vault_address"].map(_lower),
        allocations["tx_hash"].map(_lower),
        allocations["sender_address"].map(_lower),
        strict=True,
    ):
        if (chain_id, vault_address, tx_hash) in depositor_txs:
            labels.append("depositor_driven")
        elif sender in allocators.get((chain_id, vault_address), set()):
            labels.append("allocator")
        else:
            labels.append("other")
    return pd.Series(labels, index=allocations.index, name="class", dtype=object)


_HOUR = pd.Timedelta(hours=1)


def hour_end(times: pd.Series) -> pd.Series:
    """Label UTC event times with the end of their market hour.

    An event at ``t`` belongs to the hour ``(T - 1h, T]`` labelled ``T``, so
    it is compared with the change between the snapshots at ``T - 1h`` and
    ``T``; an event exactly on the hour keeps its own label.
    """
    return pd.to_datetime(times, utc=True).dt.as_unit("ns").dt.ceil("h")


def hourly_supply_changes(supply: pd.DataFrame) -> pd.DataFrame:
    """Hour-on-hour supply change per market from on-the-hour snapshots.

    Args:
        supply: Columns ``chain_id``, ``market_id``, ``time`` (UTC, on the
            hour) and ``supply_assets`` (any consistent unit).

    Returns:
        Columns ``chain_id``, ``market_id``, ``time``, ``supply_prev`` (the
        snapshot at ``time - 1h``), ``net_change`` and ``rel_change``
        (``net_change / supply_prev``, ``NaN`` when ``supply_prev`` is not
        positive). Hours whose previous snapshot is missing are dropped,
        never filled.
    """
    base = supply[["chain_id", "market_id", "time", "supply_assets"]].dropna(subset=["supply_assets"])
    base = base.assign(time=pd.to_datetime(base["time"], utc=True).dt.as_unit("ns"))
    prev = base.assign(time=base["time"] + _HOUR).rename(columns={"supply_assets": "supply_prev"})
    out = base.merge(prev, on=["chain_id", "market_id", "time"], how="inner")
    out["net_change"] = out["supply_assets"] - out["supply_prev"]
    out["rel_change"] = (out["net_change"] / out["supply_prev"]).where(out["supply_prev"] > 0)
    columns = ["chain_id", "market_id", "time", "supply_prev", "net_change", "rel_change"]
    return out[columns].sort_values(["time", "chain_id", "market_id"], kind="stable").reset_index(drop=True)


def attribute_supply_changes(
    changes: pd.DataFrame,
    flows: pd.DataFrame,
    *,
    min_share: float,
    source_column: str = "source",
    flow_column: str = "assets_delta_native",
) -> pd.DataFrame:
    """Attribute each market-hour supply change to its dominant flow source.

    ``flows`` rows with a ``market_id`` are bucketed with :func:`hour_end`
    and summed per ``(chain_id, market_id, hour, source)``. For every row of
    ``changes`` each source's hourly flow is projected on the sign of
    ``net_change``; the source with the largest projection is assigned when
    that projection is positive and at least ``min_share * |net_change|``.
    No threshold is assumed: ``min_share`` must be chosen by the caller (for
    example ``0.5`` for a strict-majority reading). Selecting which hours
    count as events (relative-size cut-offs, de-duplication) is left to the
    caller, e.g. by filtering ``changes`` on ``rel_change`` first.

    Args:
        changes: Columns ``chain_id``, ``market_id``, ``time`` (hour end)
            and ``net_change``, e.g. :func:`hourly_supply_changes`.
        flows: Columns ``chain_id``, ``market_id``, ``time`` (event time),
            ``source_column`` and ``flow_column`` (signed, in the same unit
            as ``net_change``), e.g. Vault V2 allocations with
            :func:`classify_allocations` labels in ``source``.
        min_share: Minimum projected share of ``|net_change|``; must be > 0.
        source_column / flow_column: Names of the label and flow columns.

    Returns:
        ``changes`` plus one ``flow_<source>`` column per source (``0.0``
        when absent), ``source`` (``None`` when no source qualifies, or when
        ``net_change`` is zero) and ``source_share`` (the assigned
        projection over ``|net_change|``, ``NaN`` when unassigned).
    """
    if not min_share > 0:
        raise ValueError(f"min_share must be > 0, got {min_share}")
    keys = ["chain_id", "market_id", "time"]
    mapped = flows[flows["market_id"].notna()]
    hourly = (
        mapped.assign(time=hour_end(mapped["time"]), _flow=pd.to_numeric(mapped[flow_column], errors="coerce"))
        .groupby([*keys, source_column])["_flow"]
        .sum()
        .unstack(source_column, fill_value=0.0)
    )
    sources = sorted(str(name) for name in hourly.columns)
    hourly = hourly.rename(columns=lambda name: f"flow_{name}").reset_index()
    out = changes.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True).dt.as_unit("ns")
    if sources:
        out = out.merge(hourly, on=keys, how="left")
    flow_columns = [f"flow_{name}" for name in sources]
    for column in flow_columns:
        out[column] = out[column].astype(float).fillna(0.0)
    out["source"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    out["source_share"] = np.nan
    if not sources or out.empty:
        return out
    net = out["net_change"].astype(float).to_numpy()
    projected = out[flow_columns].to_numpy() * np.sign(net)[:, None]
    best = projected.argmax(axis=1)
    best_flow = projected[np.arange(len(out)), best]
    assigned = (best_flow > 0) & (best_flow >= min_share * np.abs(net))
    out.loc[assigned, "source"] = np.array(sources, dtype=object)[best[assigned]]
    out.loc[assigned, "source_share"] = best_flow[assigned] / np.abs(net[assigned])
    return out


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


def _event_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """Typed, de-duplicated and time-ordered event frame (also when empty)."""
    frame = pd.DataFrame(rows, columns=columns).astype({"time": "int64", "chain_id": "int64", "log_index": "int64"})
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    return (
        frame.drop_duplicates(["chain_id", "vault_address", "tx_hash", "log_index"])
        .sort_values(["time", "chain_id", "vault_address", "tx_hash", "log_index"], kind="stable")
        .reset_index(drop=True)
    )


class MorphoVaultV2Loader(Loader):
    """Load Vault V2 listings, allocation events and depositor transactions.

    Event windows are bisected in time until each part fits in one page, and
    a part that returns fewer distinct events than its ``countTotal`` emits a
    ``UserWarning``. BigInt amounts are parsed
    exactly, including scientific notation such as ``"1.5e+21"``.

    Args:
        start_time / end_time: Inclusive UTC event window. ``end_time``
            defaults to construction time.
        chain_ids: Chains to scan; ``None`` scans every chain the API serves.
        asset_addresses: Vault asset (loan token) addresses to keep,
            filtered server-side; ``None`` keeps every asset.
        tvl_floor_usd: Minimum current vault TVL requested server-side;
            ``0`` disables the filter (vaults without a USD TVL are then
            kept). Defaults to ``1_000_000`` USD.
        vault_page_size: Number of vaults requested per discovery page.
        event_page_size: Number of allocation/transaction rows requested per
            page; the Morpho API accepts at most 1,000.
        vault_batch_size: Number of vaults per depositor-transaction query.
        loader_type: Cache format. Only ``PICKLE`` is supported because the
            result contains dataclasses and multiple data frames.
        http: Injectable HTTP client for offline tests.
    """

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime | None = None,
        *,
        chain_ids: Iterable[int] | None = None,
        asset_addresses: Iterable[str] | None = None,
        tvl_floor_usd: float = _DEFAULT_TVL_FLOOR_USD,
        vault_page_size: int = 200,
        event_page_size: int = 1000,
        vault_batch_size: int = 50,
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
        if vault_batch_size <= 0:
            raise ValueError(f"vault_batch_size must be > 0, got {vault_batch_size}")
        self.chain_ids = None if chain_ids is None else tuple(sorted({int(chain) for chain in chain_ids}))
        if self.chain_ids == ():
            raise ValueError("chain_ids must not be empty; pass None to scan every chain")
        self.asset_addresses = (
            None if asset_addresses is None else tuple(sorted({_lower(address) for address in asset_addresses}))
        )
        if self.asset_addresses == ():
            raise ValueError("asset_addresses must not be empty; pass None to keep every asset")
        self.start_time = to_utc(start_time)
        self.end_time = to_utc(end_time) if end_time is not None else utcnow()
        if self.end_time < self.start_time:
            raise ValueError(f"end_time {self.end_time} precedes start_time {self.start_time}")
        self.tvl_floor_usd = float(tvl_floor_usd)
        self.vault_page_size = int(vault_page_size)
        self.event_page_size = int(event_page_size)
        self.vault_batch_size = int(vault_batch_size)
        self._http = http or HttpClient()

    def _cache_key(self) -> str:
        chains = "all" if self.chain_ids is None else "_".join(str(chain) for chain in self.chain_ids)
        assets = (
            "all"
            if self.asset_addresses is None
            else hashlib.sha256(",".join(self.asset_addresses).encode()).hexdigest()[:16]
        )
        return (
            f"schema-{_CACHE_SCHEMA}-chains-{chains}-assets-{assets}-start-{to_seconds(self.start_time)}"
            f"-end-{to_seconds(self.end_time)}-tvl-{self.tvl_floor_usd:g}"
            f"-vault-page-{self.vault_page_size}-event-page-{self.event_page_size}"
        )

    def list_vaults(self) -> tuple[MorphoVaultV2Listing, ...]:
        """Current vault listings matching the filters, without events (one paged query, uncached)."""
        where: dict[str, Any] = {}
        if self.chain_ids is not None:
            where["chainId_in"] = list(self.chain_ids)
        if self.asset_addresses is not None:
            where["assetAddress_in"] = list(self.asset_addresses)
        if self.tvl_floor_usd > 0:
            where["totalAssetsUsd_gte"] = self.tvl_floor_usd
        items = _paged_items(
            self._http,
            _VAULTS_QUERY,
            "vaultV2s",
            {"where": where},
            self.vault_page_size,
            lambda item: (int((item.get("chain") or {}).get("id") or 0), _lower(item.get("address"))),
        )
        listings = {(listing.chain_id, listing.vault_address): listing for listing in map(_listing, items)}
        return tuple(listings[key] for key in sorted(listings))

    def extract(self) -> None:
        """Fetch vaults, allocations, vault transactions and the candidate markets' params."""
        vaults = self.list_vaults()
        start = to_seconds(self.start_time)
        end = to_seconds(self.end_time)
        allocations: list[dict[str, Any]] = []
        for vault in vaults:
            variables = {"chainId": vault.chain_id, "vaultAddress": vault.vault_address}
            for lo, hi in _slices(start, end, _ALLOCATION_SLICE_SECONDS):
                allocations.extend(
                    {"chain_id": vault.chain_id, "vault_address": vault.vault_address, **event}
                    for event in _allocation_slice(self._http, variables, self.event_page_size, lo, hi)
                )
        by_chain: dict[int, list[MorphoVaultV2Listing]] = {}
        for vault in vaults:
            by_chain.setdefault(vault.chain_id, []).append(vault)
        transactions: list[dict[str, Any]] = []
        markets: dict[int, dict[str, MorphoMarketParams]] = {}
        for chain_id, chain_vaults in sorted(by_chain.items()):
            for i in range(0, len(chain_vaults), self.vault_batch_size):
                where = {
                    "chainId_in": [chain_id],
                    "vaultAddress_in": [vault.vault_address for vault in chain_vaults[i:i + self.vault_batch_size]],
                    "type_in": list(_TRANSACTION_SIGNS),
                }
                for lo, hi in _slices(start, end, _TRANSACTION_SLICE_SECONDS):
                    transactions.extend(
                        {"chain_id": chain_id, **event}
                        for event in _transaction_slice(self._http, where, self.event_page_size, lo, hi)
                    )
            if any(row["chain_id"] == chain_id for row in allocations):
                markets[chain_id] = fetch_market_params(
                    chain_id,
                    loan_addresses={vault.asset_address for vault in chain_vaults},
                    http=self._http,
                )
        self._data = {
            "vaults": vaults,
            "allocations": allocations,
            "transactions": transactions,
            "markets": markets,
        }

    def transform(self) -> None:
        """Scale amounts, sign flows and attach id-matched Morpho Blue market ids."""
        raw = self._data or {}
        listings: tuple[MorphoVaultV2Listing, ...] = tuple(raw.get("vaults", ()))
        by_vault = {(listing.chain_id, listing.vault_address): listing for listing in listings}
        index = _MarketIndex(raw.get("markets", {}))

        allocation_rows = []
        for item in raw.get("allocations", []):
            listing = by_vault[(int(item["chain_id"]), _lower(item.get("vault_address")))]
            event_type = str(item.get("type") or "")
            if event_type not in _ALLOCATION_SIGNS:
                raise MorphoLoaderException(f"unsupported Vault V2 allocation type {event_type!r}")
            assets_native = _scale(item.get("assets"), listing.asset_decimals)
            adapter_address = _lower(item.get("adapter"))
            ids = tuple(_lower(value) for value in item.get("ids") or [])
            market_id = index.resolve(listing.chain_id, listing.asset_address, adapter_address, ids)
            allocation_rows.append(
                {
                    "time": int(item["timestamp"]),
                    "chain_id": listing.chain_id,
                    "vault_address": listing.vault_address,
                    "asset_address": listing.asset_address,
                    "asset_symbol": listing.asset_symbol,
                    "type": event_type,
                    "adapter_address": adapter_address,
                    "sender_address": _lower(item.get("sender")),
                    "assets_native": assets_native,
                    "assets_delta_native": (
                        None if assets_native is None else _ALLOCATION_SIGNS[event_type] * assets_native
                    ),
                    "change_native": _scale(item.get("change"), listing.asset_decimals),
                    "market_id": market_id,
                    "map_method": "id" if market_id is not None else "none",
                    "ids": ids,
                    "tx_hash": _lower(item.get("txHash")),
                    "log_index": int(item.get("logIndex") or 0),
                }
            )
        allocations = _event_frame(allocation_rows, _ALLOCATION_COLUMNS)

        transaction_rows = []
        for item in raw.get("transactions", []):
            vault = item.get("vault") or {}
            listing = by_vault.get(
                (int((vault.get("chain") or {}).get("id") or item["chain_id"]), _lower(vault.get("address")))
            )
            if listing is None:
                continue
            event_type = str(item.get("type") or "")
            if event_type not in _TRANSACTION_SIGNS:
                raise MorphoLoaderException(f"unsupported Vault V2 transaction type {event_type!r}")
            assets_native = _scale(item.get("assets"), listing.asset_decimals)
            transaction_rows.append(
                {
                    "time": int(item["timestamp"]),
                    "chain_id": listing.chain_id,
                    "vault_address": listing.vault_address,
                    "asset_address": listing.asset_address,
                    "asset_symbol": listing.asset_symbol,
                    "type": event_type,
                    "assets_native": assets_native,
                    "assets_delta_native": (
                        None if assets_native is None else _TRANSACTION_SIGNS[event_type] * assets_native
                    ),
                    "tx_hash": _lower(item.get("txHash")),
                    "log_index": int(item.get("logIndex") or 0),
                }
            )
        self._data = MorphoVaultV2History(
            vaults=listings,
            allocations=allocations,
            transactions=_event_frame(transaction_rows, _TRANSACTION_COLUMNS),
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
